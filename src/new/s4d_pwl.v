`timescale 1ns / 1ps
`default_nettype none
//=====================================================================
// s4d_pwl  --  funzione approssimata a tratti lineari (piecewise linear)
//
//   Usata per GELU (in s4d_layer) e sigmoide (in s4d_mixing_glu).
//
//   Intervallo coperto: [-2^XH, +2^XH) = [-8, 8) con XH = 3, diviso in
//   NSEG = 256 segmenti uguali (larghezza 1/16).
//     xs   = x + 2^XH                  (x spostato in [0, 2^(XH+1)))
//     idx  = bit alti di xs            -> quale segmento
//     frac = FRACW bit bassi di xs     -> posizione dentro il segmento
//     y    = base[idx] + slope[idx] * frac / 2^FRACW   (arrotondato)
//
//   LUT (LUT_FILE): NSEG parole da 2*W_OUT bit = {slope, base}, entrambe
//   nel formato d'uscita Q.F_OUT.
//
//   Fuori intervallo:
//     x <  -2^XH : y = Y_LO_Q
//     x >= +2^XH : y = x        se MODE_HI = 1 (GELU: gelu(x) ~ x)
//                  y = Y_HI_Q   se MODE_HI = 0 (sigmoide: 1.0)
//
//   Pipeline a LAT = 5 stadi, al massimo UN'operazione complessa per
//   ciclo (timing, 2026-10-15; prima 3 stadi con moltiplicazione, due
//   somme, confronti e mux nell'ultimo):
//     1 x + 2^XH e confronti con i limiti (in parallelo) -> idx, frac
//     2 lettura della LUT
//     3 prodotto slope * frac                       (registro M del DSP)
//     4 somma prodotto + (base << FRACW) + mezzo LSB (registro P del DSP)
//     5 shift (solo fili), saturazione, mux fuori intervallo -> uscita
//   Lo stadio 4 usa l'identita' esatta
//     base + ((prod + rnd) >>> FRACW) = (prod + rnd + base*2^FRACW) >>> FRACW
//   e il termine (base << FRACW) + rnd e' una concatenazione (rnd < 2^FRACW),
//   quindi un solo sommatore. Nessun overflow: |prod + rnd| < 2^(W_OUT-1+FRACW)
//   e la somma con base*2^FRACW sta in PW3 = W_OUT+FRACW+1 bit, come prima.
//
//   in_tag / out_tag: TAG_W bit che attraversano la pipeline insieme al
//   dato, con lo stesso enable (es. "ultimo canale del frame", oppure il
//   valore a del GLU). Chi usa il modulo non deve conoscerne la latenza.
//=====================================================================

module s4d_pwl #(
    parameter integer W_IN      = 16,   // ingresso (Q.F_IN)
    parameter integer F_IN      = 11,
    parameter integer W_OUT     = 16,   // uscita (Q.F_OUT)
    parameter integer F_OUT     = 11,
    parameter integer NSEG      = 256,  // segmenti
    parameter integer LOG2_NSEG = 8,
    parameter integer XH        = 3,    // semi-ampiezza dell'intervallo = 2^XH
    parameter integer MODE_HI   = 1,    // 1: sopra l'intervallo y = x; 0: y = Y_HI_Q
    parameter integer Y_LO_Q    = 0,    // valore sotto l'intervallo (Q.F_OUT)
    parameter integer Y_HI_Q    = 0,    // valore sopra l'intervallo se MODE_HI = 0
    parameter integer TAG_W     = 1,    // bit del tag che viaggia con il dato
    parameter         LUT_FILE  = "gelu_lut.mem"
)(
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire                    in_valid,
    output wire                    in_ready,
    input  wire signed [W_IN-1:0]  in_data,
    input  wire [TAG_W-1:0]        in_tag,     // accompagna in_data
    input  wire                    out_ready,
    output reg                     out_valid,
    output reg  signed [W_OUT-1:0] out_data,
    output reg  [TAG_W-1:0]        out_tag     // accompagna out_data
);

    // Pipeline STALLABILE globalmente: out_ready basso congela tutti gli
    // stadi insieme. Stesso schema di s4d_affine.
    // NOTA: in_ready e' combinatorio da out_ready -> percorso passante.
    wire en = out_ready;
    assign in_ready = out_ready;

    localparam integer FRACW  = F_IN + XH + 1 - LOG2_NSEG;   // bit della posizione nel segmento
    localparam integer XW     = W_IN + 1;                    // un bit in piu' per x + offset
    // offset = 2^XH in Q.F_IN
    localparam signed [XW-1:0] OFFS  = {{(XW-XH-1-F_IN){1'b0}}, 1'b1, {(XH+F_IN){1'b0}}};
    localparam signed [XW-1:0] X_LO  = -OFFS;                // limite inferiore -2^XH
    localparam signed [XW-1:0] X_HI  =  OFFS;                // limite superiore +2^XH

    localparam integer PW3 = W_OUT + FRACW + 1;              // larghezza di prodotto e somma

    initial if (FRACW < 1) begin
        $display("s4d_pwl: FRACW=%0d, servono almeno 1 bit di posizione nel segmento", FRACW);
        $finish;
    end

    // LUT dei segmenti: {slope, base}
    reg [2*W_OUT-1:0] lut [0:NSEG-1];
    initial $readmemh(LUT_FILE, lut);

    wire signed [XW-1:0] x   = {in_data[W_IN-1], in_data};   // x esteso di un bit
    wire signed [XW-1:0] xs  = x + OFFS;                     // x spostato in [0, 2^(XH+1))

    // valid dei 4 stadi interni (lo stadio 5 e' out_valid)
    reg v1, v2, v3, v4;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v1 <= 1'b0; v2 <= 1'b0; v3 <= 1'b0; v4 <= 1'b0;
        end else if (en) begin
            v1 <= in_valid; v2 <= v1; v3 <= v2; v4 <= v3;
        end
    end

    //=================================================================
    // Stadio 1: somma dell'offset e confronti (in parallelo)
    //=================================================================
    reg                      below1, above1;       // sotto / sopra l'intervallo
    reg [LOG2_NSEG-1:0]      idx1;                 // segmento
    reg [FRACW-1:0]          frac1;                // posizione nel segmento
    reg signed [W_IN-1:0]    x1;                   // x originale (serve se MODE_HI = 1)
    reg [TAG_W-1:0]          tag1;

    always @(posedge clk) if (en) begin
        below1 <= (x <  X_LO);
        above1 <= (x >= X_HI);
        idx1   <= xs[FRACW +: LOG2_NSEG];
        frac1  <= xs[FRACW-1:0];
        x1     <= in_data;
        tag1   <= in_tag;
    end

    //=================================================================
    // Stadio 2: lettura della LUT
    //=================================================================
    reg [2*W_OUT-1:0]     seg2;                    // {slope, base} del segmento
    reg [FRACW-1:0]       frac2;
    reg                   below2, above2;
    reg signed [W_IN-1:0] x2;
    reg [TAG_W-1:0]       tag2;

    always @(posedge clk) if (en) begin
        seg2   <= lut[idx1];
        frac2  <= frac1;
        below2 <= below1;
        above2 <= above1;
        x2     <= x1;
        tag2   <= tag1;
    end

    wire signed [W_OUT-1:0] base2  = $signed(seg2[0     +: W_OUT]);
    wire signed [W_OUT-1:0] slope2 = $signed(seg2[W_OUT +: W_OUT]);

    //=================================================================
    // Stadio 3: prodotto slope * frac (frac e' senza segno)
    //   in parallelo: termine costante (base << FRACW) + mezzo LSB, che e'
    //   una concatenazione {base, 1, 0...0} (nessun sommatore)
    //=================================================================
    reg signed [PW3-1:0]  prod3;                   // slope * frac
    reg signed [PW3-1:0]  addend3;                 // base * 2^FRACW + 2^(FRACW-1)
    reg                   below3, above3;
    reg signed [W_IN-1:0] x3;
    reg [TAG_W-1:0]       tag3;

    always @(posedge clk) if (en) begin
        prod3   <= slope2 * $signed({1'b0, frac2});
        addend3 <= $signed({base2[W_OUT-1], base2, 1'b1, {(FRACW-1){1'b0}}});
        below3  <= below2;
        above3  <= above2;
        x3      <= x2;
        tag3    <= tag2;
    end

    //=================================================================
    // Stadio 4: un solo sommatore
    //=================================================================
    reg signed [PW3-1:0]  sum4;                    // prod + rnd + base * 2^FRACW
    reg                   below4, above4;
    reg signed [W_IN-1:0] x4;
    reg [TAG_W-1:0]       tag4;

    always @(posedge clk) if (en) begin
        sum4   <= prod3 + addend3;
        below4 <= below3;
        above4 <= above3;
        x4     <= x3;
        tag4   <= tag3;
    end

    //=================================================================
    // Stadio 5: shift (fili), saturazione, valori fuori intervallo
    //=================================================================
    wire signed [PW3-1:0] interp = sum4 >>> FRACW;

    localparam signed [W_OUT-1:0] YLO = Y_LO_Q[W_OUT-1:0];
    localparam signed [W_OUT-1:0] YHI = Y_HI_Q[W_OUT-1:0];
    localparam signed [PW3-1:0] OMAX = {{(PW3-W_OUT+1){1'b0}}, {(W_OUT-1){1'b1}}};
    localparam signed [PW3-1:0] OMIN = {{(PW3-W_OUT+1){1'b1}}, {(W_OUT-1){1'b0}}};

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            out_data  <= {W_OUT{1'b0}};
            out_tag   <= {TAG_W{1'b0}};
        end else if (en) begin
            out_valid <= v4;
            out_tag   <= tag4;

            if (below4)
                out_data <= YLO;
            else if (above4)
                // MODE_HI = 1 presuppone stesso formato in ingresso e in uscita
                out_data <= (MODE_HI != 0) ? x4[W_OUT-1:0] : YHI;
            else if (interp > OMAX)
                out_data <= OMAX[W_OUT-1:0];
            else if (interp < OMIN)
                out_data <= OMIN[W_OUT-1:0];
            else
                out_data <= interp[W_OUT-1:0];
        end
    end

endmodule

`default_nettype wire
