`timescale 1ns / 1ps
`default_nettype none
//=====================================================================
// s4d_affine  --  trasformazione affine per canale (BatchNorm fusa)
//
//   Per ogni campione del canale ch:
//
//     y = sat( round( (w[ch]*x + b[ch]) / 2^sh ) )
//
//   I campioni arrivano in ordine di canale 0, 1, ..., H-1, 0, 1, ...
//   e il contatore interno ch tiene traccia del canale corrente.
//
//   Pesi e bias: da file .mem (LOADABLE = 0) oppure scritti a runtime
//   dalle porte ld_* (LOADABLE = 1). Lo shift sh vale SH dopo il reset
//   e, con LOADABLE = 1, puo' essere riscritto da ld_sh.
//
//   Pipeline a 6 stadi, stallabile globalmente con out_ready, al massimo
//   UN'operazione complessa per ciclo (timing, 2026-10-15; prima 3 stadi
//   con w*x+b in un ciclo e somma + shift variabile + saturazione in un
//   altro):
//     S0 lettura di w[ch] e b[ch], registro di x
//     S1 prodotto w*x                               (registro M del DSP)
//     S2 somma del bias: w*x + b                    (registro P del DSP)
//     S3 somma del mezzo LSB: acc + 2^(sh-1)
//     S4 shift aritmetico variabile di sh bit
//     S5 saturazione a W_OUT bit -> uscita
//   Aritmetica identica alla versione a 3 stadi (stesse larghezze, stessi
//   troncamenti): cambia solo dove stanno i registri.
//   Il mezzo LSB 2^(sh-1) e' un registro calcolato da sh_r (che cambia
//   solo al caricamento dei pesi, molti cicli prima dei dati).
//=====================================================================

module s4d_affine #(
    parameter integer H      = 128,    // canali
    parameter integer LOG2_H = 7,
    parameter integer W_IN   = 16,     // ingresso x
    parameter integer W_OUT  = 16,     // uscita y
    parameter integer W_W    = 16,     // peso w
    parameter integer W_B    = 32,     // bias b (nella scala del prodotto w*x)
    parameter integer W_ACC  = 34,     // accumulatore w*x + b
    parameter integer SH     = 11,     // shift di uscita dopo il reset

    parameter integer LOADABLE = 0,    // 1 = pesi/bias/shift scritti dalle porte ld_*
    parameter         W_FILE = "l6_bn_w.mem",  // modulo riusato per 6+1 ruoli diversi
    parameter         B_FILE = "l6_bn_b.mem"   // (nessun nome e' "il" nome giusto): punto
                                                // al BN finale del decoder, l'unico caso in
                                                // cui questo default viene davvero letto
                                                // (LOADABLE=0). Chi lo usa per un layer deve
                                                // sempre sovrascrivere esplicitamente.
)(
    input  wire                    clk,
    input  wire                    rst_n,

    // ---- caricamento pesi e bias del canale ld_addr ----
    input  wire                    ld_we_w,
    input  wire                    ld_we_b,
    input  wire [LOG2_H-1:0]       ld_addr,
    input  wire signed [W_W-1:0]   ld_w,
    input  wire signed [W_B-1:0]   ld_b,

    // ---- caricamento dello shift ----
    input  wire                    ld_we_sh,
    input  wire [5:0]              ld_sh,

    // ---- ingresso ----
    input  wire                    in_valid,
    output wire                    in_ready,
    input  wire signed [W_IN-1:0]  in_data,
    input  wire                    in_last_seq,   // ultimo campione della sequenza

    // ---- uscita ----
    input  wire                    out_ready,
    output reg                     out_valid,
    output reg  signed [W_OUT-1:0] out_data,
    output reg                     out_last_ch,   // ultimo canale del frame
    output reg                     out_last_seq   // ultimo campione della sequenza
);

    // memorie dei pesi e dei bias, una voce per canale
    reg signed [W_W-1:0] w_rom [0:H-1];
    reg signed [W_B-1:0] b_rom [0:H-1];

    generate
    if (LOADABLE == 0) begin : g_init
        // valori fissi da file
        initial begin
            $readmemh(W_FILE, w_rom);
            $readmemh(B_FILE, b_rom);
        end
    end else begin : g_load
        // valori scritti a runtime
        always @(posedge clk) begin
            if (ld_we_w) w_rom[ld_addr] <= ld_w;
            if (ld_we_b) b_rom[ld_addr] <= ld_b;
        end
    end
    endgenerate

    // shift di uscita: SH dopo il reset, poi eventualmente caricato
    reg [5:0] sh_r;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)                             sh_r <= SH[5:0];
        else if ((LOADABLE != 0) && ld_we_sh)   sh_r <= ld_sh;
    end

    // stallo globale: tutta la pipeline avanza solo con out_ready
    // (in_ready e' combinatorio da out_ready)
    wire en = out_ready;
    assign in_ready = out_ready;

    wire acc_in = in_valid & in_ready;     // campione accettato

    reg [LOG2_H-1:0] ch;                   // canale del prossimo campione in ingresso

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)          ch <= {LOG2_H{1'b0}};
        else if (acc_in)     ch <= (ch == H-1) ? {LOG2_H{1'b0}} : ch + 1'b1;
    end

    //=================================================================
    // S0: lettura di peso e bias del canale, registro dell'ingresso
    //=================================================================
    reg signed [W_IN-1:0] x0;
    reg signed [W_W-1:0]  w0;
    reg signed [W_B-1:0]  b0;
    reg [LOG2_H-1:0]      ch0;
    reg                   v0, ls0;         // valid, ultimo della sequenza

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v0 <= 1'b0;
            ls0 <= 1'b0;
        end else if (en) begin
            v0  <= acc_in;
            ls0 <= acc_in & in_last_seq;
        end
    end

    always @(posedge clk) begin
        if (en) begin
            x0  <= in_data;
            w0  <= w_rom[ch];
            b0  <= b_rom[ch];
            ch0 <= ch;
        end
    end

    //=================================================================
    // Flag che viaggiano con il dato negli stadi S1..S4:
    //   v_pipe[k]    = dato valido allo stadio S(k+1)
    //   last_pipe[k] = e' l'ultimo canale del frame
    //   seq_pipe[k]  = e' l'ultimo campione della sequenza
    //=================================================================
    reg [3:0] v_pipe, last_pipe, seq_pipe;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v_pipe    <= 4'b0;
            last_pipe <= 4'b0;
            seq_pipe  <= 4'b0;
        end else if (en) begin
            v_pipe    <= {v_pipe[2:0],    v0};
            last_pipe <= {last_pipe[2:0], v0 && (ch0 == H-1)};
            seq_pipe  <= {seq_pipe[2:0],  v0 && (ch0 == H-1) && ls0};
        end
    end

    //=================================================================
    // S1: prodotto w*x
    //=================================================================
    reg signed [W_W+W_IN-1:0] prod1;       // w*x
    reg signed [W_B-1:0]      b1;          // bias allineato al prodotto

    always @(posedge clk) begin
        if (en) begin
            prod1 <= w0 * x0;
            b1    <= b0;
        end
    end

    //=================================================================
    // S2: somma del bias (estesi a W_ACC bit)
    //=================================================================
    reg signed [W_ACC-1:0] acc2;           // w*x + b

    always @(posedge clk) begin
        if (en)
            acc2 <= $signed({{(W_ACC-W_W-W_IN){prod1[W_W+W_IN-1]}}, prod1})
                  + $signed({{(W_ACC-W_B){b1[W_B-1]}}, b1});
    end

    //=================================================================
    // S3: somma del mezzo LSB d'uscita (round-half-up)
    //   rnd_r = 2^(sh_r-1), oppure 0 se sh_r = 0: registro ricalcolato a
    //   ogni ciclo da sh_r, che e' fermo durante l'elaborazione
    //=================================================================
    reg signed [W_ACC-1:0] rnd_r;
    always @(posedge clk)
        rnd_r <= (sh_r > 0) ? ({{(W_ACC-1){1'b0}}, 1'b1} <<< (sh_r-1))
                            : {W_ACC{1'b0}};

    reg signed [W_ACC-1:0] acc3;           // w*x + b + 2^(sh-1)

    always @(posedge clk) begin
        if (en) acc3 <= acc2 + rnd_r;
    end

    //=================================================================
    // S4: shift aritmetico variabile di sh_r bit
    //=================================================================
    reg signed [W_ACC-1:0] shifted4;

    always @(posedge clk) begin
        if (en) shifted4 <= acc3 >>> sh_r;
    end

    //=================================================================
    // S5: saturazione a W_OUT bit -> registro d'uscita
    //=================================================================
    localparam signed [W_ACC-1:0] SAT_MAX = (1 <<< (W_OUT-1)) - 1;
    localparam signed [W_ACC-1:0] SAT_MIN = -(1 <<< (W_OUT-1));

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid    <= 1'b0;
            out_data     <= {W_OUT{1'b0}};
            out_last_ch  <= 1'b0;
            out_last_seq <= 1'b0;
        end else if (en) begin
            out_valid    <= v_pipe[3];
            out_last_ch  <= last_pipe[3];
            out_last_seq <= seq_pipe[3];

            if      (shifted4 > SAT_MAX) out_data <= SAT_MAX[W_OUT-1:0];
            else if (shifted4 < SAT_MIN) out_data <= SAT_MIN[W_OUT-1:0];
            else                         out_data <= shifted4[W_OUT-1:0];
        end
    end

endmodule

`default_nettype wire
