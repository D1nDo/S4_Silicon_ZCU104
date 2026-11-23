`timescale 1ns / 1ps
`default_nettype none
//=====================================================================
// s4d_layer  --  uno strato S4D completo
//
//   Elabora la sequenza un timestep alla volta; ogni timestep e' un
//   "frame" di H campioni (canale 0..H-1). Catena di elaborazione:
//
//     x --+--> s4d_affine (BatchNorm) --> s4d_biquad_bank (S4D + D*u)
//         |        u_norm                      u_bank
//         |                                       |
//         |                                       v
//         |                               s4d_pwl (GELU)
//         |                                   u_gelu
//         |                                       |
//         |                                       v
//         |                               s4d_mixing_glu (lineare H->2H, GLU)
//         |                                   u_mix
//         |                                       |
//         +--> FIFO residual ----------------> ( + ) --> saturazione --> y
//
//     y[t][ch] = sat16( x[t][ch] + GLU( Mix( GELU( S4D( BN(x) ) ) ) )[t][ch] )
//
//   Il ramo residual e' una FIFO che conserva x finche' il ramo
//   principale produce il risultato dello stesso campione (latenza di
//   diversi frame: doppi buffer nel banco biquad e nel mixing).
//
//   2026-10-15 (fan-out dello stallo d'uscita): il ramo GLU e il
//   residual scorrono liberi; lo stallo d'uscita agisce solo sulla CODA
//   D'USCITA (TAIL_DEPTH posti, con registro di testa) e, tramite i
//   crediti (tail_pop), sull'emissione dalla testa dello snapshot del
//   mixing. Prima out_ready (oen) abilitava direttamente ~800 registri
//   qui e nel mixing, piu' ~8000 dello snapshot. Versione precedente in
//   S4D_RTL/new_pre_oen_uram.
//
//   Tutti i parametri di default sono sovrascritti dal top
//   (s4d_core_axis): i valori qui sotto sono solo di riserva.
//=====================================================================

module s4d_layer #(
    // ---- dimensioni (vedi s4d_core_axis) ----
    parameter integer H        = 128,
    parameter integer LOG2_H   = 7,
    parameter integer NUNITS   = 8,
    parameter integer LOG2_NU  = 3,
    parameter integer MODES    = 32,
    parameter integer LOG2_MOD = 5,
    parameter integer NCH      = 16,
    parameter integer LOG2_NCH = 4,
    parameter integer LOG2_NB  = LOG2_MOD + LOG2_NCH,
    parameter integer NMAC     = 256,       // era 64: un MAC per uscita (mixing a fase singola)
    parameter integer LOG2_NPH = 0,         // era 2: una sola fase -> 0 bit
    parameter         WMEM_STYLE = "ultra", // memoria dei pesi del mixing: "ultra" | "block" | "distributed"

    // ---- formati a virgola fissa (vedi s4d_core_axis) ----
    parameter integer W_ACT   = 16,
    parameter integer F_ACT   = 11,
    parameter integer W_U     = 16,
    parameter integer F_U     = 11,
    parameter integer W_A     = 24,
    parameter integer F_A     = 22,
    parameter integer W_B     = 16,
    parameter integer F_B     = 15,
    parameter integer W_S     = 32,
    parameter integer F_Y     = 19,
    parameter integer W_D     = 16,
    parameter integer F_D     = 14,
    parameter integer W_BSH   = 5,
    parameter integer W_BACC  = 40,
    parameter integer W_SSM   = 16,
    parameter integer F_SSM   = 11,
    parameter integer W_W     = 12,
    parameter integer F_W     = 11,
    parameter integer W_MACC  = 32,
    parameter integer F_SIGI  = 11,
    parameter integer F_SIGO  = 14,
    parameter integer FIFO_LOG2 = 10,   // profondita' della FIFO residual = 2^FIFO_LOG2
    parameter integer TAIL_LOG2 = 5,    // coda d'uscita: 2^TAIL_LOG2 posti (crediti del mixing)

    parameter integer LOADABLE = 0,     // 1 = pesi scritti a runtime dalle porte ld_*
    parameter integer LD_W    = 64,     // larghezza delle parole di caricamento pesi
    parameter integer BN_SH       = 11, // shift fisso del BatchNorm (se non caricabile)
    parameter integer BN_W_W      = 16,
    parameter integer BN_W_B      = 32,
    parameter integer BN_W_ACC    = 34,

    // s4d_layer e' UN layer fisico riusato per NLAYER ruoli (time-multiplexato):
    // non esiste "il" file giusto qui, chi lo istanzia sovrascrive sempre. I
    // default puntano al layer 0 solo perche' e' un file che esiste davvero,
    // non un placeholder inventato.
    parameter BN_W_FILE  = "l0_bn_w.mem",
    parameter BN_B_FILE  = "l0_bn_b.mem",
    parameter GELU_FILE  = "gelu_lut.mem",
    parameter SIG_FILE   = "sigmoid_lut.mem",
    parameter WOUT_FILE  = "l0_s4d_wout.mem",
    parameter BOUT_FILE  = "l0_s4d_bout.mem"
)(
    input  wire                    clk,
    input  wire                    rst_n,

    // ---- caricamento coefficienti biquad (da s4d_wload_axis) ----
    input  wire                    cfg_we,
    input  wire [LOG2_NU-1:0]      cfg_unit,
    input  wire [LOG2_NB-1:0]      cfg_addr,
    input  wire [2*W_A+2*W_B-1:0]  cfg_data,
    input  wire                    cfgd_we,
    input  wire [LOG2_H-1:0]       cfgd_ch,
    input  wire signed [W_D-1:0]   cfgd_data,
    input  wire [W_BSH-1:0]        cfgd_bsh,

    // ---- caricamento BatchNorm ----
    input  wire                    ld_bn_we_sh,
    input  wire [5:0]              ld_bn_sh,
    input  wire                    ld_bn_we_w,
    input  wire                    ld_bn_we_b,
    input  wire [LOG2_H-1:0]       ld_bn_addr,
    input  wire signed [BN_W_W-1:0] ld_bn_w,
    input  wire signed [BN_W_B-1:0] ld_bn_b,

    // ---- caricamento mixing ----
    input  wire                    ld_wo_we,
    input  wire [LOG2_NPH+LOG2_H-1:0] ld_wo_addr,
    input  wire [5:0]              ld_wo_sel,   // era [3:0]: 48 banchi da 64 bit per riga
    input  wire [LD_W-1:0]         ld_wo_data,
    input  wire                    ld_bo_we,
    input  wire [LOG2_H:0]         ld_bo_addr,
    input  wire signed [W_MACC-1:0] ld_bo_data,

    // ---- ingresso: un campione per beat, frame di H canali ----
    input  wire                    in_valid,
    output wire                    in_ready,
    input  wire signed [W_ACT-1:0] in_data,
    input  wire                    in_last_seq,  // ultimo campione della sequenza

    // ---- uscita ----
    input  wire                    out_ready,
    output reg                     out_valid,
    output reg  signed [W_ACT-1:0] out_data,
    output reg                     out_last_ch,  // ultimo canale del frame
    output reg                     out_last_seq, // ultimo campione della sequenza

    // ---- diagnostica ----
    output wire                    ovf,          // overflow nel banco biquad (sticky)
    output reg                     err_bank_ovr  // sentinella sul ready GELU -> mixing (sticky)
);

    // ------------------------------------------------------------------
    // BACKPRESSURE A CREDITI
    // Il mixing emette un campione dalla testa del suo snapshot solo se
    // ha un credito; da li' il campione percorre GLU e residual senza
    // fermarsi e finisce nella coda d'uscita (TAIL_DEPTH posti). Ogni
    // campione che lascia la coda (tail_pop) restituisce un credito.
    // Il pop della FIFO residual avviene quando il risultato del GLU
    // arriva (gl_valid): GLU e residual avanzano insieme per costruzione,
    // quindi il residual non puo' disallinearsi.
    // ------------------------------------------------------------------
    localparam integer FDEPTH     = (1 << FIFO_LOG2);   // posti della FIFO residual (1024)
    localparam integer TAIL_DEPTH = (1 << TAIL_LOG2);   // posti della coda d'uscita (32)

    wire tail_pop;                                      // un campione lascia la coda d'uscita

    //-----------------------------------------------------------------
    // FIFO del residual: conserva il campione d'ingresso x (e il flag
    // di fine sequenza) fino a quando il ramo principale produce il suo
    // risultato. Lettura asincrona (fifo[frd]) -> realizzata in LUTRAM.
    //-----------------------------------------------------------------
    reg signed [W_ACT-1:0] fifo    [0:FDEPTH-1];   // campioni x
    reg                    fifo_ls [0:FDEPTH-1];   // flag in_last_seq di ogni campione
    reg [FIFO_LOG2-1:0]    fwr, frd;               // puntatori di scrittura / lettura
    reg [FIFO_LOG2:0]      fcnt;                   // campioni presenti

    // si accettano ingressi solo se restano piu' di H posti liberi
    // (margine di un frame intero)
    wire fifo_ready = (fcnt < FDEPTH - H);

    // uscita del BatchNorm
    wire                   ln_valid;
    wire signed [W_U-1:0]  ln_data;
    wire                   ln_last_ch, ln_last_seq;
    wire                   ln_ready;     // il BatchNorm accetta un ingresso
    wire                   bk_ready;     // il banco biquad accetta un ingresso

    //=================================================================
    // BatchNorm per canale: y = (w[ch]*x + b[ch]) >> shift
    // (modulo s4d_affine, file s4d_affine.v)
    //=================================================================
    s4d_affine #(
        .H(H), .LOG2_H(LOG2_H),
        .W_IN(W_ACT), .W_OUT(W_U),
        .W_W(BN_W_W), .W_B(BN_W_B), .W_ACC(BN_W_ACC), .SH(BN_SH),
        .LOADABLE(LOADABLE),
        .W_FILE(BN_W_FILE), .B_FILE(BN_B_FILE)
    ) u_norm (
        .clk(clk), .rst_n(rst_n),
        .ld_we_w(ld_bn_we_w), .ld_we_b(ld_bn_we_b),
        .ld_addr(ld_bn_addr), .ld_w(ld_bn_w), .ld_b(ld_bn_b),
        .ld_we_sh(ld_bn_we_sh), .ld_sh(ld_bn_sh),
        .in_valid(in_valid && fifo_ready), .in_ready(ln_ready),
        .in_data(in_data), .in_last_seq(in_last_seq),
        .out_ready(bk_ready),
        .out_valid(ln_valid), .out_data(ln_data),
        .out_last_ch(ln_last_ch), .out_last_seq(ln_last_seq)
    );

    // uscita del banco biquad
    wire                     bk_valid;
    wire signed [W_SSM-1:0]  bk_data;
    wire [LOG2_H-1:0]        bk_ch;
    wire                     bk_last;     // ultimo canale del frame

    wire ge_in_ready;                     // la GELU accetta un ingresso
    wire mix_ready;                       // il mixing accetta un ingresso

    //=================================================================
    // Convoluzione S4D: per ogni canale somma di MODES biquad + D*u
    //=================================================================
    s4d_biquad_bank #(
        .H(H), .LOG2_H(LOG2_H), .NUNITS(NUNITS), .LOG2_NU(LOG2_NU),
        .MODES(MODES), .LOG2_MOD(LOG2_MOD),
        .NCH(NCH), .LOG2_NCH(LOG2_NCH), .LOG2_NB(LOG2_NB),
        .W_U(W_U), .F_U(F_U), .W_A(W_A), .F_A(F_A),
        .W_B(W_B), .F_B(F_B), .W_S(W_S), .F_Y(F_Y),
        .W_D(W_D), .F_D(F_D), .W_BSH(W_BSH), .W_ACC(W_BACC),
        .W_OUT(W_SSM), .F_OUT(F_SSM)
    ) u_bank (
        .clk(clk), .rst_n(rst_n),
        .cfg_we(cfg_we), .cfg_unit(cfg_unit), .cfg_addr(cfg_addr), .cfg_data(cfg_data),
        .cfgd_we(cfgd_we), .cfgd_ch(cfgd_ch), .cfgd_data(cfgd_data), .cfgd_bsh(cfgd_bsh),
        .in_valid(ln_valid), .in_ready(bk_ready), .in_data(ln_data),
        .in_last_seq(ln_last_seq),
        .out_valid(bk_valid), .out_ready(ge_in_ready), .out_data(bk_data),
        .out_ch(bk_ch), .out_last(bk_last),
        .ovf(ovf)
    );

    // uscita della GELU
    wire                    ge_valid;
    wire signed [W_SSM-1:0] ge_data;
    wire                    ge_last;      // bk_last arrivato insieme al dato (tag della PWL)

    // Con s4d_pwl dotato di in_ready/out_ready la catena
    // bank -> gelu -> mixing e' continua: questo caso non puo' piu'
    // verificarsi. Il flag resta come sentinella: se scatta, significa
    // che qualcuno ha rotto la propagazione del ready.
    // NOTA (analisi): ge_valid && !mix_ready e' anche lo stato di un
    // normale stallo AXI (GELU con dato pronto, mixing con i 2 buffer
    // pieni). Nessun dato si perde, ma il flag puo' alzarsi quando
    // l'uscita del core applica backpressure nell'ultima passata.
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)                       err_bank_ovr <= 1'b0;
        else if (ge_valid && !mix_ready)  err_bank_ovr <= 1'b1;
    end

    //=================================================================
    // GELU approssimata a tratti lineari (256 segmenti in [-8, 8)),
    // fuori intervallo: 0 a sinistra, x a destra (MODE_HI=1)
    //=================================================================
    s4d_pwl #(
        .W_IN(W_SSM), .F_IN(F_SSM), .W_OUT(W_SSM), .F_OUT(F_SSM),
        .NSEG(256), .LOG2_NSEG(8), .XH(3),
        .MODE_HI(1), .Y_LO_Q(0), .Y_HI_Q(0),
        .TAG_W(1), .LUT_FILE(GELU_FILE)
    ) u_gelu (
        .clk(clk), .rst_n(rst_n),
        .in_valid(bk_valid), .in_ready(ge_in_ready), .in_data(bk_data),
        .in_tag(bk_last),                     // il flag di fine frame viaggia con il dato
        .out_ready(mix_ready),
        .out_valid(ge_valid), .out_data(ge_data), .out_tag(ge_last)
    );

    // uscita del mixing/GLU
    wire                    gl_valid;
    wire signed [W_ACT-1:0] gl_data;
    wire                    gl_last;

    //=================================================================
    // Mixing lineare H -> 2H seguito da GLU: out = a * sigmoid(g)
    //=================================================================
    s4d_mixing_glu #(
        .H(H), .LOG2_H(LOG2_H),
        .NMAC(NMAC), .LOG2_NPH(LOG2_NPH),
        .W_IN(W_SSM), .F_IN(F_SSM),
        .W_W(W_W), .F_W(F_W), .W_ACC(W_MACC),
        .W_SIG(16), .F_SIGI(F_SIGI), .F_SIGO(F_SIGO),
        .W_A(16), .F_A(F_ACT),
        .W_OUT(W_ACT), .F_OUT(F_ACT),
        .LOADABLE(LOADABLE), .LD_W(LD_W),
        .LOG2_NBANK(6), .WMEM_STYLE(WMEM_STYLE),   // 48 banchi -> 6 bit di selezione
        .TAIL_DEPTH(TAIL_DEPTH),                     // crediti = posti della coda d'uscita
        .W_FILE(WOUT_FILE), .B_FILE(BOUT_FILE), .SIG_FILE(SIG_FILE)
    ) u_mix (
        .clk(clk), .rst_n(rst_n),
        .ld_we_w(ld_wo_we), .ld_addr_w(ld_wo_addr), .ld_sel_w(ld_wo_sel),
        .ld_data_w(ld_wo_data),
        .ld_we_b(ld_bo_we), .ld_addr_b(ld_bo_addr), .ld_data_b(ld_bo_data),
        .in_valid(ge_valid), .in_ready(mix_ready),
        .in_data(ge_data), .in_last(ge_last),
        .tail_pop(tail_pop),
        .out_valid(gl_valid), .out_data(gl_data), .out_last(gl_last)
    );

    // l'ingresso passa solo se il BatchNorm e la FIFO residual hanno posto
    assign in_ready = ln_ready && fifo_ready;

    wire fifo_push = in_valid && in_ready;   // x entra nel layer -> anche nella FIFO
    wire fifo_pop  = gl_valid;               // arriva il risultato -> si consuma il suo x

    // puntatori e conteggio della FIFO residual
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fwr  <= {FIFO_LOG2{1'b0}};
            frd  <= {FIFO_LOG2{1'b0}};
            fcnt <= {(FIFO_LOG2+1){1'b0}};
        end else begin
            if (fifo_push) fwr <= fwr + 1'b1;
            if (fifo_pop)  frd <= frd + 1'b1;
            fcnt <= fcnt + (fifo_push ? 1'b1 : 1'b0) - (fifo_pop ? 1'b1 : 1'b0);
        end
    end

    // scrittura nella FIFO residual
    always @(posedge clk) if (rst_n && fifo_push) fifo[fwr] <= in_data;

    always @(posedge clk) if (fifo_push) fifo_ls[fwr] <= in_last_seq;

    //-----------------------------------------------------------------
    // Somma residual: x (dalla FIFO) + uscita del GLU, su W_ACT+1 bit
    // per non perdere il riporto, poi saturazione a W_ACT bit.
    // Timing (2026-10-15): prima lettura della FIFO in LUTRAM (1024
    // posti), somma e saturazione erano nello stesso ciclo. Ora:
    //   R1  lettura della FIFO (x e flag) e copia di gl_data
    //   R2  somma x + gl
    //   R3  saturazione -> coda d'uscita
    // Scorrimento libero (2026-10-15): il pop della FIFO avviene in R1,
    // nello stesso ciclo in cui il risultato del GLU viene registrato,
    // quindi il residual non si disallinea.
    //-----------------------------------------------------------------
    localparam integer RW = W_ACT + 1;
    localparam signed [RW-1:0] RMAX = {{(RW-W_ACT+1){1'b0}}, {(W_ACT-1){1'b1}}};   // +32767
    localparam signed [RW-1:0] RMIN = {{(RW-W_ACT+1){1'b1}}, {(W_ACT-1){1'b0}}};   // -32768

    reg [LOG2_H-1:0] och;    // canale del prossimo risultato in R1 (per out_last_ch)

    // ---- R1: lettura della FIFO residual ----
    reg                    r1_valid, r1_last_ch, r1_last_seq;
    reg signed [W_ACT-1:0] r1_x, r1_gl;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r1_valid    <= 1'b0;
            r1_last_ch  <= 1'b0;
            r1_last_seq <= 1'b0;
            och         <= {LOG2_H{1'b0}};
        end else begin
            r1_valid    <= gl_valid;
            r1_last_ch  <= gl_valid && (och == H-1);
            // fine sequenza: ultimo canale di un frame il cui x era l'ultimo
            r1_last_seq <= gl_valid && (och == H-1) && fifo_ls[frd];
            if (gl_valid) och <= (och == H-1) ? {LOG2_H{1'b0}} : (och + 1'b1);
        end
    end

    always @(posedge clk) begin
        r1_x  <= fifo[frd];
        r1_gl <= gl_data;
    end

    // ---- R2: somma su RW bit ----
    reg                    r2_valid, r2_last_ch, r2_last_seq;
    reg signed [RW-1:0]    r2_sum;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r2_valid    <= 1'b0;
            r2_last_ch  <= 1'b0;
            r2_last_seq <= 1'b0;
        end else begin
            r2_valid    <= r1_valid;
            r2_last_ch  <= r1_last_ch;
            r2_last_seq <= r1_last_seq;
        end
    end

    always @(posedge clk)
        r2_sum <= {r1_x[W_ACT-1], r1_x} + {r1_gl[W_ACT-1], r1_gl};

    //-----------------------------------------------------------------
    // R3: saturazione (scorrimento libero) -> coda d'uscita
    //-----------------------------------------------------------------
    reg                    r3_valid, r3_last_ch, r3_last_seq;
    reg signed [W_ACT-1:0] r3_data;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r3_valid    <= 1'b0;
            r3_last_ch  <= 1'b0;
            r3_last_seq <= 1'b0;
        end else begin
            r3_valid    <= r2_valid;
            r3_last_ch  <= r2_last_ch;
            r3_last_seq <= r2_last_seq;
        end
    end

    always @(posedge clk) begin
        if      (r2_sum > RMAX) r3_data <= RMAX[W_ACT-1:0];
        else if (r2_sum < RMIN) r3_data <= RMIN[W_ACT-1:0];
        else                    r3_data <= r2_sum[W_ACT-1:0];
    end

    //-----------------------------------------------------------------
    // CODA D'USCITA: memoria di TAIL_DEPTH posti (LUTRAM) + registro di
    // testa, che e' direttamente l'uscita del layer (out_valid/out_data/
    // out_last_*: tutte da registro).
    //   - la testa si libera (tail_head_take) se vuota o se viene letta;
    //     allora pesca dalla memoria, o, se la memoria e' vuota,
    //     direttamente dall'ingresso r3 (nessun ciclo perso a full rate);
    //   - altrimenti il dato in arrivo va in memoria.
    // Non puo' traboccare: i campioni nella coda (memoria + testa) sono al
    // piu' quelli "in volo" contati dai crediti del mixing, <= TAIL_DEPTH-1.
    // out_ready pilota solo questa logica (~40 carichi).
    //-----------------------------------------------------------------
    localparam integer TAIL_W = W_ACT + 2;              // {last_seq, last_ch, dato}

    reg [TAIL_W-1:0]    tail_mem [0:TAIL_DEPTH-1];     // memoria della coda (lettura asincrona)
    reg [TAIL_LOG2-1:0] tail_wr, tail_rd;              // puntatori di scrittura / lettura
    reg [TAIL_LOG2:0]   tail_cnt;                      // elementi in memoria (testa esclusa)

    wire [TAIL_W-1:0] tail_in        = {r3_last_seq, r3_last_ch, r3_data};
    wire              tail_mem_empty = (tail_cnt == {(TAIL_LOG2+1){1'b0}});

    assign tail_pop = out_valid && out_ready;           // la testa viene consumata
    wire tail_head_take = !out_valid || out_ready;      // la testa puo' essere ricaricata

    // il dato in arrivo va in memoria, tranne quando passa direttamente in testa
    wire tail_push = r3_valid && !(tail_head_take && tail_mem_empty);
    // la testa pesca dalla memoria
    wire tail_mem_pop = tail_head_take && !tail_mem_empty;

    always @(posedge clk) if (tail_push) tail_mem[tail_wr] <= tail_in;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tail_wr  <= {TAIL_LOG2{1'b0}};
            tail_rd  <= {TAIL_LOG2{1'b0}};
            tail_cnt <= {(TAIL_LOG2+1){1'b0}};
        end else begin
            if (tail_push)    tail_wr <= tail_wr + 1'b1;
            if (tail_mem_pop) tail_rd <= tail_rd + 1'b1;
            tail_cnt <= tail_cnt + (tail_push ? 1'b1 : 1'b0) - (tail_mem_pop ? 1'b1 : 1'b0);
        end
    end

    // registro di testa = uscita del layer
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid    <= 1'b0;
            out_data     <= {W_ACT{1'b0}};
            out_last_ch  <= 1'b0;
            out_last_seq <= 1'b0;
        end else if (tail_head_take) begin
            if (!tail_mem_empty) begin
                // il piu' vecchio in memoria
                out_valid <= 1'b1;
                {out_last_seq, out_last_ch, out_data} <= tail_mem[tail_rd];
            end else begin
                // memoria vuota: direttamente da R3 (o testa vuota)
                out_valid <= r3_valid;
                {out_last_seq, out_last_ch, out_data} <= r3_valid ? tail_in : {TAIL_W{1'b0}};
            end
        end
    end

    // controllo solo in simulazione: la coda non deve mai traboccare
// synthesis translate_off
    always @(posedge clk)
        if (rst_n && tail_push && (tail_cnt == TAIL_DEPTH)) begin
            $display("s4d_layer: ERRORE, coda d'uscita piena (crediti non rispettati) t=%0t", $time);
        end
// synthesis translate_on

endmodule

`default_nettype wire
