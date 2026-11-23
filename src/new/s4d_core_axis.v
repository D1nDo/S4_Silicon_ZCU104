`timescale 1ns / 1ps
`default_nettype none
//=====================================================================
// s4d_core_axis  --  TOP del core S4D
//
//   Esegue un'intera inferenza S4D a NLAYER strati su una sequenza di
//   L timestep x H canali, usando UN SOLO strato fisico (s4d_layer)
//   riusato NLAYER volte (time-multiplexing). I pesi del layer vengono
//   ricaricati dal DMA prima di ogni passata.
//
//   Gerarchia:
//     s4d_core_axis
//       |- s4d_wload_axis   (u_wload)  : riceve lo stream dei pesi e li
//       |                                scrive nelle memorie del layer
//       |- s4d_seqbuf       (u_buf)    : buffer della sequenza, L*H parole
//       |- s4d_layer        (u_layer)  : lo strato S4D fisico
//            |- s4d_affine      (BatchNorm)        [s4d_affine.v]
//            |- s4d_biquad_bank (convoluzione S4D) [s4d_biquad_bank.v]
//            |     |- s4d_biquad_unit x NUNITS     [s4d_biquad_unit.v]
//            |- s4d_pwl         (GELU)             [s4d_pwl.v]
//            |- s4d_mixing_glu  (lineare H->2H + GLU) [s4d_mixing_glu.v]
//                  |- s4d_pwl   (sigmoide)
//
//   Flusso di un'inferenza (macchina a stati st):
//
//     ST_IDLE  aspetta il primo beat dell'encoder
//     ST_ENC   scrive gli L*H beat dell'encoder nel seqbuf
//              (indirizzo = t*H + canale, perche' l'encoder emette
//              H canali per ogni timestep)
//     ST_LOAD  u_wload carica i pesi del layer m dallo stream s_axis_w
//     ST_PASS  legge il seqbuf dall'indirizzo 0 all'ultimo e lo manda
//              nel layer:
//                - passate 0..NLAYER-2: l'uscita del layer viene
//                  riscritta NELLO STESSO seqbuf ("in place"), sempre
//                  dietro al puntatore di lettura;
//                - ultima passata: l'uscita va su m_axis_out.
//              Finita la sequenza: m <- m+1 e si torna a ST_LOAD,
//              oppure, se era l'ultima passata, ST_TAIL.
//     ST_TAIL  aspetta che l'ultimo beat sia uscito
//     ST_DONE  un ciclo, poi di nuovo ST_IDLE
//
//   Memoria: il seqbuf e' L*H = 131072 parole x 16 bit = 2 Mbit
//   (~64 BRAM36).
//=====================================================================

module s4d_core_axis #(
    // ---- dimensioni del modello ----
    parameter integer H        = 128,    // canali del modello
    parameter integer LOG2_H   = 7,
    parameter integer L        = 1024,   // sCIFAR (era 4096/Speech Commands)
    parameter integer LOG2_L   = 10,      // sCIFAR (era 12/Speech Commands)
    parameter integer NLAYER   = 6,      // strati S4D (tutti sullo stesso layer fisico)

    // ---- organizzazione del banco di biquad ----
    parameter integer NUNITS   = 32,     // unita' biquad in parallelo (era 8; 2026-10-01 ZCU104)
    parameter integer LOG2_NU  = 5,      // (era 3)
    parameter integer MODES    = 32,     // modi (coppie di poli) per canale
    parameter integer LOG2_MOD = 5,
    parameter integer NCH      = 4,      // canali serviti da ogni unita' (NUNITS*NCH = H), era 16
    parameter integer LOG2_NCH = 2,      // (era 4)
    parameter integer LOG2_NB  = LOG2_MOD + LOG2_NCH,   // = 7: NCH*MODES = 128 biquad per unita' (era 512)
    // ---- mixing lineare ----
    parameter integer NMAC     = 256,  // 2026-10-01 (ZCU104): un MAC per uscita del mixing,
                                        // una sola fase (~131 cicli/frame invece di ~641).
                                        // Era 64 (4 fasi, ~39ms/img sulla Red Pitaya).
    parameter integer LOG2_NPH = 0,    // una sola fase: TH/NMAC = 256/256 = 1 (era 2)
    parameter         WMEM_STYLE = "ultra",   // memoria pesi del mixing (URAM su ZCU104)
    // LUT di GELU e sigmoide (lette in sintesi): un file per dataset, perche'
    // la GELU dipende da F_SSM (sCIFAR 8, ECG e Speech Commands 9)
    parameter         GELU_FILE  = "gelu_lut_ECG.mem",
    parameter         SIG_FILE   = "sigmoid_lut_ECG.mem",

    // ---- formati a virgola fissa (W_* = larghezza, F_* = bit frazionari) ----
    parameter integer W_ACT    = 16,     // attivazioni tra i layer (Q5.11)
    parameter integer F_ACT    = 11,
    parameter integer W_U      = 16,     // ingresso del banco biquad (uscita BatchNorm)
    parameter integer F_U      = 11,
    parameter integer W_A      = 24,     // coefficienti a1,a2 dei biquad (denominatore)
    parameter integer F_A      = 22,
    parameter integer W_B      = 16,     // coefficienti b0,b1 dei biquad (numeratore)
    parameter integer F_B      = 15,
    parameter integer W_S      = 32,     // stato dei biquad e uscita y di ogni modo
    parameter integer F_Y      = 19,
    parameter integer W_D      = 16,     // coefficiente D (skip connection u*D)
    parameter integer F_D      = 13,
    parameter integer W_BSH    = 5,      // shift per canale dei coefficienti b (esponente comune)
    parameter integer W_BACC   = 40,     // accumulatore della somma sui modi
    parameter integer W_SSM    = 16,     // uscita del banco biquad / ingresso GELU
    parameter integer F_SSM    = 8,       // dalla quantizzazione sCIFAR
                                          // (mem_trained_bn/quant_report.json);
                                          // ricontrollare se si riaddestra/
                                          // riquantizza il modello
    parameter integer W_W      = 12,     // pesi del mixing lineare
    parameter integer F_W      = 11,
    parameter integer W_MACC   = 32,     // accumulatori del mixing (e bias del mixing)
    parameter integer F_SIGI   = 11,     // ingresso della sigmoide (gate del GLU)
    parameter integer F_SIGO   = 14,     // uscita della sigmoide

    // ---- BatchNorm (s4d_affine) ----
    parameter integer BN_W_W   = 16,     // peso (scala) BN
    parameter integer BN_W_B   = 32,     // bias BN
    parameter integer BN_W_ACC = 34,     // accumulatore BN
    parameter integer FIFO_LOG2 = 10,    // profondita' FIFO del residual = 2^10

    parameter integer AXI_DW   = 64      // larghezza dello stream dei pesi
)(
    input  wire                    clk,
    input  wire                    rst_n,            // reset asincrono attivo basso

    // ---- sequenza in ingresso dall'encoder: L*H beat, tlast sull'ultimo ----
    input  wire                    s_axis_enc_tvalid,
    output wire                    s_axis_enc_tready,
    input  wire signed [W_ACT-1:0] s_axis_enc_tdata,
    input  wire                    s_axis_enc_tlast,
    // stream dei pesi: 6 layer consecutivi, in ordine
    input  wire                    s_axis_w_tvalid,
    output wire                    s_axis_w_tready,
    input  wire [AXI_DW-1:0]       s_axis_w_tdata,
    input  wire                    s_axis_w_tlast,

    // ---- uscita dell'ultimo layer: L*H beat, tlast sull'ultimo ----
    output wire                    m_axis_out_tvalid,
    input  wire                    m_axis_out_tready,
    output wire signed [W_ACT-1:0] m_axis_out_tdata,
    output wire                    m_axis_out_tlast,

    // Flag di stato, sticky. Erano reg/wire interni: un errore che
    // nessuno puo' leggere non protegge da niente.
    //   [0] err_out_ovr  uscita sovrascritta (ora NON deve mai scattare)
    //   [1] err_enc_len  lunghezza della sequenza in ingresso errata
    //   [2] err_ovw      la scrittura ha sorpassato la lettura nel seqbuf
    //   [3] layer_ovf    overflow aritmetico nel banco di biquad
    //   [4] err_bank     ready rotto fra banco e mixing (sentinella)
    output wire [4:0]              status_err,

    // Riallineamento dello streamer dei pesi (s4d_weightrom_axis).
    // wload_start e' LO STESSO filo che avvia u_wload: lo streamer salta
    // all'inizio del layer wload_layer nello stesso ciclo, prima che
    // s4d_wload alzi tready (ciclo dopo).
    output wire                    wload_start,   // impulso: inizia il caricamento di un layer
    output wire [2:0]              wload_layer    // layer che sta per essere caricato (0..NLAYER-1)
);

    localparam integer NWORDS = L * H;              // parole nel seqbuf (131072)
    localparam integer AW     = LOG2_L + LOG2_H;    // bit di indirizzo del seqbuf (17)

    localparam [AW-1:0] ADDR_LAST = NWORDS - 1;     // ultimo indirizzo della sequenza

    //=================================================================
    // RESET LOCALE (timing 2026-10-02)
    //   Prima rst_n (peripheral_aresetn del Processor System Reset) andava
    //   direttamente a decine di migliaia di flip-flop: Vivado replicava i
    //   registri d'uscita dell'IP di reset e quei percorsi erano fra i
    //   piu' lenti (slack misurato 0.47 ns a 177.8 MHz).
    //   Ora rst_n entra solo in 4 "ponti" di reset, uno per blocco:
    //     - attivazione ASINCRONA immediata (come prima: rst_n basso
    //       azzera subito tutto);
    //     - rilascio SINCRONO, 2 cicli dopo la salita di rst_n, uguale
    //       per tutti i ponti (quindi tutti i blocchi ripartono insieme,
    //       e insieme allo streamer dei pesi, che ha lo stesso ponte).
    //   keep: impedisce alla sintesi di riunire i 4 ponti in uno solo.
    //=================================================================
    (* keep = "true" *) reg [1:0] rst_bridge_top;     // logica di questo file
    (* keep = "true" *) reg [1:0] rst_bridge_wload;   // u_wload
    (* keep = "true" *) reg [1:0] rst_bridge_buf;     // u_buf (seqbuf)
    (* keep = "true" *) reg [1:0] rst_bridge_layer;   // u_layer

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rst_bridge_top   <= 2'b00;
            rst_bridge_wload <= 2'b00;
            rst_bridge_buf   <= 2'b00;
            rst_bridge_layer <= 2'b00;
        end else begin
            rst_bridge_top   <= {rst_bridge_top[0],   1'b1};
            rst_bridge_wload <= {rst_bridge_wload[0], 1'b1};
            rst_bridge_buf   <= {rst_bridge_buf[0],   1'b1};
            rst_bridge_layer <= {rst_bridge_layer[0], 1'b1};
        end
    end

    wire rst_top_n   = rst_bridge_top[1];     // reset locale, attivo basso
    wire rst_wload_n = rst_bridge_wload[1];
    wire rst_buf_n   = rst_bridge_buf[1];
    wire rst_layer_n = rst_bridge_layer[1];

    // stati della macchina principale (vedi intestazione)
    localparam [2:0] ST_IDLE = 3'd0,
                     ST_ENC  = 3'd1,
                     ST_LOAD = 3'd2,
                     ST_PASS = 3'd3,
                     ST_TAIL = 3'd4,
                     ST_DONE = 3'd5;

   //(* mark_debug = "true" *) reg [2:0]     st;
   //(* mark_debug = "true" *) reg [2:0]     m;

   reg [2:0]     st;                 // stato corrente
   reg [2:0]     m;                  // indice della passata/layer in corso (0..NLAYER-1)
   reg [AW-1:0]  rptr, wptr;         // puntatori di lettura / scrittura nel seqbuf
   reg           rd_run;             // 1 = restano parole da leggere nella passata

   //(* mark_debug = "true" *) reg [2:0]     st;
   //(* mark_debug = "true" *) reg [2:0]     m;
   //(* mark_debug = "true" *) reg [AW-1:0]  rptr, wptr;
   //(* mark_debug = "true" *) reg           rd_run;



    // 1 = siamo nell'ultima passata: l'uscita del layer va su m_axis_out
    wire last_pass = (m == NLAYER[2:0] - 3'd1);
    //(* mark_debug = "true" *) wire                     wl_busy, wl_done;

    //-----------------------------------------------------------------
    // Segnali tra u_wload e u_layer (scrittura dei pesi nelle memorie)
    //-----------------------------------------------------------------
    wire                     wl_busy, wl_done;   // caricamento in corso / finito (impulso)
    reg                      wl_start;           // impulso: carica i pesi del prossimo layer
    // coefficienti dei biquad (a1,a2,b0,b1) -> s4d_biquad_bank
    wire                     cfg_we;
    wire [LOG2_NU-1:0]       cfg_unit;
    wire [LOG2_NB-1:0]       cfg_addr;
    wire [2*W_A+2*W_B-1:0]   cfg_data;
    // coefficiente D e shift dei b, per canale -> s4d_biquad_bank
    wire                     cfgd_we;
    wire [LOG2_H-1:0]        cfgd_ch;
    wire signed [W_D-1:0]    cfgd_data;
    wire [W_BSH-1:0]         cfgd_bsh;
    // BatchNorm: shift, pesi e bias per canale -> s4d_affine
    wire                     ld_bn_we_sh;
    wire [5:0]               ld_bn_sh;
    wire                     ld_bn_we_w, ld_bn_we_b;
    wire [LOG2_H-1:0]        ld_bn_addr;
    wire signed [BN_W_W-1:0] ld_bn_w;
    wire signed [BN_W_B-1:0] ld_bn_b;
    // mixing lineare: pesi (a righe da 64 bit) e bias -> s4d_mixing_glu
    wire                     ld_wo_we, ld_bo_we;
    wire [LOG2_NPH+LOG2_H-1:0] ld_wo_addr;
    wire [5:0]               ld_wo_sel;   // era [3:0]: 48 banchi da 64 bit
    wire [AXI_DW-1:0]        ld_wo_data;
    wire [LOG2_H:0]          ld_bo_addr;
    wire signed [W_MACC-1:0] ld_bo_data;

    //=================================================================
    // Caricatore dei pesi: a ogni wl_start consuma dallo stream
    // s_axis_w i beat di UN layer e li smista nelle memorie del layer
    //=================================================================
    s4d_wload_axis #(
        .H(H), .LOG2_H(LOG2_H), .NUNITS(NUNITS), .LOG2_NU(LOG2_NU),
        .LOG2_NB(LOG2_NB), .NMAC(NMAC), .LOG2_NPH(LOG2_NPH),
        .W_A(W_A), .W_B(W_B), .W_D(W_D), .W_BSH(W_BSH),
        .BN_W_W(BN_W_W), .BN_W_B(BN_W_B), .W_MACC(W_MACC), .W_W(W_W), .AXI_DW(AXI_DW),
        .WO_SEL_W(6)                    // 48 beat per riga dei pesi del mixing
    ) u_wload (
        .clk(clk), .rst_n(rst_wload_n),        // reset locale (vedi RESET LOCALE)
        .start(wl_start),
        .busy(wl_busy), .done(wl_done),.s_axis_w_tvalid(s_axis_w_tvalid),
        .s_axis_w_tready(s_axis_w_tready),
        .s_axis_w_tdata (s_axis_w_tdata),
        .s_axis_w_tlast (s_axis_w_tlast),
        .cfg_we(cfg_we), .cfg_unit(cfg_unit), .cfg_addr(cfg_addr),
        .cfg_data(cfg_data),
        .cfgd_we(cfgd_we), .cfgd_ch(cfgd_ch), .cfgd_data(cfgd_data),
        .cfgd_bsh(cfgd_bsh),
        .ld_bn_we_sh(ld_bn_we_sh), .ld_bn_sh(ld_bn_sh),
        .ld_bn_we_w(ld_bn_we_w), .ld_bn_we_b(ld_bn_we_b),
        .ld_bn_addr(ld_bn_addr), .ld_bn_w(ld_bn_w), .ld_bn_b(ld_bn_b),
        .ld_wo_we(ld_wo_we), .ld_wo_addr(ld_wo_addr), .ld_wo_sel(ld_wo_sel),
        .ld_wo_data(ld_wo_data),
        .ld_bo_we(ld_bo_we), .ld_bo_addr(ld_bo_addr), .ld_bo_data(ld_bo_data)
    );

    wire en = 1'b1;                 // il core parte quando arrivano i dati
    reg  img_done;                  // solo uso interno
    //(* mark_debug = "true" *) wire layer_ovf, err_bank, err_ovw;
    wire layer_ovf, err_bank, err_ovw;   // flag di errore dai sottomoduli
    wire lay_out_ready;                  // ready verso l'uscita del layer
    reg  err_out_ovr, err_enc_len;       // flag di errore generati qui

    // beat accettato dall'encoder; si accetta solo durante ST_ENC
    // (s_axis_enc_tready dipende solo dal registro di stato)
    wire enc_beat = s_axis_enc_tvalid && s_axis_enc_tready;
    assign s_axis_enc_tready = (st == ST_ENC);

    //-----------------------------------------------------------------
    // Porte del seqbuf
    //-----------------------------------------------------------------
    wire                    buf_wr_en;
    wire [AW-1:0]           buf_wr_addr;
    wire signed [W_ACT-1:0] buf_wr_data;
    wire                    buf_rd_en;
    wire signed [W_ACT-1:0] buf_rd_data;   // disponibile RD_LAT cicli dopo buf_rd_en

    //-----------------------------------------------------------------
    // Lettura del seqbuf verso il layer: 1 parola per ciclo.
    //   do_rd       : lancia una lettura all'indirizzo rptr
    //   rd_vpipe/   : valid e flag "ultima parola" lungo la latenza del
    //   rd_lpipe      seqbuf (RD_LAT cicli)
    //   rdq_*       : coda di RDQ_DEPTH parole che presenta i dati al layer
    //   rd_credits  : letture in volo + parole in coda; si lancia una
    //                 lettura solo se c'e' posto garantito in coda
    //                 -> la coda non trabocca mai, anche se il layer stalla.
    //   Profondita' >= RD_LAT+2: il giro completo lettura -> coda -> layer
    //   dura RD_LAT+1 cicli, quindi a regime le letture non si fermano.
    //   (2026-10-05: RD_LAT 2 -> 3 per la cascata di BRAM del seqbuf, coda
    //   4 -> 8 posti. Versione precedente: s4d_core_axi.v.bak3.)
    //   (Prima: rd_inflight + registro sk, una lettura ogni 2 cicli, che
    //   bastava solo finche' il banco biquad impiegava 512 cicli/frame.)
    //-----------------------------------------------------------------
    localparam integer RD_LAT    = 3;     // latenza del seqbuf (uscita BRAM + mux + registro)
    localparam integer RDQ_DEPTH = 8;     // posti della coda (>= RD_LAT + 2)
    localparam integer RDQ_IDX_W = 3;     // bit degli indici della coda (8 posti)
    localparam integer RDQ_CNT_W = 4;     // bit dei contatori (0..8)

    initial if ((RD_LAT < 2) || (RDQ_DEPTH < RD_LAT + 2) || ((1 << RDQ_IDX_W) != RDQ_DEPTH)) begin
        $display("s4d_core_axis: parametri di lettura del seqbuf incoerenti (RD_LAT=%0d, RDQ_DEPTH=%0d)", RD_LAT, RDQ_DEPTH);
        $finish;
    end

    reg  [RD_LAT-1:0]       rd_vpipe;     // letture in volo (bit RD_LAT-1 = arriva ora)
    reg  [RD_LAT-1:0]       rd_lpipe;     // ... e se sono l'ultima parola della sequenza

    reg  signed [W_ACT-1:0] rdq_data [0:RDQ_DEPTH-1];
    reg                     rdq_last [0:RDQ_DEPTH-1];
    reg  [RDQ_IDX_W-1:0]    rdq_wr_idx, rdq_rd_idx;   // puntatori della coda
    reg  [RDQ_CNT_W-1:0]    rdq_count;                // parole in coda (0..RDQ_DEPTH)
    reg  [RDQ_CNT_W-1:0]    rd_credits;               // in volo + in coda (0..RDQ_DEPTH)

    wire lay_in_ready;                                // il layer accetta un campione

    wire rdq_valid  = (rdq_count != {RDQ_CNT_W{1'b0}});   // c'e' un campione per il layer
    wire rdq_pop    = rdq_valid && lay_in_ready;      // il layer lo prende
    wire rd_arrive  = rd_vpipe[RD_LAT-1];             // il dato di una lettura arriva ora

    // si legge se restano parole e c'e' posto garantito in coda
    wire do_rd = rd_run && (rd_credits < RDQ_DEPTH);

    assign buf_rd_en = do_rd;

    //=================================================================
    // Buffer della sequenza (Block RAM, dual port, lettura in RD_LAT cicli)
    //   chk_en attiva il controllo "la scrittura non deve raggiungere
    //   la lettura" solo durante le passate
    //=================================================================
    s4d_seqbuf #(.W(W_ACT), .NWORDS(NWORDS), .AW(AW), .RD_LAT(RD_LAT)) u_buf (
        .clk(clk), .rst_n(rst_buf_n),        // reset locale (vedi RESET LOCALE)
        .wr_en(buf_wr_en), .wr_addr(buf_wr_addr), .wr_data(buf_wr_data),
        .rd_en(buf_rd_en), .rd_addr(rptr), .rd_data(buf_rd_data),
        .chk_en((st == ST_PASS) && rd_run), .err_ovw(err_ovw)
    );

    // gestione delle letture in volo e della coda
    always @(posedge clk or negedge rst_top_n) begin
        if (!rst_top_n) begin
            rd_vpipe   <= {RD_LAT{1'b0}};
            rd_lpipe   <= {RD_LAT{1'b0}};
            rdq_wr_idx <= {RDQ_IDX_W{1'b0}};
            rdq_rd_idx <= {RDQ_IDX_W{1'b0}};
            rdq_count  <= {RDQ_CNT_W{1'b0}};
            rd_credits <= {RDQ_CNT_W{1'b0}};
        end else if (st != ST_PASS) begin
            // fuori dalle passate niente letture pendenti (come prima)
            rd_vpipe   <= {RD_LAT{1'b0}};
            rd_lpipe   <= {RD_LAT{1'b0}};
            rdq_wr_idx <= {RDQ_IDX_W{1'b0}};
            rdq_rd_idx <= {RDQ_IDX_W{1'b0}};
            rdq_count  <= {RDQ_CNT_W{1'b0}};
            rd_credits <= {RDQ_CNT_W{1'b0}};
        end else begin
            // valid e "ultima parola" avanzano lungo la latenza del seqbuf
            rd_vpipe <= {rd_vpipe[RD_LAT-2:0], do_rd};
            rd_lpipe <= {rd_lpipe[RD_LAT-2:0], do_rd && (rptr == ADDR_LAST)};

            // il dato arrivato entra in coda
            if (rd_arrive) rdq_wr_idx <= rdq_wr_idx + 1'b1;
            // il layer prende la testa
            if (rdq_pop)   rdq_rd_idx <= rdq_rd_idx + 1'b1;

            rdq_count  <= rdq_count  + (rd_arrive ? 1'b1 : 1'b0) - (rdq_pop ? 1'b1 : 1'b0);
            rd_credits <= rd_credits + (do_rd     ? 1'b1 : 1'b0) - (rdq_pop ? 1'b1 : 1'b0);
        end
    end

    // dati della coda (niente reset)
    always @(posedge clk)
        if (rd_arrive) begin
            rdq_data[rdq_wr_idx] <= buf_rd_data;
            rdq_last[rdq_wr_idx] <= rd_lpipe[RD_LAT-1];
        end

    // verso il layer: testa della coda (stessi nomi di prima, ora fili)
    wire                    sk_valid = rdq_valid;
    wire signed [W_ACT-1:0] sk_data  = rdq_data[rdq_rd_idx];
    wire                    sk_last  = rdq_last[rdq_rd_idx];

    // uscita del layer
    wire                    lay_valid;
    wire signed [W_ACT-1:0] lay_data;
    wire                    lay_last_seq;

    //=================================================================
    // Lo strato S4D fisico, riusato per tutte le NLAYER passate.
    //   LOADABLE=1: pesi scritti a runtime da u_wload (niente .mem)
    //   BN_SH=0   : lo shift del BatchNorm arriva anch'esso dallo stream
    //   Le LUT di GELU e sigmoide sono uguali per tutti i layer: .mem
    //=================================================================
    s4d_layer #(
        .H(H), .LOG2_H(LOG2_H), .NUNITS(NUNITS), .LOG2_NU(LOG2_NU),
        .MODES(MODES), .LOG2_MOD(LOG2_MOD), .NCH(NCH), .LOG2_NCH(LOG2_NCH),
        .LOG2_NB(LOG2_NB), .NMAC(NMAC), .LOG2_NPH(LOG2_NPH),
        .W_ACT(W_ACT), .F_ACT(F_ACT),
        .W_U(W_U), .F_U(F_U), .W_A(W_A), .F_A(F_A), .W_B(W_B), .F_B(F_B),
        .W_S(W_S), .F_Y(F_Y), .W_D(W_D), .F_D(F_D), .W_BSH(W_BSH),
        .W_BACC(W_BACC), .W_SSM(W_SSM), .F_SSM(F_SSM),
        .W_W(W_W), .F_W(F_W), .W_MACC(W_MACC),
        .F_SIGI(F_SIGI), .F_SIGO(F_SIGO),
        .FIFO_LOG2(FIFO_LOG2),
        .LOADABLE(1), .LD_W(AXI_DW), .BN_SH(0),
        .WMEM_STYLE(WMEM_STYLE),
        .BN_W_W(BN_W_W), .BN_W_B(BN_W_B), .BN_W_ACC(BN_W_ACC),
        .GELU_FILE(GELU_FILE),
        .SIG_FILE (SIG_FILE)
    ) u_layer (
        .clk(clk), .rst_n(rst_layer_n),        // reset locale (vedi RESET LOCALE)
        .cfg_we(cfg_we), .cfg_unit(cfg_unit), .cfg_addr(cfg_addr),
        .cfg_data(cfg_data),
        .cfgd_we(cfgd_we), .cfgd_ch(cfgd_ch), .cfgd_data(cfgd_data),
        .cfgd_bsh(cfgd_bsh),
        .ld_bn_we_sh(ld_bn_we_sh), .ld_bn_sh(ld_bn_sh),
        .ld_bn_we_w(ld_bn_we_w), .ld_bn_we_b(ld_bn_we_b),
        .ld_bn_addr(ld_bn_addr), .ld_bn_w(ld_bn_w), .ld_bn_b(ld_bn_b),
        .ld_wo_we(ld_wo_we), .ld_wo_addr(ld_wo_addr), .ld_wo_sel(ld_wo_sel),
        .ld_wo_data(ld_wo_data),
        .ld_bo_we(ld_bo_we), .ld_bo_addr(ld_bo_addr), .ld_bo_data(ld_bo_data),
        .in_valid(sk_valid), .in_ready(lay_in_ready), .in_data(sk_data),
        .in_last_seq(sk_valid && sk_last),
        .out_ready(lay_out_ready),
        .out_valid(lay_valid), .out_data(lay_data),
        .out_last_ch(), .out_last_seq(lay_last_seq),
        .ovf(layer_ovf), .err_bank_ovr(err_bank)
    );

    //-----------------------------------------------------------------
    // Scrittura nel seqbuf
    //   ST_ENC : dati dell'encoder
    //   ST_PASS: uscita del layer (non nell'ultima passata), scritta
    //            "in place" a wptr, che resta sempre dietro a rptr
    //-----------------------------------------------------------------
    assign buf_wr_en   = (st == ST_ENC)  ? enc_beat
                       : (st == ST_PASS) ? (lay_valid && !last_pass)
                       : 1'b0;
    assign buf_wr_addr = wptr;
    assign buf_wr_data = (st == ST_ENC) ? s_axis_enc_tdata : lay_data;

    //(* mark_debug = "true" *) wire dec_fire = lay_valid && last_pass && (st == ST_PASS);
    // nell'ultima passata il layer ha un dato valido per l'uscita
    wire dec_fire = lay_valid && last_pass && (st == ST_PASS);

    // ------------------------------------------------------------------
    // SKID BUFFER di uscita (2 posizioni).
    //
    // Sostituisce il registro singolo, che sovrascriveva il dato quando
    // il consumatore stallava (perdita silenziosa, segnalata solo da
    // err_out_ovr).
    //
    // sk_ready e' REGISTRATO (= !skid_v), non combinatorio da tready:
    // evita di allungare il percorso critico dal DMA fino al layer.
    // Throughput pieno quando non c'e' stallo: con skid_v=0 il beat va
    // direttamente in uscita, 1 per ciclo.
    // ------------------------------------------------------------------

    //(* mark_debug = "true" *) reg                    out_v,  skid_v;
    reg                    out_v,  skid_v;   // valid: registro d'uscita / registro di riserva
    reg signed [W_ACT-1:0] out_d,  skid_d;   // dati
    reg                    out_l,  skid_l;   // tlast

    // si accettano beat dal layer finche' il registro di riserva e' vuoto
    wire sk_out_ready = !skid_v;

    // Il layer va stallato SOLO nell'ultimo pass: negli altri l'uscita
    // finisce nel seqbuf, che accetta sempre.
    assign lay_out_ready = last_pass ? sk_out_ready : 1'b1;

    assign m_axis_out_tvalid = out_v;
    assign m_axis_out_tdata  = out_d;
    assign m_axis_out_tlast  = out_l;

    always @(posedge clk or negedge rst_top_n) begin
        if (!rst_top_n) begin
            out_v       <= 1'b0;
            out_d       <= {W_ACT{1'b0}};
            out_l       <= 1'b0;
            skid_v      <= 1'b0;
            skid_d      <= {W_ACT{1'b0}};
            skid_l      <= 1'b0;
            err_out_ovr <= 1'b0;
            err_enc_len <= 1'b0;
        end else begin
            if (!out_v || m_axis_out_tready) begin
                // lo stadio di uscita si libera: pesca dallo skid se pieno,
                // altrimenti direttamente dall'ingresso
                if (skid_v) begin
                    out_v  <= 1'b1;
                    out_d  <= skid_d;
                    out_l  <= skid_l;
                    skid_v <= 1'b0;
                end else begin
                    out_v  <= dec_fire;
                    out_d  <= lay_data;
                    out_l  <= lay_last_seq;
                end
            end else if (dec_fire && !skid_v) begin
                // uscita bloccata: il beat entrante va nello skid
                skid_v <= 1'b1;
                skid_d <= lay_data;
                skid_l <= lay_last_seq;
            end

            // Sentinella di sovrascrittura.
            // ATTENZIONE: dec_fire e' un VALID, non un fire: il layer
            // tiene alto out_valid per tutta la durata dello stallo, come
            // vuole AXI-Stream. Il beat viene davvero ACCETTATO solo con
            // lay_out_ready alto, quindi la condizione di perdita e'
            // "accettato mentre entrambi gli stadi sono pieni".
            // Con lay_out_ready = !skid_v questo e' strutturalmente
            // impossibile: se scatta, qualcuno ha rotto il ready.
            if (dec_fire && lay_out_ready && skid_v
                         && out_v && !m_axis_out_tready)
                err_out_ovr <= 1'b1;

            // tlast dell'encoder deve arrivare esattamente sull'ultimo
            // indirizzo (L*H-esimo beat): altrimenti sequenza di lunghezza errata
            if (enc_beat && (s_axis_enc_tlast != (wptr == ADDR_LAST)))
                err_enc_len <= 1'b1;
        end
    end

    assign status_err = {err_bank, layer_ovf, err_ovw,
                         err_enc_len, err_out_ovr};

    // Uscite verso lo streamer dei pesi.
    // wl_start e' un registro (impulso di 1 ciclo); m viene aggiornato
    // sullo stesso fronte in cui wl_start si alza (ST_PASS: m <= m+1,
    // wl_start <= 1; ST_ENC: m e' gia' 0), quindi nel ciclo in cui
    // wl_start vale 1, m indica gia' il layer da caricare.
    assign wload_start = wl_start;
    assign wload_layer = m;

    //=================================================================
    // Macchina a stati principale
    //=================================================================
    always @(posedge clk or negedge rst_top_n) begin
        if (!rst_top_n) begin
            st       <= ST_IDLE;
            m        <= 3'd0;
            rptr     <= {AW{1'b0}};
            wptr     <= {AW{1'b0}};
            rd_run   <= 1'b0;
            wl_start <= 1'b0;
            img_done <= 1'b0;
        end else begin
            // impulsi di un ciclo
            wl_start <= 1'b0;
            img_done <= 1'b0;

            case (st)

                // attesa: riparte dal layer 0 appena l'encoder manda dati
                ST_IDLE: begin
                    wptr <= {AW{1'b0}};
                    m    <= 3'd0;
                    if (en && s_axis_enc_tvalid) st <= ST_ENC;
                end

                // scrittura della sequenza dell'encoder nel seqbuf;
                // all'ultimo beat parte il caricamento dei pesi del layer 0
                ST_ENC: begin
                    if (enc_beat) begin
                        if (wptr == ADDR_LAST) begin
                            wptr     <= {AW{1'b0}};
                            wl_start <= 1'b1;
                            st       <= ST_LOAD;
                        end else begin
                            wptr <= wptr + 1'b1;
                        end
                    end
                end

                // attesa del caricamento pesi, poi si prepara la passata
                ST_LOAD: begin
                    if (wl_done) begin
                        rptr   <= {AW{1'b0}};
                        wptr   <= {AW{1'b0}};
                        rd_run <= 1'b1;
                        st     <= ST_PASS;
                    end
                end

                // una passata del layer su tutta la sequenza
                ST_PASS: begin
                    // avanzamento della lettura
                    if (do_rd) begin
                        if (rptr == ADDR_LAST) rd_run <= 1'b0;
                        else                   rptr <= rptr + 1'b1;
                    end

                    // avanzamento della scrittura in place (non nell'ultima passata)
                    if (lay_valid && !last_pass) begin
                        if (wptr != ADDR_LAST) wptr <= wptr + 1'b1;
                    end

                     // il layer ha emesso l'ultimo campione della sequenza
						if (lay_last_seq && lay_out_ready) begin
							if (last_pass) begin
								st <= ST_TAIL;
							end else begin
								m        <= m + 3'd1;      // layer successivo
								wl_start <= 1'b1;          // carica i suoi pesi
								st       <= ST_LOAD;
							end
						end
					end

                // aspetta che l'ultimo beat lasci il registro d'uscita
                ST_TAIL: if (!out_v) st <= ST_DONE;

                ST_DONE: begin
                    img_done <= 1'b1;
                    st       <= ST_IDLE;
                end

                default: st <= ST_IDLE;

            endcase
        end
    end

endmodule

`default_nettype wire
