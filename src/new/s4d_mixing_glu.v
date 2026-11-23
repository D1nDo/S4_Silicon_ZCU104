`timescale 1ns / 1ps
`default_nettype none
//=====================================================================
// s4d_mixing_glu  --  mixing lineare H -> 2H seguito da GLU
//                     (versione a FASE SINGOLA: NMAC = 2H = 256)
//
//   Per ogni frame (H campioni x[0..H-1]):
//
//     acc[c] = bias[c] + sum_{h<H} W[c][h] * x[h]        c = 0..2H-1
//     a[i]   = acc[i]          (prima meta')     i = 0..H-1
//     g[i]   = acc[H+i]        (seconda meta')
//     out[i] = sat( a[i] * sigmoid(g[i]) )                (GLU)
//
//   Struttura:
//
//     ingresso --> g_buf (2 frame) --> MOTORE MAC --> acc[0..2H-1] --> SNAPSHOT --> GLU --> uscita
//                                      H cicli,       un MAC per       a_snap/g_snap  1 per ciclo
//                                      3 stadi        ogni uscita c    (scorrimento)  (a crediti)
//
//   - Un MAC per uscita: il MAC c aggiorna solo acc[c] e ha il proprio
//     bias in un registro -> nessun multiplexer per fase/uscita.
//   - Pesi: la riga h contiene i 2H pesi W[c][h] (c = 0 negli LSB,
//     NMAC*W_W = 3072 bit), divisa in NBANK = 48 banchi da 64 bit,
//     ciascuno profondo H. Una riga letta per ciclo.
//   - Pipeline MAC (7 stadi): t indirizzi ai gruppi di banchi e x[h] letto
//     -> t+1 indirizzi ai banchi (x ritardato) -> t+2 riga di pesi letta
//     -> t+3 riga nel registro d'uscita della memoria e x copiato per
//     gruppo di 16 MAC -> t+4 peso e x nei registri d'ingresso del MAC
//     -> t+5 prodotto registrato -> t+6 accumulo
//     (forma adatta a DSP48E2: A/B = ingressi, M = prodotto, P = acc).
//   - Snapshot: a fine frame acc viene copiato in due registri a
//     scorrimento; la GLU legge sempre la testa (indice 0). Il motore
//     MAC intanto elabora il frame successivo.
//   Throughput: ~H + 7 = 135 cicli per frame senza stallo in uscita
//   (versione precedente a 4 fasi con NMAC=64: ~641 cicli).
//
//   2026-10-15 (fan-out dello stallo d'uscita e URAM):
//     - ramo GLU A SCORRIMENTO LIBERO: niente piu' enable oen su Q1..G3,
//       sigmoide e registro d'uscita. Lo stallo agisce solo sull'EMISSIONE
//       dalla testa dello snapshot, tramite crediti: si emette solo se i
//       risultati "in volo" (emessi e non ancora usciti dalla coda
//       d'uscita del layer, segnalati da tail_pop) sono meno di
//       TAIL_DEPTH-2. La coda del layer (TAIL_DEPTH posti) non trabocca mai.
//       Prima: oen = out_ready pilotava ~8000 enable (snapshot) + ~800
//       registri della pipeline GLU e del residual.
//     - snapshot: controllo replicato per gruppi di SNAP_GROUP posizioni
//       (copie registrate di glu_active, glu_last e issue_ok, dont_touch),
//       cosi' ogni enable pilota 2*SNAP_GROUP*W_ACC flip-flop invece di
//       2*H*W_ACC = 8192.
//     - pesi: con WMEM_STYLE = "ultra" i banchi sono istanze
//       xpm_memory_sdpram con MEMORY_PRIMITIVE = "ultra" (Vivado rifiutava
//       ram_style = "ultra" sui banchi da 128 righe: Synth 8-6849, ripiego
//       su LUTRAM, 48 x 20 RAM64M8). Latenza di lettura 2 (registro OREG
//       della URAM) per TUTTI gli stili; stadio A3 aggiunto per riallineare
//       x ai pesi. In simulazione senza librerie Xilinx (Icarus) definire
//       S4D_BEHAV_MEM: si usa un modello comportamentale equivalente.
//     Aritmetica identica; versione precedente in S4D_RTL/new_pre_oen_uram.
//
//   2026-10-15 (timing, al massimo un'operazione complessa per ciclo):
//     - registri d'ingresso dei MAC (A/B del DSP) tra la riga di pesi e
//       il moltiplicatore;
//     - ramo GLU: riquantizzazione in 2 stadi (somma, saturazione), la
//       sigmoide porta con se' a e il flag di fine frame (tag della PWL),
//       prodotto a*sigmoid(g) in 3 stadi (prodotto, mezzo LSB, saturazione).
//     Aritmetica identica; versione precedente in S4D_RTL/new_pre_pipeline.
//
//   2026-10-02 (timing): copie locali di x[h], dei flag dei MAC e degli
//   indirizzi/dati dei banchi di pesi, per ridurre il fan-out misurato
//   a 150 MHz (x[h] -> 256 DSP, mac_h -> ~6150 celle). Versione prima di
//   questa modifica: mixing_glu.v.bak (sul Desktop).
//=====================================================================

module s4d_mixing_glu #(
    parameter integer H          = 128,   // ingressi (e uscite del GLU)
    parameter integer LOG2_H     = 7,
    parameter integer NMAC       = 256,   // = 2H: un MAC per ogni uscita, una sola fase
    parameter integer LOG2_NPH   = 0,     // fasi = 1 -> 0 bit (resta per compatibilita' con s4d_layer)
    parameter integer W_IN       = 16,    // ingresso x (Q.F_IN)
    parameter integer F_IN       = 11,
    parameter integer W_W        = 12,    // pesi (Q.F_W)
    parameter integer F_W        = 11,
    parameter integer W_ACC      = 32,    // accumulatori e bias (Q.F_W+F_IN)
    parameter integer W_SIG      = 16,    // ingresso/uscita della sigmoide
    parameter integer F_SIGI     = 11,    // bit frazionari del gate g in ingresso alla sigmoide
    parameter integer F_SIGO     = 14,    // bit frazionari dell'uscita della sigmoide
    parameter integer W_A        = 16,    // valore a (Q.F_A)
    parameter integer F_A        = 11,
    parameter integer W_OUT      = 16,    // uscita (Q.F_OUT)
    parameter integer F_OUT      = 11,

    parameter integer LOADABLE   = 0,     // 1 = pesi/bias scritti dalle porte ld_*
    parameter integer LD_W       = 64,    // larghezza di un banco di pesi (= beat dello stream)
    parameter integer LOG2_NBANK = 6,     // bit di ld_sel_w: NBANK = NMAC*W_W/LD_W = 48
    parameter         WMEM_STYLE = "ultra",  // memoria dei pesi: "ultra" | "block" | "distributed"
    parameter integer TAIL_DEPTH = 32,    // posti della coda d'uscita del layer (crediti di emissione)
    parameter integer SNAP_GROUP = 2,     // posizioni dello snapshot servite da una copia del controllo
    // riusato per NLAYER ruoli (vedi s4d_layer.v): default = layer 0, un file
    // vero, non un placeholder inventato -- chi lo istanzia sovrascrive sempre
    parameter         W_FILE     = "l0_s4d_wout.mem",
    parameter         B_FILE     = "l0_s4d_bout.mem",
    parameter         SIG_FILE   = "sigmoid_lut.mem"
)(
    input  wire                        clk,
    input  wire                        rst_n,

    // ---- caricamento pesi: banco ld_sel_w della riga ld_addr_w (= ingresso h) ----
    input  wire                        ld_we_w,
    input  wire [LOG2_NPH+LOG2_H-1:0]  ld_addr_w,     // con LOG2_NPH=0: solo h (0..H-1)
    input  wire [LOG2_NBANK-1:0]       ld_sel_w,      // 0..NBANK-1 (era 4 bit)
    input  wire [LD_W-1:0]             ld_data_w,
    // ---- caricamento bias dell'uscita ld_addr_b (0..2H-1) ----
    input  wire                        ld_we_b,
    input  wire [LOG2_H:0]             ld_addr_b,
    input  wire signed [W_ACC-1:0]     ld_data_b,

    // ---- ingresso: frame di H campioni, in_last sull'ultimo ----
    input  wire                        in_valid,
    output wire                        in_ready,
    input  wire signed [W_IN-1:0]      in_data,
    input  wire                        in_last,

    // ---- uscita: H campioni per frame, SENZA ready (pipeline libera) ----
    // tail_pop: impulso, un risultato ha lasciato la coda d'uscita del
    // layer (restituisce un credito di emissione)
    input  wire                        tail_pop,
    output reg                         out_valid,
    output reg  signed [W_OUT-1:0]     out_data,
    output reg                         out_last
);

    // ------------------------------------------------------------------
    // BACKPRESSURE A CREDITI
    // Il ramo GLU (Q1, Q2, sigmoide, G1..G3, uscita) non si ferma mai:
    // ogni campione emesso dalla testa dello snapshot esce dopo una
    // latenza fissa e finisce, attraverso il residual, nella coda
    // d'uscita del layer (TAIL_DEPTH posti). Si emette solo se c'e' un
    // credito (issue_ok). Se l'uscita del layer stalla, la coda si
    // riempie, i crediti finiscono e lo snapshot si ferma; il motore MAC
    // allora resta in MAC_DRAIN, nfull non decrementa e in_ready va basso
    // dopo due frame (stessa catena di stallo della versione con oen).
    // ------------------------------------------------------------------

    localparam integer TH    = 2*H;                // uscite del lineare (256)
    localparam integer NBANK = (NMAC*W_W) / LD_W;  // banchi da 64 bit per riga (48)
    localparam integer FACC  = F_W + F_IN;         // bit frazionari degli accumulatori
    localparam integer RQ_G  = FACC - F_SIGI;      // shift acc -> gate g (Q.F_SIGI)
    localparam integer RQ_A  = FACC - F_A;         // shift acc -> valore a (Q.F_A)
    localparam integer SH_O  = F_A + F_SIGO - F_OUT;  // shift di a*sigmoid(g) -> Q.F_OUT

    // ---- controlli sui parametri (questa versione e' SOLO a fase singola) ----
    initial begin
        if (NMAC != TH) begin
            $display("s4d_mixing_glu: richiede NMAC = 2H (%0d), trovato %0d", TH, NMAC);
            $finish;
        end
        if (LOG2_NPH != 0) begin
            $display("s4d_mixing_glu: con una sola fase LOG2_NPH deve essere 0");
            $finish;
        end
        if (NBANK * LD_W != NMAC * W_W) begin
            $display("s4d_mixing_glu: NMAC*W_W=%0d non multiplo di LD_W=%0d", NMAC*W_W, LD_W);
            $finish;
        end
        if ((1 << LOG2_NBANK) < NBANK) begin
            $display("s4d_mixing_glu: LOG2_NBANK=%0d troppo stretto per NBANK=%0d", LOG2_NBANK, NBANK);
            $finish;
        end
        if (TAIL_DEPTH < 4) begin
            $display("s4d_mixing_glu: TAIL_DEPTH=%0d troppo piccolo (minimo 4)", TAIL_DEPTH);
            $finish;
        end
        if (SNAP_GROUP < 1 || (H % SNAP_GROUP) != 0) begin
            $display("s4d_mixing_glu: SNAP_GROUP=%0d deve dividere H=%0d", SNAP_GROUP, H);
            $finish;
        end
    end

    //-----------------------------------------------------------------
    // Bias: uno per MAC, in REGISTRI. Il MAC c legge sempre bias_reg[c]
    // (indice costante) -> nessun multiplexer in lettura.
    //-----------------------------------------------------------------
    (* ram_style = "registers" *)
    reg signed [W_ACC-1:0] bias_reg [0:TH-1];

    generate
    if (LOADABLE == 0) begin : g_bias_init
        initial $readmemh(B_FILE, bias_reg);
    end else begin : g_bias_load
        always @(posedge clk)
            if (ld_we_b) bias_reg[ld_addr_b] <= ld_data_b;
    end
    endgenerate

    //-----------------------------------------------------------------
    // Pesi: riga h = i 256 pesi W[c][h], c = 0 negli LSB (3072 bit),
    // divisa in NBANK = 48 banchi da 64 bit, ciascuno profondo H = 128.
    //
    // TIMING (150 -> 200+ MHz): prima l'indirizzo di lettura mac_h andava
    // direttamente a tutti i 48 banchi (fan-out ~6150, slack misurato 0.62 ns)
    // e i dati/indirizzi di scrittura ld_* a tutti i banchi. Ora:
    //   - ogni banco ha la SUA copia registrata dell'indirizzo e dell'enable
    //     di lettura (bank_rd_*_q) e di scrittura (bank_we_q, bank_wr_addr_q);
    //   - i dati di scrittura sono copiati una volta ogni WR_GROUP banchi.
    // Le copie hanno dont_touch, altrimenti la sintesi le riunirebbe in un
    // solo registro (rimozione dei registri equivalenti).
    // 2026-10-05: distribuzione ad ALBERO a 2 livelli (prima: 1 livello).
    // Misurato a 266.7 MHz: mac_h -> 48 copie bank_rd_addr_q slack +0.005 ns,
    // ld_wo_addr -> bank_wr_addr_q +0.008 ns (3.5 ns di solo filo).
    //   livello 1: NWR_GROUPS = 8 copie (una ogni WR_GROUP = 6 banchi)
    //   livello 2: una copia per banco
    // Lettura: indirizzo al gruppo al ciclo t, al banco al ciclo t+1, riga
    // letta al ciclo t+2 e registrata di nuovo al ciclo t+3 (wbank_q),
    // cioe' valida 4 cicli dopo w_rd_en. Il secondo registro e' l'OREG
    // della URAM (o il registro d'uscita della BRAM): stessa latenza per
    // tutti gli stili, cosi' la pipeline MAC non dipende da WMEM_STYLE.
    // Scrittura: 2 cicli di ritardo rispetto alle porte ld_* (ininfluente:
    // i pesi si usano solo centinaia di cicli dopo il caricamento).
    // Versione precedente: mixing_glu.v.bak3 (sul Desktop).
    //-----------------------------------------------------------------
    wire                  w_rd_en;      // il motore MAC legge una riga in questo ciclo
    wire [LOG2_H-1:0]     w_rd_addr;    // riga = ingresso h
    wire [NMAC*W_W-1:0]   w_row;        // riga letta, valida 4 cicli dopo w_rd_en

    // URAM solo se i pesi si scrivono dalle porte: le URAM non si
    // inizializzano dal bitstream, quindi con LOADABLE=0 (pesi da W_FILE)
    // lo stile "ultra" usa la BRAM inizializzata da file.
    localparam integer USE_URAM = (WMEM_STYLE == "ultra") && (LOADABLE != 0);
    localparam integer USE_BRAM = (WMEM_STYLE == "block") ||
                                  ((WMEM_STYLE == "ultra") && (LOADABLE == 0));

    localparam integer WR_GROUP    = 6;                             // banchi serviti da una copia dei dati di scrittura
    localparam integer NWR_GROUPS  = (NBANK + WR_GROUP - 1) / WR_GROUP;   // 8 copie

    // ---- livello 1 dell'albero: una copia per gruppo di WR_GROUP banchi ----
    // lettura: indirizzo ed enable
    (* dont_touch = "true" *) reg              grp_rd_en_q   [0:NWR_GROUPS-1];
    (* dont_touch = "true" *) reg [LOG2_H-1:0] grp_rd_addr_q [0:NWR_GROUPS-1];
    // scrittura: enable, banco selezionato, indirizzo
    (* dont_touch = "true" *) reg                  grp_we_q      [0:NWR_GROUPS-1];
    (* dont_touch = "true" *) reg [LOG2_NBANK-1:0] grp_sel_q     [0:NWR_GROUPS-1];
    (* dont_touch = "true" *) reg [LOG2_H-1:0]     grp_wr_addr_q [0:NWR_GROUPS-1];

    // dati di scrittura: un registro unico (per allinearli agli indirizzi,
    // che attraversano 2 livelli) e poi una copia per gruppo
    reg [LD_W-1:0] ld_data_pre_q;
    (* dont_touch = "true" *)
    reg [LD_W-1:0] ld_data_group_q [0:NWR_GROUPS-1];

    integer wr_group_idx;
    always @(posedge clk) begin
        ld_data_pre_q <= ld_data_w;
        for (wr_group_idx = 0; wr_group_idx < NWR_GROUPS; wr_group_idx = wr_group_idx + 1) begin
            grp_rd_en_q[wr_group_idx]     <= w_rd_en;
            grp_rd_addr_q[wr_group_idx]   <= w_rd_addr;
            grp_we_q[wr_group_idx]        <= ld_we_w;
            grp_sel_q[wr_group_idx]       <= ld_sel_w;
            grp_wr_addr_q[wr_group_idx]   <= ld_addr_w[LOG2_H-1:0];
            ld_data_group_q[wr_group_idx] <= ld_data_pre_q;
        end
    end

    genvar bank_idx;
    generate
    for (bank_idx = 0; bank_idx < NBANK; bank_idx = bank_idx + 1) begin : g_wbank

        wire [LD_W-1:0] wbank_q;        // uscita del banco (2 registri dopo l'indirizzo)

        // ---- copie locali (una per banco) di indirizzi ed enable ----
        (* dont_touch = "true" *) reg              bank_rd_en_q;     // lettura: enable
        (* dont_touch = "true" *) reg [LOG2_H-1:0] bank_rd_addr_q;   // lettura: riga h
        (* dont_touch = "true" *) reg              bank_we_q;        // scrittura: questo banco
        (* dont_touch = "true" *) reg [LOG2_H-1:0] bank_wr_addr_q;   // scrittura: riga

        localparam integer MY_WR_GROUP = bank_idx / WR_GROUP;   // gruppo dell'albero di questo banco

        // livello 2 dell'albero: copia del gruppo -> copia del banco
        always @(posedge clk) begin
            bank_rd_en_q   <= grp_rd_en_q[MY_WR_GROUP];
            bank_rd_addr_q <= grp_rd_addr_q[MY_WR_GROUP];
            bank_we_q      <= grp_we_q[MY_WR_GROUP] && (grp_sel_q[MY_WR_GROUP] == bank_idx);
            bank_wr_addr_q <= grp_wr_addr_q[MY_WR_GROUP];
        end

        // dati di scrittura della copia del gruppo di questo banco
        wire [LD_W-1:0] bank_wr_data = ld_data_group_q[MY_WR_GROUP];

        if (USE_URAM) begin : g_uram
`ifdef S4D_BEHAV_MEM
            // modello comportamentale (simulazione senza librerie Xilinx):
            // stesso comportamento della xpm qui sotto, latenza 2
            reg [LD_W-1:0] wmem [0:H-1];
            reg [LD_W-1:0] uram_rd_q, uram_oreg_q;
            always @(posedge clk)
                if (bank_we_q) wmem[bank_wr_addr_q] <= bank_wr_data;
            always @(posedge clk) begin
                if (bank_rd_en_q) uram_rd_q <= wmem[bank_rd_addr_q];   // lettura (latch della URAM)
                uram_oreg_q <= uram_rd_q;                              // OREG
            end
            assign wbank_q = uram_oreg_q;
`else
            // URAM istanziata esplicitamente: con ram_style = "ultra"
            // Vivado rifiutava i banchi da 128 righe (Synth 8-6849) e
            // ripiegava su LUTRAM. Una URAM288 (4096 x 72) per banco:
            // 48 URAM su 96 dello xczu7ev.
            xpm_memory_sdpram #(
                .ADDR_WIDTH_A       (LOG2_H),
                .ADDR_WIDTH_B       (LOG2_H),
                .AUTO_SLEEP_TIME    (0),
                .BYTE_WRITE_WIDTH_A (LD_W),           // scrittura a parola intera
                .CLOCKING_MODE      ("common_clock"),
                .ECC_MODE           ("no_ecc"),
                .MEMORY_INIT_FILE   ("none"),
                .MEMORY_INIT_PARAM  ("0"),
                .MEMORY_OPTIMIZATION("true"),
                .MEMORY_PRIMITIVE   ("ultra"),
                .MEMORY_SIZE        (H*LD_W),         // bit totali del banco
                .MESSAGE_CONTROL    (0),
                .READ_DATA_WIDTH_B  (LD_W),
                .READ_LATENCY_B     (2),              // latch + OREG della URAM
                .READ_RESET_VALUE_B ("0"),
                .RST_MODE_A         ("SYNC"),
                .RST_MODE_B         ("SYNC"),
                .SIM_ASSERT_CHK     (0),
                .USE_EMBEDDED_CONSTRAINT(0),
                .USE_MEM_INIT       (0),
                .WAKEUP_TIME        ("disable_sleep"),
                .WRITE_DATA_WIDTH_A (LD_W),
                .WRITE_MODE_B       ("read_first")     // l'unico ammesso con le URAM
            ) u_wbank_uram (
                .clka           (clk),
                .ena            (1'b1),
                .wea            (bank_we_q),
                .addra          (bank_wr_addr_q),
                .dina           (bank_wr_data),
                .injectsbiterra (1'b0),
                .injectdbiterra (1'b0),
                .clkb           (clk),
                .rstb           (1'b0),
                .enb            (bank_rd_en_q),
                .regceb         (1'b1),
                .addrb          (bank_rd_addr_q),
                .doutb          (wbank_q),
                .sbiterrb       (),
                .dbiterrb       (),
                .sleep          (1'b0)
            );
`endif
        end else if (USE_BRAM) begin : g_bram
            (* ram_style = "block" *)       reg [LD_W-1:0] wmem [0:H-1];
            reg [LD_W-1:0] bram_rd_q, bram_oreg_q;   // lettura + registro d'uscita della BRAM
            if (LOADABLE != 0) begin : g_wr
                always @(posedge clk)
                    if (bank_we_q) wmem[bank_wr_addr_q] <= bank_wr_data;
            end else begin : g_init
                // pesi da file: ogni riga del file ha NMAC*W_W bit (formato a
                // riga unica), il banco bank_idx ne prende la fetta da 64 bit
                reg [NMAC*W_W-1:0] w_file [0:H-1];
                integer row_idx;
                initial begin
                    $readmemh(W_FILE, w_file);
                    for (row_idx = 0; row_idx < H; row_idx = row_idx + 1)
                        wmem[row_idx] = w_file[row_idx][bank_idx*LD_W +: LD_W];
                end
            end
            always @(posedge clk) begin
                if (bank_rd_en_q) bram_rd_q <= wmem[bank_rd_addr_q];
                bram_oreg_q <= bram_rd_q;                 // assorbito nel DOB_REG della BRAM
            end
            assign wbank_q = bram_oreg_q;
        end else begin : g_lutram
            (* ram_style = "distributed" *) reg [LD_W-1:0] wmem [0:H-1];
            reg [LD_W-1:0] lut_rd_q, lut_oreg_q;      // lettura + secondo registro
            if (LOADABLE != 0) begin : g_wr
                always @(posedge clk)
                    if (bank_we_q) wmem[bank_wr_addr_q] <= bank_wr_data;
            end
            always @(posedge clk) begin
                if (bank_rd_en_q) lut_rd_q <= wmem[bank_rd_addr_q];
                lut_oreg_q <= lut_rd_q;
            end
            assign wbank_q = lut_oreg_q;
        end

        assign w_row[bank_idx*LD_W +: LD_W] = wbank_q;
    end
    endgenerate

    //-----------------------------------------------------------------
    // Doppio buffer d'ingresso: g_buf[{banco, h}] = x[h] (uscita GELU)
    //-----------------------------------------------------------------
    reg signed [W_IN-1:0] g_buf [0:2*H-1];
    reg                   wbank, rbank;   // banco in scrittura / in lettura
    reg [LOG2_H-1:0]      icnt;           // campione in scrittura
    reg [1:0]             nfull;          // frame completi non ancora letti dai MAC (0..2)

    assign in_ready = (nfull < 2'd2);     // da registro: nessun percorso combinatorio

    wire cap_fire = in_valid && in_ready;          // campione acquisito
    wire cap_done = cap_fire && (icnt == H-1);     // frame completo

    always @(posedge clk) if (cap_fire) g_buf[{wbank, icnt}] <= in_data;

    //=================================================================
    // MOTORE MAC: una passata di H cicli per frame, tutti i 256 MAC
    // in parallelo, ognuno sulla propria uscita c.
    //
    //   ciclo t   : MAC_RUN, mac_h = h -> indirizzi copiati nei gruppi di
    //               banchi, x[h] letto da g_buf (x_a)
    //   ciclo t+1 : stadio A2 -> indirizzi copiati nei banchi, x_a2 = x_a
    //   ciclo t+2 : stadio A3 -> riga di pesi letta (latch della memoria),
    //               x_a3 = x_a2
    //   ciclo t+3 : stadio B -> riga di pesi nel registro d'uscita della
    //               memoria (wbank_q), x_a3 copiato in NMAC_GROUPS registri
    //               (x_b_copy), uno per gruppo di MAC
    //   ciclo t+4 : stadio B2 -> registri d'ingresso dei MAC
    //   ciclo t+5 : stadio C -> prodotto registrato (prod_c)
    //   ciclo t+6 : stadio D -> acc[c] aggiornato
    //   (stadio A2 aggiunto il 2026-10-05 insieme all'albero degli indirizzi
    //   dei banchi; stadio A3 il 2026-10-15 per l'OREG della URAM: pesi e x
    //   restano allineati)
    //
    // TIMING: prima x[h] usciva dalla BRAM di g_buf e andava direttamente a
    // tutti i 256 DSP (fan-out 3840, slack misurato 0.138 ns a 150 MHz, 72%
    // instradamento). Ora ogni copia x_b_copy[g] serve solo MAC_GROUP = 16
    // MAC; anche i flag valid/primo-ingresso sono copiati per gruppo.
    // Costo: 1 ciclo di latenza in piu' per frame, throughput invariato.
    //=================================================================
    localparam [1:0] MAC_IDLE  = 2'd0,    // nessun frame da elaborare
                     MAC_RUN   = 2'd1,    // H cicli: un ingresso h per ciclo
                     MAC_DRAIN = 2'd2;    // svuota la pipeline e consegna acc allo snapshot

    localparam integer MAC_GROUP   = 16;                 // MAC serviti da una copia di x
    localparam integer NMAC_GROUPS = NMAC / MAC_GROUP;   // 16 copie

    reg [1:0]        mac_state;
    reg [LOG2_H-1:0] mac_h;               // ingresso h corrente (0..H-1)

    wire mac_last_h = (mac_state == MAC_RUN) && (mac_h == H-1);   // ultimo ingresso del frame

    // flag di controllo NON replicati (servono solo a acc_complete)
    reg mac_v_a, mac_first_a;             // stadio A: x_a e indirizzi dei gruppi validi
    reg mac_v_a2, mac_first_a2;           // stadio A2: x_a2 e indirizzi dei banchi validi
    reg mac_v_a3, mac_first_a3;           // stadio A3: x_a3 valido, riga di pesi nel latch della memoria
    reg mac_v_b;                        // stadio B: x_b_copy e w_row validi
    reg mac_v_b2;                         // stadio B2: registri d'ingresso dei MAC validi
    reg mac_v_c;                          // stadio C: prodotti validi (accumulo in corso)

    // dato e flag replicati per gruppo di MAC (dont_touch: non riunire le copie)
    reg signed [W_IN-1:0] x_a;                                        // x[h] da g_buf
    reg signed [W_IN-1:0] x_a2;                                       // x[h] allineato all'albero degli indirizzi
    reg signed [W_IN-1:0] x_a3;                                       // x[h] allineato alla lettura della memoria
    (* dont_touch = "true" *) reg signed [W_IN-1:0] x_b_copy        [0:NMAC_GROUPS-1];
    (* dont_touch = "true" *) reg                   mac_v_b_copy    [0:NMAC_GROUPS-1];
    (* dont_touch = "true" *) reg                   mac_first_b_copy[0:NMAC_GROUPS-1];
    (* dont_touch = "true" *) reg                   mac_v_b2_copy   [0:NMAC_GROUPS-1];
    (* dont_touch = "true" *) reg                   mac_first_b2_copy[0:NMAC_GROUPS-1];
    (* dont_touch = "true" *) reg                   mac_v_c_copy    [0:NMAC_GROUPS-1];
    (* dont_touch = "true" *) reg                   mac_first_c_copy[0:NMAC_GROUPS-1];

    // acc completi: siamo in DRAIN e nessun aggiornamento e' piu' in volo
    wire acc_complete = (mac_state == MAC_DRAIN) && !mac_v_a && !mac_v_a2 && !mac_v_a3
                      && !mac_v_b && !mac_v_b2 && !mac_v_c;

    // lo snapshot puo' ricevere un nuovo frame (assegnato piu' sotto)
    wire snap_free;
    // consegna: acc -> snapshot, in questo ciclo
    wire snap_take = acc_complete && snap_free;

    // lettura della riga di pesi: un ciclo per ingresso, solo in MAC_RUN
    assign w_rd_en   = (mac_state == MAC_RUN);
    assign w_rd_addr = mac_h;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wbank     <= 1'b0;
            rbank     <= 1'b0;
            icnt      <= {LOG2_H{1'b0}};
            nfull     <= 2'd0;
            mac_state <= MAC_IDLE;
            mac_h     <= {LOG2_H{1'b0}};
        end else begin
            // ---- acquisizione dell'ingresso ----
            if (cap_fire) begin
                if (icnt == H-1) begin
                    icnt  <= {LOG2_H{1'b0}};
                    wbank <= ~wbank;               // frame completo: cambia banco
                end else begin
                    icnt <= icnt + 1'b1;
                end
            end

            // ---- macchina a stati del motore MAC ----
            case (mac_state)
                MAC_IDLE: if (nfull != 2'd0) begin
                    mac_state <= MAC_RUN;
                    mac_h     <= {LOG2_H{1'b0}};
                end

                MAC_RUN: begin
                    if (mac_h == H-1) begin
                        mac_state <= MAC_DRAIN;
                        rbank     <= ~rbank;       // prossimo frame: l'altro banco
                    end else begin
                        mac_h <= mac_h + 1'b1;
                    end
                end

                MAC_DRAIN: if (snap_take) begin
                    // acc consegnati: se c'e' gia' un altro frame si riparte
                    // subito (il suo primo aggiornamento di acc arriva 7 cicli
                    // dopo, quindi non tocca i valori appena copiati)
                    mac_h     <= {LOG2_H{1'b0}};
                    mac_state <= (nfull != 2'd0) ? MAC_RUN : MAC_IDLE;
                end

                default: mac_state <= MAC_IDLE;
            endcase

            // frame pronti: +1 a fine acquisizione, -1 quando i MAC hanno
            // letto l'ultimo ingresso (il banco d'ingresso torna libero)
            case ({cap_done, mac_last_h})
                2'b10: nfull <= nfull + 2'd1;
                2'b01: nfull <= nfull - 2'd1;
                default: ;
            endcase
        end
    end

    // ---- flag di controllo (non replicati) ----
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mac_v_a      <= 1'b0;
            mac_first_a  <= 1'b0;
            mac_v_a2     <= 1'b0;
            mac_first_a2 <= 1'b0;
            mac_v_a3     <= 1'b0;
            mac_first_a3 <= 1'b0;
            mac_v_b      <= 1'b0;
            mac_v_b2     <= 1'b0;
            mac_v_c      <= 1'b0;
        end else begin
            mac_v_a      <= (mac_state == MAC_RUN);
            mac_first_a  <= (mac_state == MAC_RUN) && (mac_h == 0);
            mac_v_a2     <= mac_v_a;
            mac_first_a2 <= mac_first_a;
            mac_v_a3     <= mac_v_a2;
            mac_first_a3 <= mac_first_a2;
            mac_v_b      <= mac_v_a3;
            mac_v_b2     <= mac_v_b;
            mac_v_c      <= mac_v_b2;
        end
    end

    // ---- flag replicati per gruppo di MAC (pilotano solo i DSP del gruppo) ----
    integer group_idx;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (group_idx = 0; group_idx < NMAC_GROUPS; group_idx = group_idx + 1) begin
                mac_v_b_copy[group_idx]     <= 1'b0;
                mac_first_b_copy[group_idx] <= 1'b0;
                mac_v_b2_copy[group_idx]    <= 1'b0;
                mac_first_b2_copy[group_idx] <= 1'b0;
                mac_v_c_copy[group_idx]     <= 1'b0;
                mac_first_c_copy[group_idx] <= 1'b0;
            end
        end else begin
            for (group_idx = 0; group_idx < NMAC_GROUPS; group_idx = group_idx + 1) begin
                mac_v_b_copy[group_idx]     <= mac_v_a3;
                mac_first_b_copy[group_idx] <= mac_first_a3;
                mac_v_b2_copy[group_idx]     <= mac_v_b_copy[group_idx];
                mac_first_b2_copy[group_idx] <= mac_first_b_copy[group_idx];
                mac_v_c_copy[group_idx]     <= mac_v_b2_copy[group_idx];
                mac_first_c_copy[group_idx] <= mac_first_b2_copy[group_idx];
            end
        end
    end

    // ---- dati: x[h] da g_buf (stadio A), ritardi (A2, A3), una copia per gruppo (B) ----
    always @(posedge clk) begin                            // dati: niente reset
        x_a  <= g_buf[{rbank, mac_h}];
        x_a2 <= x_a;
        x_a3 <= x_a2;
    end

    always @(posedge clk)
        for (group_idx = 0; group_idx < NMAC_GROUPS; group_idx = group_idx + 1)
            x_b_copy[group_idx] <= x_a3;

    //-----------------------------------------------------------------
    // 256 MAC: prodotto registrato (stadio C) e accumulo (stadio D).
    // Struttura pensata per il DSP48E2: A=x, B=peso, registro M = prodotto,
    // registro P = accumulatore; il primo ingresso somma il bias (porta C)
    // invece di P. Il MAC c usa la copia del suo gruppo: c / MAC_GROUP.
    //-----------------------------------------------------------------
    wire [TH*W_ACC-1:0] acc_flat;         // tutti gli accumulatori, uscita c in [c*W_ACC +: W_ACC]

    genvar mac_idx;
    generate
    for (mac_idx = 0; mac_idx < NMAC; mac_idx = mac_idx + 1) begin : g_mac

        localparam integer MY_GROUP = mac_idx / MAC_GROUP;                          // gruppo di questo MAC

        wire signed [W_W-1:0]      weight_c = $signed(w_row[mac_idx*W_W +: W_W]);   // W[c][h]
        reg  signed [W_W-1:0]      weight_in;                                       // registro B del DSP
        reg  signed [W_IN-1:0]     x_in;                                            // registro A del DSP
        reg  signed [W_W+W_IN-1:0] prod_c;                                          // W[c][h]*x[h]
        reg  signed [W_ACC-1:0]    acc_c;                                           // accumulatore dell'uscita c

        // stadio B2: registri d'ingresso del moltiplicatore
        always @(posedge clk)
            if (mac_v_b_copy[MY_GROUP]) begin
                weight_in <= weight_c;
                x_in      <= x_b_copy[MY_GROUP];
            end

        // stadio C: prodotto registrato
        always @(posedge clk)
            if (mac_v_b2_copy[MY_GROUP]) prod_c <= weight_in * x_in;

        // stadio D: accumulo; primo ingresso del frame -> riparte dal bias
        always @(posedge clk)
            if (mac_v_c_copy[MY_GROUP])
                acc_c <= (mac_first_c_copy[MY_GROUP] ? bias_reg[mac_idx] : acc_c)
                       + {{(W_ACC-W_W-W_IN){prod_c[W_W+W_IN-1]}}, prod_c};

        assign acc_flat[mac_idx*W_ACC +: W_ACC] = acc_c;
    end
    endgenerate

    //=================================================================
    // SNAPSHOT: copia degli accumulatori di un frame completo, letta
    // dalla GLU in testa (indice 0) e fatta scorrere di 1 a ogni uscita.
    // Cosi' la GLU non usa multiplexer 128:1 e il motore MAC puo'
    // gia' calcolare il frame successivo.
    //   a_snap[k] = acc[k]      (valore a, prima meta')
    //   g_snap[k] = acc[H+k]    (gate g, seconda meta')
    //
    // Emissione a crediti: la testa entra nella GLU (glu_fire) solo se
    // issue_ok, cioe' se la coda d'uscita del layer ha posto garantito.
    //=================================================================
    // contenuto dello snapshot, posizione k in [k*W_ACC +: W_ACC]; i
    // registri veri sono dentro i gruppi (g_snap[*].a_q / g_q)
    wire [H*W_ACC-1:0] a_snap_flat;
    wire [H*W_ACC-1:0] g_snap_flat;

    // ---- stato "principale" del controllo (usato da MAC, Q1 e crediti) ----
    reg              glu_active;          // 1 = lo snapshot contiene uscite non ancora emesse
    reg [LOG2_H-1:0] glu_cnt;             // indice dell'uscita in testa (0..H-1)
    reg              glu_last;            // registro: glu_cnt == H-1 (la testa e' l'ultima del frame)
    reg              issue_ok;            // registro: c'e' un credito di emissione

    // ---- crediti: risultati emessi e non ancora usciti dalla coda del layer ----
    localparam integer CRED_W = $clog2(TAIL_DEPTH + 1);
    reg [CRED_W-1:0] inflight;

    // la testa entra nella GLU in questo ciclo
    wire glu_fire = glu_active && issue_ok;

    // lo snapshot e' libero se vuoto, o se sta consegnando ORA l'ultima uscita
    assign snap_free = !glu_active || (glu_fire && glu_last);

    // ---- stato del ciclo successivo: lo stesso valore va al registro
    //      principale e a tutte le sue copie per gruppo ----
    wire glu_active_next = snap_take              ? 1'b1        // nuovo frame
                         : (glu_fire && glu_last) ? 1'b0        // emessa l'ultima uscita
                         :                          glu_active;
    wire glu_last_next   = (H == 1)   ? 1'b1
                         : snap_take  ? 1'b0                    // si riparte da glu_cnt = 0
                         : glu_fire   ? (glu_cnt == H-2)        // la prossima testa e' l'ultima
                         :              glu_last;
    // credito con un ciclo di anticipo sul conteggio: issue_ok al ciclo t
    // vale inflight(t-1) < TAIL_DEPTH-2, quindi inflight(t) <= TAIL_DEPTH-2
    // e, dopo un'eventuale emissione, inflight <= TAIL_DEPTH-1: la coda
    // del layer (TAIL_DEPTH posti) non puo' traboccare
    wire issue_ok_next   = (inflight < TAIL_DEPTH - 2);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            glu_active <= 1'b0;
            glu_cnt    <= {LOG2_H{1'b0}};
            glu_last   <= (H == 1);
            issue_ok   <= 1'b0;
            inflight   <= {CRED_W{1'b0}};
        end else begin
            glu_active <= glu_active_next;
            glu_last   <= glu_last_next;
            issue_ok   <= issue_ok_next;
            if (snap_take)
                glu_cnt <= {LOG2_H{1'b0}};                       // nuovo frame
            else if (glu_fire)
                glu_cnt <= glu_last ? {LOG2_H{1'b0}} : (glu_cnt + 1'b1);
            // +1 per ogni emissione, -1 per ogni risultato uscito dal layer
            inflight <= inflight + (glu_fire ? 1'b1 : 1'b0) - (tail_pop ? 1'b1 : 1'b0);
        end
    end

    //-----------------------------------------------------------------
    // Dati dello snapshot, a gruppi di SNAP_GROUP posizioni.
    // Ogni gruppo ha le PROPRIE copie registrate di glu_active, glu_last e
    // issue_ok (stesso ingresso D dei registri principali, quindi sempre
    // uguali a loro) e ricava localmente carico e scorrimento: ogni
    // enable pilota 2*SNAP_GROUP*W_ACC flip-flop invece di 2*H*W_ACC.
    // dont_touch: senza, la sintesi riunirebbe le copie equivalenti.
    //-----------------------------------------------------------------
    localparam integer NSNAP_GROUPS = H / SNAP_GROUP;

    genvar snap_grp;
    generate
    for (snap_grp = 0; snap_grp < NSNAP_GROUPS; snap_grp = snap_grp + 1) begin : g_snap

        (* dont_touch = "true" *) reg grp_active_q;   // copia di glu_active
        (* dont_touch = "true" *) reg grp_last_q;     // copia di glu_last
        (* dont_touch = "true" *) reg grp_issue_q;    // copia di issue_ok

        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                grp_active_q <= 1'b0;
                grp_last_q   <= (H == 1);
                grp_issue_q  <= 1'b0;
            end else begin
                grp_active_q <= glu_active_next;
                grp_last_q   <= glu_last_next;
                grp_issue_q  <= issue_ok_next;
            end
        end

        // stesse equazioni di glu_fire / snap_free / snap_take, sulle copie
        wire grp_fire = grp_active_q && grp_issue_q;
        wire grp_take = acc_complete && (!grp_active_q || (grp_fire && grp_last_q));

        // le SNAP_GROUP posizioni di questo gruppo
        genvar pos_in_grp;
        for (pos_in_grp = 0; pos_in_grp < SNAP_GROUP; pos_in_grp = pos_in_grp + 1) begin : g_pos
            localparam integer SNAP_IDX = snap_grp*SNAP_GROUP + pos_in_grp;   // posizione 0..H-1

            reg [W_ACC-1:0] a_q;          // a_snap[SNAP_IDX]
            reg [W_ACC-1:0] g_q;          // g_snap[SNAP_IDX]

            // valore che entra con lo scorrimento: la posizione successiva
            // (l'ultima posizione resta ferma, come nella versione precedente)
            wire [W_ACC-1:0] a_next_pos = (SNAP_IDX < H-1) ? a_snap_flat[((SNAP_IDX+1) % H)*W_ACC +: W_ACC] : a_q;
            wire [W_ACC-1:0] g_next_pos = (SNAP_IDX < H-1) ? g_snap_flat[((SNAP_IDX+1) % H)*W_ACC +: W_ACC] : g_q;

            always @(posedge clk) begin                    // dati: niente reset
                if (grp_take) begin
                    // carico parallelo del frame appena completato
                    a_q <= acc_flat[SNAP_IDX*W_ACC     +: W_ACC];
                    g_q <= acc_flat[(H+SNAP_IDX)*W_ACC +: W_ACC];
                end else if (grp_fire) begin
                    // scorrimento verso la testa
                    a_q <= a_next_pos;
                    g_q <= g_next_pos;
                end
            end

            assign a_snap_flat[SNAP_IDX*W_ACC +: W_ACC] = a_q;
            assign g_snap_flat[SNAP_IDX*W_ACC +: W_ACC] = g_q;
        end
    end
    endgenerate

    // testa dello snapshot (posizione 0), letta da Q1
    wire signed [W_ACC-1:0] a_head = $signed(a_snap_flat[0 +: W_ACC]);
    wire signed [W_ACC-1:0] g_head = $signed(g_snap_flat[0 +: W_ACC]);

    //-----------------------------------------------------------------
    // Riquantizzazione della testa dello snapshot (2 stadi, scorrimento libero):
    //   Q1  somma del mezzo LSB (se lo shift e' a destra)
    //   Q2  shift fisso (fili) e saturazione a 16 bit con segno
    // Stessa aritmetica della vecchia funzione rq: somma su W_ACC bit,
    // poi (v + 2^(s-1)) >>> s per s > 0, oppure v <<< -s per s <= 0.
    //-----------------------------------------------------------------
    localparam signed [W_ACC-1:0] RND_G = (RQ_G > 0) ? (1 <<< (RQ_G - 1)) : 0;
    localparam signed [W_ACC-1:0] RND_A = (RQ_A > 0) ? (1 <<< (RQ_A - 1)) : 0;
    localparam signed [W_ACC-1:0] SIG_HI = (1 <<< (W_SIG-1)) - 1;     // limiti a W_SIG bit
    localparam signed [W_ACC-1:0] SIG_LO = -(1 <<< (W_SIG-1));
    localparam signed [W_ACC-1:0] A_HI   = (1 <<< (W_A-1)) - 1;       // limiti a W_A bit
    localparam signed [W_ACC-1:0] A_LO   = -(1 <<< (W_A-1));

    // shift a destra (s > 0) o a sinistra (s <= 0) di una quantita' fissa
    function signed [W_ACC-1:0] fixed_shift;
        input signed [W_ACC-1:0] v;
        input integer            s;
        begin
            fixed_shift = (s > 0) ? (v >>> s) : (v <<< (-s));
        end
    endfunction

    // ---- Q1: somma del mezzo LSB ----
    reg                    q1_valid, q1_last;
    reg signed [W_ACC-1:0] q1_g, q1_a;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            q1_valid <= 1'b0;
            q1_last  <= 1'b0;
        end else begin
            q1_valid <= glu_fire;                // la testa e' stata emessa
            q1_last  <= glu_fire && glu_last;
        end
    end

    always @(posedge clk) begin
        q1_g <= g_head + RND_G;
        q1_a <= a_head + RND_A;
    end

    // ---- Q2: shift fisso e saturazione ----
    wire signed [W_ACC-1:0] q1_g_sh = fixed_shift(q1_g, RQ_G);
    wire signed [W_ACC-1:0] q1_a_sh = fixed_shift(q1_a, RQ_A);

    reg                    q2_valid, q2_last;
    reg signed [W_SIG-1:0] g_q;                     // gate in Q.F_SIGI
    reg signed [W_A-1:0]   a_q;                     // valore in Q.F_A

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            q2_valid <= 1'b0;
            q2_last  <= 1'b0;
        end else begin
            q2_valid <= q1_valid;
            q2_last  <= q1_last;
        end
    end

    always @(posedge clk) begin
        if      (q1_g_sh > SIG_HI) g_q <= SIG_HI[W_SIG-1:0];
        else if (q1_g_sh < SIG_LO) g_q <= SIG_LO[W_SIG-1:0];
        else                       g_q <= q1_g_sh[W_SIG-1:0];
        if      (q1_a_sh > A_HI)   a_q <= A_HI[W_A-1:0];
        else if (q1_a_sh < A_LO)   a_q <= A_LO[W_A-1:0];
        else                       a_q <= q1_a_sh[W_A-1:0];
    end

    wire                    sig_valid;
    wire                    sig_in_ready;    // sempre 1 (out_ready = 1), non serve consultarlo
    wire signed [W_SIG-1:0] sig_out;         // sigmoid(g) in Q.F_SIGO
    wire signed [W_A-1:0]   sig_a;           // a, arrivato insieme a sigmoid(g)
    wire                    sig_last;        // ultimo campione del frame

    //=================================================================
    // Sigmoide approssimata a tratti lineari (256 segmenti in [-8, 8)),
    // fuori intervallo: 0 a sinistra, 1.0 a destra. Il tag porta a e il
    // flag di fine frame attraverso la sua pipeline (prima: catene a_d[]
    // e last_d[] da tenere allineate a mano alla latenza della PWL).
    //=================================================================
    s4d_pwl #(
        .W_IN(W_SIG), .F_IN(F_SIGI), .W_OUT(W_SIG), .F_OUT(F_SIGO),
        .NSEG(256), .LOG2_NSEG(8), .XH(3),
        .MODE_HI(0), .Y_LO_Q(0), .Y_HI_Q(1 << F_SIGO),
        .TAG_W(W_A + 1),
        .LUT_FILE(SIG_FILE)
    ) u_sig (
        .clk       (clk),
        .rst_n     (rst_n),
        .in_valid  (q2_valid),
        .in_ready  (sig_in_ready),
        .in_data   (g_q),
        .in_tag    ({q2_last, a_q}),
        .out_ready (1'b1),                    // pipeline libera: nessuno stallo
        .out_valid (sig_valid),
        .out_data  (sig_out),
        .out_tag   ({sig_last, sig_a})
    );

    //-----------------------------------------------------------------
    // Prodotto GLU a * sigmoid(g), arrotondamento a Q.F_OUT, saturazione
    // (3 stadi, scorrimento libero):
    //   G1  prodotto a * sigmoid(g)            (registro M del DSP)
    //   G2  + mezzo LSB                        (registro P del DSP)
    //   G3  shift fisso (fili), saturazione -> registro d'uscita
    // Nessun overflow possibile: |a*sig| <= 2^30 su PWO = 32 bit.
    //-----------------------------------------------------------------
    localparam integer PWO = W_A + W_SIG;
    localparam signed [PWO-1:0] OMAX = {{(PWO-W_OUT+1){1'b0}}, {(W_OUT-1){1'b1}}};
    localparam signed [PWO-1:0] OMIN = {{(PWO-W_OUT+1){1'b1}}, {(W_OUT-1){1'b0}}};
    localparam signed [PWO-1:0] PRND = {{(PWO-1){1'b0}}, 1'b1} <<< (SH_O-1);   // mezzo LSB d'uscita

    reg                  g1_valid, g1_last, g2_valid, g2_last;
    reg signed [PWO-1:0] g1_prod, g2_sum;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            g1_valid <= 1'b0; g1_last <= 1'b0;
            g2_valid <= 1'b0; g2_last <= 1'b0;
        end else begin
            g1_valid <= sig_valid;  g1_last <= sig_valid && sig_last;
            g2_valid <= g1_valid;   g2_last <= g1_last;
        end
    end

    always @(posedge clk) begin
        g1_prod <= sig_a * sig_out;                 // Q.F_A+F_SIGO
        g2_sum  <= g1_prod + PRND;
    end

    wire signed [PWO-1:0] pout = g2_sum >>> SH_O;

    // registro d'uscita (scorrimento libero: il layer accoglie sempre il dato)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            out_data  <= {W_OUT{1'b0}};
            out_last  <= 1'b0;
        end else begin
            out_valid <= g2_valid;
            out_last  <= g2_last;

            if      (pout > OMAX) out_data <= OMAX[W_OUT-1:0];
            else if (pout < OMIN) out_data <= OMIN[W_OUT-1:0];
            else                  out_data <= pout[W_OUT-1:0];
        end
    end

    // controllo solo in simulazione: in_last deve cadere sull'H-esimo campione
// synthesis translate_off
    always @(posedge clk)
        if (cap_fire && (in_last != (icnt == H-1))) begin
            $display("s4d_mixing_glu: in_last a icnt=%0d, atteso a %0d -- il", icnt, H-1);
            $display("  frame in ingresso non e' lungo H: sincronizzazione rotta");
            $finish;
        end
// synthesis translate_on

endmodule

`default_nettype wire
