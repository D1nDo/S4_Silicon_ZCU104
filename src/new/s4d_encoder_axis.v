`timescale 1ns / 1ps
`default_nettype none
//=====================================================================
// s4d_encoder_axis -- encoder lineare PARAMETRICO
//   NIN ingressi da N_BIT_IN bit per timestep  ->  H canali del modello S4D
//   Ingresso AXI-Stream a AXI_DW bit, uscita AXI-Stream a W_ACT bit.
//
//   Sostituisce, con lo stesso risultato bit a bit:
//     - s4d_encoder_ecg_axis64  (ECG:              NIN=12, 16 bit con segno,  AXI_DW=64)
//     - s4d_encoder_sc_axis64   (Speech Commands:  NIN=12, 16 bit con segno,  AXI_DW=64)
//     - axis_32to8 + s4d_encoder_axis (sCIFAR:     NIN=1,   8 bit senza segno, AXI_DW=32)
//
//   Target: Zynq UltraScale+ XCZU7EV-2 (ZCU104), clock >= 250 MHz (4 ns).
//
//   Funzione calcolata, per ogni timestep t e ogni canale ch:
//
//     y[t][ch] = sat_W_ACT( round( sum_{c<NIN} W[ch][c]*x[t][c] + b[ch] ) >> OUT_SHIFT )
//
//   ---------------------------------------------------------------
//   INGRESSO (S_AXIS, AXI_DW bit) -- flusso continuo di campioni
//   ---------------------------------------------------------------
//     Ogni parola contiene SAMPLES_PER_WORD = AXI_DW / N_BIT_IN campioni,
//     il primo nei bit meno significativi:
//       s_axis_tdata[N_BIT_IN-1:0]            = campione k
//       s_axis_tdata[2*N_BIT_IN-1:N_BIT_IN]   = campione k+1   ...
//
//     I campioni formano un nastro continuo: un timestep e' fatto da NIN
//     campioni consecutivi del nastro (c0 per primo) e PUO' iniziare a
//     meta' parola. E' esattamente l'ordine dei byte in DDR (little-endian):
//       - ECG/SC : int16 c0..c11 per timestep -> 3 parole da 64 bit per timestep
//       - sCIFAR : 1 pixel uint8 per timestep -> 4 timestep per parola da 32 bit
//
//     tlast va sull'ULTIMA parola del frame. Dentro quella parola l'ultimo
//     timestep e' quello che si completa quando restano meno di NIN
//     campioni: i campioni rimanenti sono riempimento (padding) e vengono
//     scartati. Se la parola con tlast finisce senza completare un timestep,
//     il timestep parziale si scarta e si alza err_frame.
//
//   ---------------------------------------------------------------
//   USCITA (M_AXIS, W_ACT bit)
//   ---------------------------------------------------------------
//     H beat per timestep (canale 0, 1, ..., H-1).
//     tlast sull'ultimo canale dell'ultimo timestep del frame.
//
//   ---------------------------------------------------------------
//   STRUTTURA
//   ---------------------------------------------------------------
//     1. Caricatore a nastro: prende 1 campione per ciclo dalla parola
//        corrente (registro a scorrimento, niente multiplexer) e lo
//        spinge nel buffer di ingresso (anch'esso a scorrimento).
//        Servono NIN cicli per timestep, sovrapposti agli H cicli di
//        calcolo del timestep precedente: non rallenta finche' NIN <= H.
//     2. Doppio buffer: ingresso (timestep successivo) / calcolo
//        (timestep in corso, fermo per tutti gli H canali).
//     3. Pipeline senza stallo (PIPE_STAGES stadi):
//          S1, S2      lettura Block RAM pesi/bias (+ registro d'uscita)
//          S3, S4, S5  NIN prodotti nei DSP48E2 (registri A/B, M, P)
//          S6 ..       albero di somma binario su NIN prodotti + bias:
//                      TREE_LEVELS = clog2(NIN+1) livelli, un sommatore
//                      a 2 ingressi per registro
//          ultimo      arrotondamento + shift verso F_ACT
//        --> saturazione a W_ACT bit scritta nella FIFO di uscita.
//     4. Controllo di flusso a crediti: un canale entra in S1 solo se ha
//        gia' un posto garantito nella FIFO di uscita, quindi la pipeline
//        non si ferma mai e m_axis_tready pilota solo la FIFO.
//
//     Con NIN=12: TREE_LEVELS=4, PIPE_STAGES=10 (come l'encoder ECG).
//     Con NIN=1 : TREE_LEVELS=1, PIPE_STAGES=7.
//     L'aritmetica e' intera e senza overflow (controllato sotto), quindi
//     l'ordine delle somme non cambia il risultato.
//
//   ---------------------------------------------------------------
//   FORMATI
//   ---------------------------------------------------------------
//     IN_SIGNED = 1 : campioni in complemento a due (ECG, SC: Q5.11)
//     IN_SIGNED = 0 : campioni senza segno (sCIFAR: pixel 0..255)
//     F_WENC    = bit frazionari dell'accumulatore (pesi + ingresso)
//     OUT_SHIFT = F_WENC - F_ACT
//     enc_w.mem : H righe da NIN*W_WENC bit, una per canale. Con
//                 C0_AT_LSB=1 il peso di c0 sta nei W_WENC bit bassi.
//     enc_b.mem : H righe da W_BENC bit, gia' nella scala F_WENC.
//     (stessi .mem degli encoder precedenti, nessuna rigenerazione)
//=====================================================================

module s4d_encoder_axis #(
    parameter integer H          = 128,          // canali del modello (uscite per timestep)
    parameter integer NIN        = 12,           // ingressi (derivazioni/campioni) per timestep
    parameter integer N_BIT_IN   = 16,           // bit di un campione in ingresso
    parameter integer IN_SIGNED  = 1,            // 1 = campioni con segno, 0 = senza segno
    parameter integer AXI_DW     = 64,           // larghezza della parola S_AXIS

    parameter integer W_WENC     = 18,           // bit di un peso
    parameter integer F_WENC     = 31,           // bit frazionari dell'accumulatore
    parameter integer W_BENC     = 32,           // bit di un bias
    parameter integer W_ENCACC   = 40,           // bit dell'accumulatore / albero di somma

    parameter integer W_ACT      = 16,           // bit dell'attivazione in uscita
    parameter integer F_ACT      = 11,           // bit frazionari dell'uscita (Q5.11)

    parameter integer C0_AT_LSB  = 1,            // 1 = peso di c0 negli LSB della riga di enc_w.mem

    parameter         ENC_W_FILE = "enc_w_ECG.mem",  // file dei pesi
    parameter         ENC_B_FILE = "enc_b_ECG.mem"   // file dei bias
)(
    input  wire                        aclk,
    input  wire                        aresetn,          // reset attivo basso

    // ---- ingresso campioni (dal DMA) ----
    input  wire                        s_axis_tvalid,
    output wire                        s_axis_tready,
    input  wire [AXI_DW-1:0]           s_axis_tdata,     // SAMPLES_PER_WORD campioni per parola
    input  wire                        s_axis_tlast,     // ultima parola del frame

    // ---- uscita attivazioni verso il core S4D ----
    output wire                        m_axis_tvalid,
    input  wire                        m_axis_tready,
    output wire signed [W_ACT-1:0]     m_axis_tdata,
    output wire                        m_axis_tlast,

    // ---- flag di diagnostica (restano a 1 fino al reset) ----
    output reg                         sat_sticky,       // almeno un'uscita e' stata saturata
    output reg                         err_frame         // frame finito a meta' di un timestep
);

    //=================================================================
    // Funzione costante: nodi presenti al livello 'level' dell'albero
    //   livello 0 = NIN prodotti + 1 bias; ogni livello dimezza
    //   arrotondando per eccesso (il nodo dispari passa avanti).
    //=================================================================
    function integer tree_nodes;
        input integer level;
        integer node_count, step;
        begin
            node_count = NIN + 1;
            for (step = 0; step < level; step = step + 1)
                node_count = (node_count + 1) / 2;
            tree_nodes = node_count;
        end
    endfunction

    //=================================================================
    // Costanti derivate
    //=================================================================
    localparam integer LOG2H            = (H > 1) ? $clog2(H) : 1;          // bit dell'indice di canale
    localparam integer SAMPLES_PER_WORD = AXI_DW / N_BIT_IN;                // campioni in una parola AXI
    localparam integer W_WORD_SMP_IDX   = (SAMPLES_PER_WORD > 1) ? $clog2(SAMPLES_PER_WORD) : 1;
    localparam integer W_STEP_SMP_IDX   = (NIN > 1) ? $clog2(NIN) : 1;     // bit del contatore campioni/timestep
    localparam integer W_STEP           = NIN * N_BIT_IN;                   // bit di un timestep completo

    // Campione esteso per il DSP: con segno resta N_BIT_IN bit, senza
    // segno riceve uno 0 in testa (cosi' il DSP lo vede positivo)
    localparam integer W_X              = IN_SIGNED ? N_BIT_IN : (N_BIT_IN + 1);
    localparam integer W_PROD           = W_WENC + W_X;                     // bit di un prodotto peso*campione
    localparam integer OUT_SHIFT        = F_WENC - F_ACT;                   // shift accumulatore -> F_ACT

    localparam integer TREE_ADDENDS     = NIN + 1;                          // prodotti + bias
    localparam integer TREE_LEVELS      = $clog2(TREE_ADDENDS);             // livelli registrati dell'albero
    localparam integer W_LEVEL_BUS      = TREE_ADDENDS * W_ENCACC;          // bit di un livello (spazio massimo)

    localparam integer DSP_LAST_STAGE   = 5;                                // S5 = registro P dei DSP
    localparam integer PIPE_STAGES      = DSP_LAST_STAGE + TREE_LEVELS + 1; // + albero + arrotondamento
    localparam integer W_FIFO_PTR       = $clog2(PIPE_STAGES + 3);          // FIFO > latenza di un credito
    localparam integer OUT_FIFO_DEPTH   = (1 << W_FIFO_PTR);                // potenza di 2
    localparam integer W_CREDIT         = W_FIFO_PTR + 1;                   // conta fino a OUT_FIFO_DEPTH incluso

    // Bit necessari all'albero: il piu' largo degli addendi + un bit per livello
    localparam integer W_WIDEST_ADDEND  = (W_PROD > W_BENC) ? W_PROD : W_BENC;
    localparam integer W_TREE_NEEDED    = W_WIDEST_ADDEND + TREE_LEVELS;

    //=================================================================
    // Controlli sui parametri: fermano simulazione/elaborazione
    //=================================================================
    initial begin
        if (OUT_SHIFT < 0) begin
            $display("s4d_encoder_axis: F_WENC (%0d) < F_ACT (%0d)", F_WENC, F_ACT);
            $finish;
        end
        if ((AXI_DW % N_BIT_IN) != 0) begin
            // ogni parola deve contenere un numero intero di campioni
            $display("s4d_encoder_axis: AXI_DW (%0d) non multiplo di N_BIT_IN (%0d)", AXI_DW, N_BIT_IN);
            $finish;
        end
        if (W_X > 27) begin
            // porta A del DSP48E2: 27 bit
            $display("s4d_encoder_axis: campione esteso a %0d bit, oltre i 27 della porta A del DSP", W_X);
            $finish;
        end
        if (W_ENCACC < W_TREE_NEEDED) begin
            // l'albero potrebbe traboccare: il risultato non sarebbe piu' esatto
            $display("s4d_encoder_axis: W_ENCACC (%0d) < %0d bit necessari a %0d addendi",
                     W_ENCACC, W_TREE_NEEDED, TREE_ADDENDS);
            $finish;
        end
        if (NIN > H)
            // non e' un errore: il caricatore (NIN cicli/timestep) diventa
            // piu' lento del calcolo (H cicli/timestep)
            $display("s4d_encoder_axis: ATTENZIONE NIN (%0d) > H (%0d), throughput < 1 canale/ciclo", NIN, H);
    end

    //=================================================================
    // Block RAM dei pesi e dei bias (una riga per canale di uscita)
    //   Usate in sola lettura (ROM), inizializzate dai file .mem.
    //=================================================================
    (* rom_style = "block" *)
    reg        [NIN*W_WENC-1:0] enc_weight_rom [0:H-1];   // riga ch = NIN pesi del canale ch
    (* rom_style = "block" *)
    reg signed [W_BENC-1:0]     enc_bias_rom   [0:H-1];   // bias del canale ch (scala F_WENC)

    initial begin
        $readmemh(ENC_W_FILE, enc_weight_rom);
        $readmemh(ENC_B_FILE, enc_bias_rom);
    end

    //=================================================================
    // 1. CARICATORE A NASTRO
    //=================================================================

    //-----------------------------------------------------------------
    // Parola corrente: registro a scorrimento verso destra. Il campione
    // da consumare e' sempre nei N_BIT_IN bit bassi.
    //-----------------------------------------------------------------
    reg [AXI_DW-1:0]         word_shift_reg;              // parola in consumo
    reg                      word_valid;                  // 1 = restano campioni da consumare
    reg                      word_is_frame_last;          // la parola aveva tlast
    reg [W_WORD_SMP_IDX-1:0] word_sample_idx;             // indice del campione corrente nella parola

    wire [N_BIT_IN-1:0]      current_sample = word_shift_reg[N_BIT_IN-1:0];

    //-----------------------------------------------------------------
    // Buffer di ingresso: registro a scorrimento, il campione nuovo entra
    // in cima. Dopo NIN spinte c0 si trova nei bit bassi (slot 0).
    //-----------------------------------------------------------------
    reg [W_STEP-1:0]         in_step_buf;                 // NIN campioni del timestep in arrivo
    reg [W_STEP_SMP_IDX-1:0] in_sample_idx;               // campioni gia' caricati nel timestep
    reg                      in_buf_full;                 // 1 = timestep completo, in attesa del calcolo
    reg                      in_buf_is_last_step;         // 1 = e' l'ultimo timestep del frame

    // Un campione si consuma se c'e' e se il buffer di ingresso ha posto
    wire load_sample        = word_valid & ~in_buf_full;
    // Con questo campione il timestep e' completo
    wire step_complete      = load_sample & (in_sample_idx == NIN-1);
    // Dopo questo campione nella parola ne restano meno di NIN:
    // sono riempimento se la parola e' l'ultima del frame
    wire word_tail_is_pad   = (word_sample_idx + NIN >= SAMPLES_PER_WORD);
    // Fine del frame: ultimo timestep completo dentro la parola con tlast
    wire frame_end_here     = step_complete & word_is_frame_last & word_tail_is_pad;
    // Consumato l'ultimo campione della parola
    wire word_exhausted     = load_sample & (word_sample_idx == SAMPLES_PER_WORD-1);
    // La parola e' finita: esaurita, oppure il resto e' riempimento
    wire word_done          = word_exhausted | frame_end_here;
    // Parola con tlast esaurita a meta' di un timestep: frame troppo corto
    wire partial_step_drop  = word_exhausted & word_is_frame_last & ~step_complete;

    // Si accetta una parola se non ce n'e' una in corso, o se quella in
    // corso finisce in questo ciclo (nessun ciclo perso tra due parole)
    assign s_axis_tready = ~word_valid | word_done;
    wire   in_word_accept = s_axis_tvalid & s_axis_tready;   // handshake AXIS riuscito

    // Stato della parola corrente
    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn) begin
            word_valid         <= 1'b0;
            word_is_frame_last <= 1'b0;
            word_sample_idx    <= {W_WORD_SMP_IDX{1'b0}};
        end else if (in_word_accept) begin
            // nuova parola: si riparte dal campione nei bit bassi
            word_valid         <= 1'b1;
            word_is_frame_last <= s_axis_tlast;
            word_sample_idx    <= {W_WORD_SMP_IDX{1'b0}};
        end else if (word_done) begin
            word_valid         <= 1'b0;                    // nessuna parola nuova pronta
        end else if (load_sample) begin
            word_sample_idx    <= word_sample_idx + 1'b1;  // passa al campione successivo
        end
    end

    // Dati della parola: scorrimento di N_BIT_IN bit a ogni campione consumato
    always @(posedge aclk) begin
        if (in_word_accept)
            word_shift_reg <= s_axis_tdata;
        else if (load_sample)
            word_shift_reg <= word_shift_reg >> N_BIT_IN;
    end

    // Dati del timestep: il campione nuovo entra in cima, gli altri scendono
    generate
        if (NIN == 1) begin : g_step_single
            always @(posedge aclk)
                if (load_sample) in_step_buf <= current_sample;
        end else begin : g_step_shift
            always @(posedge aclk)
                if (load_sample) in_step_buf <= {current_sample, in_step_buf[W_STEP-1:N_BIT_IN]};
        end
    endgenerate

    //=================================================================
    // 2. DOPPIO BUFFER E SCANSIONE DEI CANALI
    //=================================================================
    reg [W_STEP-1:0]     cur_step_buf;             // NIN campioni del timestep in elaborazione
    reg                  cur_buf_busy;             // 1 = restano canali da lanciare
    reg                  cur_buf_is_last_step;     // 1 = e' l'ultimo timestep del frame
    reg [LOG2H-1:0]      out_channel;              // prossimo canale da lanciare = indirizzo BRAM

    wire out_channel_is_last = (out_channel == H-1);

    // Crediti: risultati "in volo" verso l'uscita
    reg  [W_CREDIT-1:0]  in_flight_count;          // in pipeline + in FIFO + registro d'uscita
    wire                 credit_available = (in_flight_count < OUT_FIFO_DEPTH);

    // Un canale viene lanciato in S1 se ce n'e' uno da fare e c'e' posto garantito
    wire launch_channel = cur_buf_busy & credit_available;

    // Passaggio ingresso -> calcolo: ingresso pieno e calcolo libero,
    // oppure sta partendo proprio l'ultimo canale del timestep corrente
    // (S1 legge cur_step_buf in questo stesso fronte: prende ancora il vecchio)
    wire move_in_to_cur = in_buf_full & (~cur_buf_busy | (launch_channel & out_channel_is_last));

    // Handshake di uscita (definito piu' in basso, serve ai crediti)
    wire out_beat_done;

    // Contatore dei campioni e stato del buffer di ingresso
    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn) begin
            in_sample_idx       <= {W_STEP_SMP_IDX{1'b0}};
            in_buf_full         <= 1'b0;
            in_buf_is_last_step <= 1'b0;
            err_frame           <= 1'b0;
        end else begin
            // timestep copiato nel buffer di calcolo -> ingresso di nuovo libero
            // (esclusivo con load_sample, che richiede ~in_buf_full)
            if (move_in_to_cur) in_buf_full <= 1'b0;

            if (load_sample) begin
                if (step_complete) begin
                    // timestep completo: passa in attesa del calcolo
                    in_sample_idx       <= {W_STEP_SMP_IDX{1'b0}};
                    in_buf_full         <= 1'b1;
                    in_buf_is_last_step <= frame_end_here;
                end else if (partial_step_drop) begin
                    // frame finito a meta' timestep: si butta il parziale
                    in_sample_idx       <= {W_STEP_SMP_IDX{1'b0}};
                    err_frame           <= 1'b1;
                end else begin
                    in_sample_idx       <= in_sample_idx + 1'b1;
                end
            end
        end
    end

    // Scansione dei canali 0..H-1 per il timestep nel buffer di calcolo
    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn) begin
            cur_buf_busy <= 1'b0;
            out_channel  <= {LOG2H{1'b0}};
        end else if (move_in_to_cur) begin
            // nuovo timestep: si riparte dal canale 0
            cur_buf_busy <= 1'b1;
            out_channel  <= {LOG2H{1'b0}};
        end else if (launch_channel) begin
            if (out_channel_is_last) cur_buf_busy <= 1'b0;         // lanciati tutti gli H canali
            else                     out_channel  <= out_channel + 1'b1;
        end
    end

    // Copia del timestep dal buffer di ingresso a quello di calcolo
    always @(posedge aclk) begin
        if (move_in_to_cur) begin
            cur_step_buf         <= in_step_buf;
            cur_buf_is_last_step <= in_buf_is_last_step;
        end
    end

    // Crediti: +1 quando si lancia un canale, -1 quando un beat esce su M_AXIS
    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn)
            in_flight_count <= {W_CREDIT{1'b0}};
        else if ( launch_channel & ~out_beat_done)
            in_flight_count <= in_flight_count + 1'b1;
        else if (~launch_channel &  out_beat_done)
            in_flight_count <= in_flight_count - 1'b1;
        // entrambi o nessuno: il conteggio non cambia
    end

    //=================================================================
    // 3. PIPELINE
    //=================================================================

    //-----------------------------------------------------------------
    // Flag valid/last lungo la pipeline (scorrono sempre, niente enable)
    //   stage_valid[k] = lo stadio Sk contiene un dato valido
    //   stage_last[k]  = lo stadio Sk contiene l'ultimo beat del frame
    //-----------------------------------------------------------------
    reg [PIPE_STAGES:1] stage_valid, stage_last;

    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn) begin
            stage_valid <= {PIPE_STAGES{1'b0}};
            stage_last  <= {PIPE_STAGES{1'b0}};
        end else begin
            stage_valid <= {stage_valid[PIPE_STAGES-1:1], launch_channel};
            stage_last  <= {stage_last[PIPE_STAGES-1:1],
                            launch_channel & cur_buf_is_last_step & out_channel_is_last};
        end
    end

    //-----------------------------------------------------------------
    // S1 + S2: lettura Block RAM in 2 cicli (latch interno + DOB_REG).
    //   Registri senza reset e senza enable: Vivado li mette dentro la
    //   primitiva RAMB. I campioni vengono ritardati di pari passo.
    //-----------------------------------------------------------------
    reg        [NIN*W_WENC-1:0] s1_weight_row, s2_weight_row;   // NIN pesi del canale
    reg signed [W_BENC-1:0]     s1_bias,       s2_bias;         // bias del canale
    reg        [W_STEP-1:0]     s1_samples,    s2_samples;      // NIN campioni del timestep

    always @(posedge aclk) begin
        s1_weight_row <= enc_weight_rom[out_channel];
        s1_bias       <= enc_bias_rom[out_channel];
        s1_samples    <= cur_step_buf;
        s2_weight_row <= s1_weight_row;
        s2_bias       <= s1_bias;
        s2_samples    <= s1_samples;
    end

    // Bias ritardato fino a S5: entra nell'albero insieme ai prodotti
    reg signed [W_BENC-1:0] s3_bias, s4_bias, s5_bias;

    always @(posedge aclk) begin
        s3_bias <= s2_bias;
        s4_bias <= s3_bias;
        s5_bias <= s4_bias;
    end

    //-----------------------------------------------------------------
    // Livelli dell'albero, tutti in un unico bus piatto:
    //   tree_bus[level*W_LEVEL_BUS + node*W_ENCACC +: W_ENCACC]
    //   livello 0 = NIN prodotti (nodi 0..NIN-1) + bias (nodo NIN), in S5
    //   livello l = uscita registrata dello stadio S(5+l)
    //-----------------------------------------------------------------
    wire [(TREE_LEVELS+1)*W_LEVEL_BUS-1:0] tree_bus;

    //-----------------------------------------------------------------
    // S3 + S4 + S5: NIN moltiplicatori in parallelo (peso * campione)
    //   DSP48E2 con pipeline completa: S3 registri A/B, S4 M, S5 P.
    //   Peso W_WENC bit sulla porta B, campione W_X bit sulla porta A.
    //-----------------------------------------------------------------
    genvar input_idx;   // indice dell'ingresso (0..NIN-1)
    generate
        for (input_idx = 0; input_idx < NIN; input_idx = input_idx + 1) begin : g_mul
            // posizione del peso di questo ingresso dentro la riga della BRAM
            localparam integer WEIGHT_SLOT = C0_AT_LSB ? input_idx : (NIN - 1 - input_idx);

            wire [N_BIT_IN-1:0] raw_sample = s2_samples[input_idx*N_BIT_IN +: N_BIT_IN];

            (* use_dsp = "yes" *)
            reg  signed [W_WENC-1:0] s3_weight_b;    // -> registro B del DSP
            reg  signed [W_X-1:0]    s3_sample_a;    // -> registro A del DSP
            reg  signed [W_PROD-1:0] s4_product_m;   // -> registro M del DSP
            reg  signed [W_PROD-1:0] s5_product_p;   // -> registro P del DSP

            always @(posedge aclk) begin
                s3_weight_b  <= s2_weight_row[WEIGHT_SLOT*W_WENC +: W_WENC];
                // con segno: copia diretta; senza segno: uno 0 in testa
                s3_sample_a  <= IN_SIGNED ? $signed(raw_sample) : $signed({1'b0, raw_sample});
                s4_product_m <= s3_weight_b * s3_sample_a;
                s5_product_p <= s4_product_m;
            end

            // livello 0 dell'albero: prodotto esteso in segno a W_ENCACC
            assign tree_bus[input_idx*W_ENCACC +: W_ENCACC] =
                {{(W_ENCACC-W_PROD){s5_product_p[W_PROD-1]}}, s5_product_p};
        end
    endgenerate

    // livello 0, ultimo nodo: bias esteso in segno
    assign tree_bus[NIN*W_ENCACC +: W_ENCACC] = {{(W_ENCACC-W_BENC){s5_bias[W_BENC-1]}}, s5_bias};

    //-----------------------------------------------------------------
    // S6 .. S(5+TREE_LEVELS): albero di somma binario generico.
    //   Ogni nodo somma due nodi adiacenti del livello precedente; se il
    //   livello precedente ha un numero dispari di nodi, l'ultimo passa
    //   avanti solo registrato. Un solo sommatore per registro.
    //-----------------------------------------------------------------
    genvar level, node;
    generate
        for (level = 1; level <= TREE_LEVELS; level = level + 1) begin : g_tree_level
            localparam integer PREV_NODES = tree_nodes(level - 1);   // nodi del livello sotto
            localparam integer THIS_NODES = tree_nodes(level);       // nodi di questo livello
            localparam integer PREV_BASE  = (level - 1) * W_LEVEL_BUS;
            localparam integer THIS_BASE  = level * W_LEVEL_BUS;

            for (node = 0; node < TREE_ADDENDS; node = node + 1) begin : g_tree_node
                if (node < THIS_NODES) begin : g_used
                    wire signed [W_ENCACC-1:0] left_child =
                        tree_bus[PREV_BASE + (2*node)*W_ENCACC +: W_ENCACC];
                    reg  signed [W_ENCACC-1:0] partial_sum;

                    if (2*node + 1 < PREV_NODES) begin : g_add
                        wire signed [W_ENCACC-1:0] right_child =
                            tree_bus[PREV_BASE + (2*node+1)*W_ENCACC +: W_ENCACC];
                        always @(posedge aclk)
                            partial_sum <= left_child + right_child;   // un solo sommatore
                    end else begin : g_pass
                        always @(posedge aclk)
                            partial_sum <= left_child;                 // nodo dispari: solo registro
                    end

                    assign tree_bus[THIS_BASE + node*W_ENCACC +: W_ENCACC] = partial_sum;
                end else begin : g_unused
                    // posto vuoto del bus: costante, Vivado lo elimina
                    assign tree_bus[THIS_BASE + node*W_ENCACC +: W_ENCACC] = {W_ENCACC{1'b0}};
                end
            end
        end
    endgenerate

    // radice dell'albero: sum_c W[ch][c]*x[c] + b[ch]
    wire signed [W_ENCACC-1:0] tree_root = tree_bus[TREE_LEVELS*W_LEVEL_BUS +: W_ENCACC];

    //-----------------------------------------------------------------
    // Ultimo stadio: arrotondamento + shift verso F_ACT (un sommatore)
    //-----------------------------------------------------------------
    // mezzo LSB di uscita: sommato prima dello shift da' round-half-up
    localparam signed [W_ENCACC-1:0] ROUND_HALF = (OUT_SHIFT == 0) ? 0 : (1 <<< (OUT_SHIFT-1));

    reg signed [W_ENCACC-1:0] rounded_value;          // valore con F_ACT bit frazionari

    always @(posedge aclk) begin
        // lo shift aritmetico e' solo un ricollegamento di fili
        rounded_value <= (tree_root + ROUND_HALF) >>> OUT_SHIFT;
    end

    //-----------------------------------------------------------------
    // Saturazione a W_ACT bit (combinatoria, scritta nella FIFO).
    //   Il valore sta in W_ACT bit se i bit dal segno di W_ACT in su sono
    //   tutti uguali: un AND/NOR largo, niente confronti con carry.
    //-----------------------------------------------------------------
    localparam integer W_HIGH_PART = W_ENCACC - W_ACT + 1;   // bit [W_ENCACC-1 : W_ACT-1]

    wire [W_HIGH_PART-1:0]  high_part       = rounded_value[W_ENCACC-1 : W_ACT-1];
    wire                    fits_in_w_act   = (&high_part) | (~|high_part);   // tutti 1 o tutti 0
    wire                    out_overflow    = ~fits_in_w_act;
    wire                    value_is_neg    = rounded_value[W_ENCACC-1];

    wire signed [W_ACT-1:0] sat_max_value   = {1'b0, {(W_ACT-1){1'b1}}};      // massimo positivo
    wire signed [W_ACT-1:0] sat_min_value   = {1'b1, {(W_ACT-1){1'b0}}};      // minimo negativo
    wire signed [W_ACT-1:0] saturated_value = fits_in_w_act ? rounded_value[W_ACT-1:0]
                                            : value_is_neg  ? sat_min_value
                                            :                 sat_max_value;

    // Flag di saturazione: si alza se un dato valido dell'ultimo stadio ha saturato
    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn)                                     sat_sticky <= 1'b0;
        else if (stage_valid[PIPE_STAGES] & out_overflow) sat_sticky <= 1'b1;
    end

    //=================================================================
    // 4. FIFO DI USCITA (OUT_FIFO_DEPTH x (W_ACT+1) bit, in LUTRAM)
    //   Parola = {tlast, dato}. Non puo' traboccare: i crediti
    //   garantiscono a ogni canale lanciato un posto gia' riservato.
    //=================================================================
    (* ram_style = "distributed" *)
    reg [W_ACT:0] out_fifo_mem [0:OUT_FIFO_DEPTH-1];

    // puntatori con un bit in piu': uguali = FIFO vuota
    reg [W_FIFO_PTR:0] out_fifo_wr_ptr;
    reg [W_FIFO_PTR:0] out_fifo_rd_ptr;
    wire               out_fifo_empty = (out_fifo_wr_ptr == out_fifo_rd_ptr);

    // scrittura: ogni dato valido che esce dall'ultimo stadio, gia' saturato
    wire out_fifo_write = stage_valid[PIPE_STAGES];

    always @(posedge aclk) begin
        if (out_fifo_write)
            out_fifo_mem[out_fifo_wr_ptr[W_FIFO_PTR-1:0]] <= {stage_last[PIPE_STAGES], saturated_value};
    end

    //-----------------------------------------------------------------
    // Registro di uscita M_AXIS: tdata/tvalid/tlast escono da flip-flop
    //-----------------------------------------------------------------
    reg                    out_reg_valid;
    reg signed [W_ACT-1:0] out_reg_data;
    reg                    out_reg_last;

    assign out_beat_done = out_reg_valid & m_axis_tready;               // beat consegnato
    // il registro si ricarica se e' vuoto o se il suo dato sta uscendo ora
    wire out_reg_load    = ~out_fifo_empty & (~out_reg_valid | m_axis_tready);

    always @(posedge aclk) begin
        if (out_reg_load)
            {out_reg_last, out_reg_data} <= out_fifo_mem[out_fifo_rd_ptr[W_FIFO_PTR-1:0]];
    end

    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn) begin
            out_fifo_wr_ptr <= {(W_FIFO_PTR+1){1'b0}};
            out_fifo_rd_ptr <= {(W_FIFO_PTR+1){1'b0}};
            out_reg_valid   <= 1'b0;
        end else begin
            if (out_fifo_write) out_fifo_wr_ptr <= out_fifo_wr_ptr + 1'b1;
            if (out_reg_load)   out_fifo_rd_ptr <= out_fifo_rd_ptr + 1'b1;

            if (out_reg_load)       out_reg_valid <= 1'b1;   // nuovo dato caricato
            else if (out_beat_done) out_reg_valid <= 1'b0;   // consegnato, niente da ricaricare
        end
    end

    //=================================================================
    // Uscita AXI-Stream
    //=================================================================
    assign m_axis_tvalid = out_reg_valid;
    assign m_axis_tdata  = out_reg_data;
    assign m_axis_tlast  = out_reg_last;

endmodule

`default_nettype wire
