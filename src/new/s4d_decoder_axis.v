`timescale 1ns / 1ps
`default_nettype none
//=====================================================================
// s4d_decoder_axis -- decoder S4D parametrico, AXI-STREAM PURO
//                     (file: s4d_decoder_axis.v)
//
//   S_AXIS : attivazioni W_ACT bit dall'ultimo layer del core,
//            ordine t-major (H canali per timestep, L timestep);
//            tlast = ultimo canale dell'ultimo timestep
//   M_AXIS : i K logit della sequenza, 32 bit per beat, tlast sull'ultimo
//
//   Calcolo (identico ai moduli precedenti quando L e' una potenza di 2):
//
//     y[t][h]  = sat16( round( (w[h]*x[t][h] + b[h]) / 2^BN_SH ) )   BatchNorm finale
//     m[h]     = sat16( floor( (sum_t y[t][h] + floor(L/2)) / L ) )  media su L
//     xq[h]    = sat8 ( round( m[h] / 2^RQ_SHIFT ) )                 riquantizzazione
//     logit[c] = b_cls[c] + sum_h W[h][c] * xq[h]                    classificatore
//
//   round = round-half-up (+ mezzo LSB, poi shift aritmetico).
//   La media e' round-half-up di sum/L anche per L qualsiasi (per L
//   dispari non esistono pareggi).
//
// Cosa cambia rispetto a decoder.v + decoder_1.v + mean_pool.v + classifier.v:
//   - i LOG2_* non sono piu' parametri: si ricavano con $clog2
//     (nel block design vanno tolti LOG2_K, LOG2_L, LOG2_H);
//   - L qualsiasi: potenza di 2 -> shift come prima (stessa latenza);
//     altrimenti divisione ESATTA per costante (moltiplicazione per il
//     reciproco arrotondato in eccesso, metodo Granlund-Montgomery);
//   - K qualsiasi >= 1 e H qualsiasi >= 1;
//   - W_ACC <= 32 (prima era obbligatorio = 32): il logit viene esteso
//     di segno sul beat a 32 bit;
//   - controlli in elaborazione sulle larghezze degli accumulatori.
//   Resta invariato s4d_affine (s4d_affine.v), condiviso con il core.
//
// Gerarchia:
//   s4d_decoder_axis                 (questo file)
//     |- s4d_affine      u_norm      BatchNorm finale       [s4d_affine.v]
//     |- s4d_meanpool    u_pool      media su L             (questo file)
//     |- s4d_classifier  u_cls       int8, logit            (questo file)
//
// Il decoder non applica backpressure in ingresso (tready sempre alto).
// Il pacchetto d'uscita deve essere svuotato prima che arrivino i logit
// della sequenza successiva (almeno L*H cicli dopo): in simulazione un
// controllo segnala la violazione. Sequenze consecutive senza pausa in
// ingresso sono ammesse (il drain del mean pool legge ogni accumulatore
// prima che il nuovo frame lo riscriva).
//=====================================================================

module s4d_decoder_axis #(
    parameter integer H        = 128,     // canali
    parameter integer L        = 1024,    // timestep per sequenza (qualsiasi >= 1)
    parameter integer K        = 5,       // classi: 5 ECG, 10 sCIFAR, 35 Speech Commands

    parameter integer W_ACT    = 16,      // attivazioni in ingresso (Q5.11)
    parameter integer W_CIN    = 8,       // ingresso del classificatore (int8)
    parameter integer W_CLS    = 8,       // pesi del classificatore (int8)
    parameter integer W_ACC    = 32,      // accumulatori / logit (<= 32, beat d'uscita a 32 bit)
    parameter integer W_POOL   = 32,      // accumulatore per canale del mean pool

    parameter integer F_ACT    = 11,      // bit frazionari delle attivazioni (documentazione)
    parameter integer RQ_SHIFT = 6,       // shift Q5.11 -> int8 prima del classificatore

    // ---- BatchNorm finale (shift fisso, pesi in ROM) ----
    // Valore per progetto (quant_report.json, hw_generics.BN_SH_FINAL):
    // PTB-XL/ECG 13, sCIFAR 12. Default = ECG (target ZCU104 attuale).
    parameter integer BN_SH    = 13,
    parameter integer BN_W_W   = 16,
    parameter integer BN_W_B   = 32,
    parameter integer BN_W_ACC = 34,

    // file .mem letti in sintesi ($readmemh)
    parameter         BN_W_FILE = "l6_bn_w_ECG.mem",
    parameter         BN_B_FILE = "l6_bn_b_ECG.mem",
    parameter         W_FILE    = "cls_w_ECG.mem",
    parameter         B_FILE    = "cls_b_ECG.mem"
)(
    input  wire                    aclk,
    input  wire                    aresetn,           // reset asincrono attivo basso

    // ---- ingresso dal core: L*H beat, tlast sull'ultimo ----
    input  wire                    s_axis_tvalid,
    output wire                    s_axis_tready,     // sempre 1
    input  wire signed [W_ACT-1:0] s_axis_tdata,
    input  wire                    s_axis_tlast,

    // ---- logit: K beat da 32 bit, tlast sull'ultimo ----
    output wire                    m_axis_tvalid,
    input  wire                    m_axis_tready,
    output wire [31:0]             m_axis_tdata,
    output wire                    m_axis_tlast
);

    // larghezze dei contatori (almeno 1 bit anche per H = 1 o K = 1)
    localparam integer H_W   = (H > 1) ? $clog2(H) : 1;   // indice di canale
    localparam integer IDX_W = (K > 1) ? $clog2(K) : 1;   // indice di classe

    //=================================================================
    // Controlli in elaborazione: un parametro incoerente ferma la sintesi
    //=================================================================
    initial begin
        if (H < 1 || L < 1 || K < 1) begin
            $display("s4d_decoder_axis: H=%0d L=%0d K=%0d devono essere >= 1", H, L, K);
            $finish;
        end
        if (W_ACC > 32) begin
            $display("s4d_decoder_axis: W_ACC=%0d, il beat d'uscita e' di 32 bit", W_ACC);
            $finish;
        end
    end

    //=================================================================
    // BatchNorm finale per canale: y = (w*x + b) >> BN_SH
    //   LOADABLE = 0: pesi da BN_W_FILE/BN_B_FILE, porte di caricamento a 0.
    //   out_ready = 1: la pipeline non si ferma mai.
    //=================================================================
    wire                    norm_valid;
    wire signed [W_ACT-1:0] norm_data;
    wire                    norm_last_ch;      // ultimo canale del timestep
    wire                    norm_last_seq;     // ultimo campione della sequenza

    s4d_affine #(
        .H(H), .LOG2_H(H_W),
        .W_IN(W_ACT), .W_OUT(W_ACT),
        .W_W(BN_W_W), .W_B(BN_W_B), .W_ACC(BN_W_ACC), .SH(BN_SH),
        .W_FILE(BN_W_FILE), .B_FILE(BN_B_FILE)
    ) u_norm (
        .clk          (aclk),
        .rst_n        (aresetn),
        .ld_we_w      (1'b0),                 // caricamento a runtime non usato
        .ld_we_b      (1'b0),
        .ld_addr      ({H_W{1'b0}}),
        .ld_w         ({BN_W_W{1'b0}}),
        .ld_b         ({BN_W_B{1'b0}}),
        .ld_we_sh     (1'b0),
        .ld_sh        (6'd0),
        .in_valid     (s_axis_tvalid),
        .in_ready     (s_axis_tready),
        .in_data      (s_axis_tdata),
        .in_last_seq  (s_axis_tlast),
        .out_ready    (1'b1),                 // nessuna backpressure
        .out_valid    (norm_valid),
        .out_data     (norm_data),
        .out_last_ch  (norm_last_ch),
        .out_last_seq (norm_last_seq)
    );

    //=================================================================
    // Media temporale per canale
    //=================================================================
    wire                    pool_valid;
    wire signed [W_ACT-1:0] pool_data;
    wire                    pool_last;

    s4d_meanpool #(
        .H(H), .L(L), .W_IN(W_ACT), .W_ACC(W_POOL), .W_OUT(W_ACT)
    ) u_pool (
        .clk         (aclk),
        .rst_n       (aresetn),
        .in_valid    (norm_valid),
        .in_data     (norm_data),
        .in_last_ch  (norm_last_ch),
        .in_last_seq (norm_last_seq),
        .out_valid   (pool_valid),
        .out_data    (pool_data),
        .out_last    (pool_last)
    );

    //=================================================================
    // Classificatore lineare int8 (solo logit, nessun argmax)
    //=================================================================
    wire                    logit_valid, logit_last;
    wire [IDX_W-1:0]        logit_idx;
    wire signed [W_ACC-1:0] logit_data;

    s4d_classifier #(
        .H(H), .K(K),
        .W_IN(W_ACT), .W_CIN(W_CIN), .W_CLS(W_CLS), .W_ACC(W_ACC),
        .RQ_SHIFT(RQ_SHIFT), .W_FILE(W_FILE), .B_FILE(B_FILE)
    ) u_cls (
        .clk         (aclk),
        .rst_n       (aresetn),
        .in_valid    (pool_valid),
        .in_data     (pool_data),
        .in_last     (pool_last),
        .out_valid   (logit_valid),
        .out_idx     (logit_idx),
        .out_logit   (logit_data),
        .out_last    (logit_last)
    );

    //=================================================================
    // Cattura dei logit: arrivano 1 per ciclo senza backpressure, vanno
    // parcheggiati prima di poterli mandare su M_AXIS
    //=================================================================
    reg signed [W_ACC-1:0] logit_buf [0:K-1];
    always @(posedge aclk) if (logit_valid) logit_buf[logit_idx] <= logit_data;

    //=================================================================
    // Invio del pacchetto: K beat letti da logit_buf
    //   parte nel ciclo in cui arriva l'ultimo logit (logit_buf[0..K-2]
    //   sono gia' scritti; logit_buf[K-1] viene scritto su questo fronte
    //   e letto non prima di K-1 beat dopo). Con K = 1 il primo beat
    //   e' preso direttamente da logit_data.
    //=================================================================
    reg              sending;     // 1 = pacchetto in corso
    reg [IDX_W-1:0]  send_idx;    // logit attualmente sul registro d'uscita
    reg [IDX_W-1:0]  next_idx;    // = send_idx + 1, tenuto in registro: il mux
                                  //   logit_buf[next_idx] non aspetta il sommatore

    reg              o_valid;     // registro d'uscita M_AXIS
    reg [31:0]       o_data;
    reg              o_last;

    assign m_axis_tvalid = o_valid;
    assign m_axis_tdata  = o_data;
    assign m_axis_tlast  = o_last;

    wire out_beat  = o_valid && m_axis_tready;        // beat d'uscita accettato
    wire pkt_start = logit_valid && logit_last;       // ultimo logit della sequenza arrivato

    // logit esteso di segno a 32 bit
    function [31:0] sext32;
        input signed [W_ACC-1:0] value;
        begin
            sext32 = {{(33-W_ACC){value[W_ACC-1]}}, value[W_ACC-2:0]};
        end
    endfunction

    // primo beat: con K = 1 il logit sta arrivando ora e non e' ancora in logit_buf
    wire signed [W_ACC-1:0] first_logit = (K == 1) ? logit_data : logit_buf[0];

    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn) begin
            sending  <= 1'b0;
            send_idx <= {IDX_W{1'b0}};
            next_idx <= {IDX_W{1'b0}};
            o_valid  <= 1'b0;
            o_data   <= 32'd0;
            o_last   <= 1'b0;
        end else if (!sending) begin
            if (pkt_start) begin
                // primo logit del pacchetto
                sending  <= 1'b1;
                send_idx <= {IDX_W{1'b0}};
                next_idx <= {{(IDX_W-1){1'b0}}, 1'b1};
                o_valid  <= 1'b1;
                o_data   <= sext32(first_logit);
                o_last   <= (K == 1);
            end
        end else if (out_beat) begin
            if (send_idx == K-1) begin
                // ultimo logit accettato: pacchetto finito
                sending <= 1'b0;
                o_valid <= 1'b0;
                o_last  <= 1'b0;
            end else begin
                // logit successivo
                send_idx <= next_idx;
                next_idx <= next_idx + 1'b1;
                o_data   <= sext32(logit_buf[next_idx]);
                o_last   <= (next_idx == K-1);
            end
        end
    end

    //=================================================================
    // Controlli solo in simulazione (nessuna logica in sintesi)
    //=================================================================
// synthesis translate_off
    localparam integer NBEAT = L * H;             // beat per sequenza
    integer sim_beat_count = 0;                   // beat ricevuti nella sequenza corrente
    always @(posedge aclk) if (aresetn) begin
        // contratto tlast: esattamente L*H beat per sequenza
        if (s_axis_tvalid && s_axis_tready) begin
            if (s_axis_tlast) begin
                if (sim_beat_count != NBEAT-1)
                    $display("s4d_decoder_axis: tlast al beat %0d, atteso %0d", sim_beat_count, NBEAT-1);
                sim_beat_count = 0;
            end else begin
                sim_beat_count = sim_beat_count + 1;
            end
        end
        // nuovi logit mentre il pacchetto precedente non e' ancora uscito
        if (logit_valid && sending)
            $display("s4d_decoder_axis: logit nuovi durante l'invio del pacchetto precedente (sovrascrittura)");
    end
// synthesis translate_on

endmodule


//=====================================================================
// s4d_meanpool -- media temporale per canale (global average pool)
//
//   Riceve L timestep x H canali in ordine t-major e, per ogni canale,
//   accumula acc[ch] = sum_t x[t][ch]. A fine sequenza emette in
//   streaming (un canale per ciclo, H cicli consecutivi):
//
//     out[ch] = sat( floor( (acc[ch] + floor(L/2)) / L ) )
//
//   cioe' la media arrotondata (round-half-up) nello stesso formato
//   dell'ingresso.
//
//   Accumulo in 2 stadi (timing: lettura LUTRAM e somma in cicli diversi):
//     A1  lettura di acc[ch] (con inoltro del valore in scrittura se il
//         canale e' lo stesso del campione precedente, cioe' H = 1)
//     A2  somma e scrittura di acc[ch]
//   Drain (dopo che l'ultimo campione della sequenza e' stato scritto):
//     D1          lettura registrata di acc[ocnt]               -> acc_rd
//     L = 2^n:    D2  acc_rd + L/2
//                 D3  shift di n bit (fili) e saturazione       -> uscita
//     L qualsiasi: divisione esatta per la costante L:
//       D2  num  = acc_rd + floor(L/2) + 2^(W_IN-1)*L   (>= 0, < 2^NUM_W)
//       D3  prod = num * MAGIC                           (registro M del DSP)
//       D4  prod registrato                              (registro P del DSP)
//       D5  quot = prod >> SHIFT (fili);  media = quot - 2^(W_IN-1)
//       D6  saturazione                                  -> uscita
//     con MAGIC = ceil(2^SHIFT / L), SHIFT = NUM_W + ceil(log2 L):
//     floor(num/L) = (num*MAGIC) >> SHIFT e' esatto per ogni num < 2^NUM_W
//     (Granlund-Montgomery, "Division by invariant integers", 1994).
//     L'offset 2^(W_IN-1)*L e' un multiplo di L: rende num non negativo
//     senza cambiare il resto, e si toglie dal quoziente come 2^(W_IN-1).
//
//   Accumulo e drain possono sovrapporsi (sequenza successiva senza
//   pausa): D1 legge acc[k] sullo stesso fronte o prima di quello in cui
//   il primo frame della nuova sequenza riscrive acc[k].
//=====================================================================
module s4d_meanpool #(
    parameter integer H      = 128,       // canali
    parameter integer L      = 1024,      // timestep per sequenza
    parameter integer W_IN   = 16,        // ingresso (Q5.11)
    parameter integer W_ACC  = 32,        // accumulatore per canale
    parameter integer W_OUT  = 16         // uscita (stesso formato dell'ingresso)
)(
    input  wire                    clk,
    input  wire                    rst_n,             // reset asincrono attivo basso

    // ---- ingresso: un campione per beat, frame di H canali ----
    input  wire                    in_valid,
    input  wire signed [W_IN-1:0]  in_data,
    input  wire                    in_last_ch,        // ultimo canale del frame (timestep)
    input  wire                    in_last_seq,       // ultimo campione della sequenza

    // ---- uscita: H medie, una per ciclo ----
    output reg                     out_valid,
    output reg  signed [W_OUT-1:0] out_data,
    output reg                     out_last           // ultimo canale (ch = H-1)
);

    localparam integer H_W     = (H > 1) ? $clog2(H) : 1;   // indice di canale
    localparam integer LOG2_L  = $clog2(L);                 // ceil(log2 L), 0 per L = 1
    localparam         L_POW2  = ((L & (L - 1)) == 0);      // 1 se L e' una potenza di 2
    localparam integer HALF_L  = L / 2;                     // floor(L/2): mezzo LSB della media

    // divisione per costante (usata solo se L non e' una potenza di 2)
    localparam integer NUM_W   = W_IN + LOG2_L;             // bit del numeratore non negativo
    localparam integer SHIFT   = NUM_W + LOG2_L;            // shift del prodotto
    localparam integer MAGIC_W = NUM_W + 1;                 // MAGIC < 2^(NUM_W+1)
    localparam integer PROD_W  = NUM_W + MAGIC_W;

    // MAGIC = ceil(2^SHIFT / L), calcolato in elaborazione su 128 bit
    function [MAGIC_W-1:0] div_magic;
        input integer divisor;
        input integer shift_bits;
        reg [127:0] power_of_two;
        begin
            power_of_two = 128'd1 << shift_bits;
            div_magic    = (power_of_two + divisor - 1) / divisor;
        end
    endfunction
    localparam [MAGIC_W-1:0] MAGIC = div_magic(L, SHIFT);

    // limiti di saturazione a W_OUT bit (in pratica mai raggiunti: la
    // media di valori a W_IN bit sta in W_IN bit)
    localparam signed [W_ACC-1:0] SAT_MAX = (1 <<< (W_OUT-1)) - 1;
    localparam signed [W_ACC-1:0] SAT_MIN = -(1 <<< (W_OUT-1));

    initial begin
        // |acc| <= 2^(W_IN-1) * L: servono W_IN + ceil(log2 L) bit con segno
        if (W_ACC < W_IN + LOG2_L) begin
            $display("s4d_meanpool: W_ACC=%0d < W_IN+ceil(log2 L)=%0d, overflow possibile",
                     W_ACC, W_IN + LOG2_L);
            $finish;
        end
        if (!L_POW2 && SHIFT > 120) begin
            $display("s4d_meanpool: SHIFT=%0d troppo grande per div_magic", SHIFT);
            $finish;
        end
    end

    //-----------------------------------------------------------------
    // Accumulo: primo timestep -> acc = x, altri timestep -> acc += x
    // (il flag first fa da "clear" all'inizio di ogni sequenza)
    //-----------------------------------------------------------------
    reg signed [W_ACC-1:0] acc [0:H-1];       // LUTRAM: 1 porta lett./scritt. + 1 di lettura
    reg [H_W-1:0]          ch;                // canale del prossimo campione in ingresso
    reg                    first;             // 1 = frame corrente = primo della sequenza

    wire signed [W_ACC-1:0] in_ext  = {{(W_ACC-W_IN){in_data[W_IN-1]}}, in_data};

    // ---- A1: lettura dell'accumulatore del canale ----
    reg                    a1_valid, a1_first, a1_last_seq;
    reg [H_W-1:0]          a1_ch;
    reg signed [W_ACC-1:0] a1_in, a1_acc;

    // ---- A2: valore da scrivere (nuovo accumulatore) ----
    wire signed [W_ACC-1:0] a2_value = a1_first ? a1_in : (a1_acc + a1_in);

    // inoltro: il canale letto ora e' quello che A2 sta scrivendo (solo H = 1)
    wire forward = a1_valid && (a1_ch == ch);

    always @(posedge clk) begin
        a1_ch    <= ch;
        a1_in    <= in_ext;
        a1_first <= first;
        a1_acc   <= forward ? a2_value : acc[ch];
        if (a1_valid) acc[a1_ch] <= a2_value;
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            a1_valid    <= 1'b0;
            a1_last_seq <= 1'b0;
        end else begin
            a1_valid    <= in_valid;
            a1_last_seq <= in_valid && in_last_seq;
        end
    end

    // contatore di canale riallineato su in_last_ch (generato da s4d_affine)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ch    <= {H_W{1'b0}};
            first <= 1'b1;              // la prima sequenza dopo il reset parte da zero
        end else if (in_valid) begin
            if (in_last_ch) begin
                ch    <= {H_W{1'b0}};
                first <= in_last_seq;   // 1 solo dopo l'ultimo frame della sequenza
            end else begin
                ch <= ch + 1'b1;
            end
        end
    end

    //-----------------------------------------------------------------
    // Drain, stadio D1: scansione ocnt = 0..H-1 e lettura registrata
    //-----------------------------------------------------------------
    reg                    drain;             // 1 = D1 attivo
    reg [H_W-1:0]          ocnt;              // canale letto in D1
    reg signed [W_ACC-1:0] acc_rd;            // acc[ocnt] registrato
    reg                    rd_valid, rd_last; // acc_rd valido / ultimo canale

    always @(posedge clk) acc_rd <= acc[ocnt];   // dato: niente reset

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            drain    <= 1'b0;
            ocnt     <= {H_W{1'b0}};
            rd_valid <= 1'b0;
            rd_last  <= 1'b0;
        end else begin
            rd_valid <= drain;
            rd_last  <= drain && (ocnt == H-1);
            if (!drain) begin
                if (a1_last_seq) begin               // ultimo campione scritto ora: si inizia a leggere
                    drain <= 1'b1;
                    ocnt  <= {H_W{1'b0}};
                end
            end else begin
                if (ocnt == H-1) drain <= 1'b0;      // letti tutti i canali
                else             ocnt  <= ocnt + 1'b1;
            end
        end
    end

    generate
    if (L_POW2) begin : g_shift
        //-------------------------------------------------------------
        // L = 2^n: D2 acc_rd + L/2;  D3 >>> n (fili) e saturazione
        //-------------------------------------------------------------
        reg signed [W_ACC-1:0] sum_d2;
        reg                    valid_d2, last_d2;

        always @(posedge clk) sum_d2 <= acc_rd + HALF_L;

        wire signed [W_ACC-1:0] avg = sum_d2 >>> LOG2_L;

        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                valid_d2  <= 1'b0;
                last_d2   <= 1'b0;
                out_valid <= 1'b0;
                out_data  <= {W_OUT{1'b0}};
                out_last  <= 1'b0;
            end else begin
                valid_d2  <= rd_valid;
                last_d2   <= rd_last;
                out_valid <= valid_d2;
                out_last  <= last_d2;
                if      (avg > SAT_MAX) out_data <= SAT_MAX[W_OUT-1:0];
                else if (avg < SAT_MIN) out_data <= SAT_MIN[W_OUT-1:0];
                else                    out_data <= avg[W_OUT-1:0];
            end
        end
    end else begin : g_divide
        //-------------------------------------------------------------
        // L qualsiasi: divisione esatta per costante (D2..D5)
        //-------------------------------------------------------------
        // offset multiplo di L che rende il numeratore non negativo
        localparam signed [W_ACC+1:0] NUM_OFFSET = (64'sd1 <<< (W_IN-1)) * L + HALF_L;
        localparam signed [W_ACC+1:0] QUOT_BIAS  = (64'sd1 <<< (W_IN-1));   // offset / L

        reg [NUM_W-1:0]  num_d2;              // D2: numeratore non negativo
        reg [PROD_W-1:0] prod_d3, prod_d4;    // D3/D4: prodotto (registri M e P del DSP)
        reg [3:0]        valid_pipe, last_pipe;   // valid/last di D2, D3, D4, D5

        // acc_rd esteso di 2 bit per sommare l'offset senza overflow
        wire signed [W_ACC+1:0] num_full = {{2{acc_rd[W_ACC-1]}}, acc_rd} + NUM_OFFSET;

        always @(posedge clk) begin
            num_d2  <= num_full[NUM_W-1:0];   // 0 <= num_full < 2^NUM_W per costruzione
            prod_d3 <= num_d2 * MAGIC;
            prod_d4 <= prod_d3;
        end

        // D5: quoziente (fili), tolta la parte dell'offset
        wire signed [W_ACC+1:0] quot = $signed({2'b00, prod_d4[PROD_W-1:SHIFT]});
        reg  signed [W_ACC+1:0] avg;
        always @(posedge clk) avg <= quot - QUOT_BIAS;
        // D6: saturazione (sotto)

        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                valid_pipe <= 4'b0000;
                last_pipe  <= 4'b0000;
                out_valid  <= 1'b0;
                out_data   <= {W_OUT{1'b0}};
                out_last   <= 1'b0;
            end else begin
                valid_pipe <= {valid_pipe[2:0], rd_valid};
                last_pipe  <= {last_pipe[2:0],  rd_last};
                out_valid  <= valid_pipe[3];
                out_last   <= last_pipe[3];
                if      (avg > SAT_MAX) out_data <= SAT_MAX[W_OUT-1:0];
                else if (avg < SAT_MIN) out_data <= SAT_MIN[W_OUT-1:0];
                else                    out_data <= avg[W_OUT-1:0];
            end
        end
    end
    endgenerate

endmodule


//=====================================================================
// s4d_classifier -- classificatore lineare int8 (H -> K), solo logit
//
//     xq[h]    = sat_int8( round( x[h] / 2^RQ_SHIFT ) )
//     logit[c] = b[c] + sum_{h<H} W[h][c] * xq[h]           c = 0..K-1
//
//   Formati: x W_IN bit; xq W_CIN bit; W W_CLS bit con segno;
//   b, acc, logit W_ACC bit nella scala del prodotto W*xq.
//
//   ROM da file .mem:
//     W_FILE : H righe da K*W_CLS bit, classe c nei bit [c*W_CLS +: W_CLS]
//     B_FILE : K righe da W_ACC bit
//
//   Pipeline (classifier.v con uno stadio in piu', timing):
//     CA (in_valid)  x + mezzo LSB della riquantizzazione
//     C0 (ca_valid)  shift (fili) e saturazione -> xq, lettura riga W[h]
//     C1 (c0_valid)  K prodotti registrati
//     C2 (c1_valid)  K accumuli (primo canale: acc = b + prod)
//     E  (mac_done)  emissione dei K logit, 1 per ciclo
//=====================================================================
module s4d_classifier #(
    parameter integer H        = 128,     // ingressi (canali)
    parameter integer K        = 5,       // classi
    parameter integer W_IN     = 16,      // ingresso x (Q5.11)
    parameter integer W_CIN    = 8,       // ingresso riquantizzato xq
    parameter integer W_CLS    = 8,       // pesi W
    parameter integer W_ACC    = 32,      // accumulatori, bias e logit
    parameter integer RQ_SHIFT = 6,       // shift di riquantizzazione x -> xq
    parameter         W_FILE   = "cls_w_ECG.mem",
    parameter         B_FILE   = "cls_b_ECG.mem"
)(
    input  wire                    clk,
    input  wire                    rst_n,             // reset asincrono attivo basso

    // ---- ingresso: H campioni, in_last sull'ultimo ----
    input  wire                    in_valid,
    input  wire signed [W_IN-1:0]  in_data,
    input  wire                    in_last,

    // ---- logit: K beat consecutivi (indice + valore) ----
    output reg                     out_valid,
    output reg  [((K > 1) ? $clog2(K) : 1)-1:0] out_idx,   // classe del logit
    output reg  signed [W_ACC-1:0] out_logit,
    output reg                     out_last           // ultimo logit (classe K-1)
);

    localparam integer H_W   = (H > 1) ? $clog2(H) : 1;
    localparam integer IDX_W = (K > 1) ? $clog2(K) : 1;
    localparam integer W_PROD = W_CLS + W_CIN;            // bit di un prodotto

    initial begin
        // somma di H prodotti senza bias: servono W_PROD + ceil(log2 H) bit
        if (W_ACC < W_PROD + $clog2(H)) begin
            $display("s4d_classifier: W_ACC=%0d < W_CLS+W_CIN+ceil(log2 H)=%0d",
                     W_ACC, W_PROD + $clog2(H));
            $finish;
        end
        if (RQ_SHIFT < 0 || RQ_SHIFT >= W_IN) begin
            $display("s4d_classifier: RQ_SHIFT=%0d fuori da [0, W_IN)", RQ_SHIFT);
            $finish;
        end
    end

    // ROM dei pesi (una riga = i K pesi di un canale) e dei bias
    reg [K*W_CLS-1:0]      w_rom [0:H-1];
    reg signed [W_ACC-1:0] b_rom [0:K-1];

    initial begin
        $readmemh(W_FILE, w_rom);
        $readmemh(B_FILE, b_rom);
    end

    //-----------------------------------------------------------------
    // Riquantizzazione x -> xq: (x + 2^(RQ_SHIFT-1)) >>> RQ_SHIFT, poi
    // saturazione a W_CIN bit; calcolo su W_IN+1 bit per il riporto
    //-----------------------------------------------------------------
    localparam signed [W_IN:0] CSAT_MAX = (1 <<< (W_CIN-1)) - 1;
    localparam signed [W_IN:0] CSAT_MIN = -(1 <<< (W_CIN-1));

    wire signed [W_IN:0] rq_rnd = (RQ_SHIFT == 0)
                                ? {(W_IN+1){1'b0}}
                                : ({{W_IN{1'b0}}, 1'b1} <<< (RQ_SHIFT-1));   // mezzo LSB
    wire signed [W_IN:0] rq_ext = {in_data[W_IN-1], in_data};

    //-----------------------------------------------------------------
    // CA: somma del mezzo LSB (W_IN+1 bit per il riporto)
    //-----------------------------------------------------------------
    reg                    ca_valid, ca_first, ca_last;
    reg [H_W-1:0]          hcnt;              // canale del prossimo ingresso
    reg [H_W-1:0]          ca_h;              // canale del dato in CA
    reg signed [W_IN:0]    ca_sum;            // x + 2^(RQ_SHIFT-1)

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ca_valid <= 1'b0;
            ca_first <= 1'b0;
            ca_last  <= 1'b0;
            hcnt     <= {H_W{1'b0}};
        end else begin
            ca_valid <= in_valid;
            ca_first <= in_valid && (hcnt == 0);
            ca_last  <= in_valid && in_last;
            if (in_valid) hcnt <= in_last ? {H_W{1'b0}} : (hcnt + 1'b1);
        end
    end

    always @(posedge clk) begin
        ca_h   <= hcnt;
        ca_sum <= rq_ext + rq_rnd;
    end

    //-----------------------------------------------------------------
    // C0: shift (fili) e saturazione -> xq; lettura della riga di pesi
    //-----------------------------------------------------------------
    wire signed [W_IN:0] rq_val = ca_sum >>> RQ_SHIFT;

    reg signed [W_CIN-1:0] xq;
    reg                    c0_valid, c0_first, c0_last;
    reg [K*W_CLS-1:0]      w_row;             // pesi W[h][0..K-1] del canale corrente

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            c0_valid <= 1'b0;
            c0_first <= 1'b0;
            c0_last  <= 1'b0;
        end else begin
            c0_valid <= ca_valid;
            c0_first <= ca_first;
            c0_last  <= ca_last;
        end
    end

    always @(posedge clk) begin
        w_row <= w_rom[ca_h];
        if      (rq_val > CSAT_MAX) xq <= CSAT_MAX[W_CIN-1:0];
        else if (rq_val < CSAT_MIN) xq <= CSAT_MIN[W_CIN-1:0];
        else                        xq <= rq_val[W_CIN-1:0];
    end

    //-----------------------------------------------------------------
    // C1 (prodotto) e C2 (accumulo), K classi in parallelo
    //-----------------------------------------------------------------
    reg c1_valid, c1_first, c1_last;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            c1_valid <= 1'b0;
            c1_first <= 1'b0;
            c1_last  <= 1'b0;
        end else begin
            c1_valid <= c0_valid;
            c1_first <= c0_first;
            c1_last  <= c0_last;
        end
    end

    reg signed [W_ACC-1:0] acc [0:K-1];       // logit in accumulo

    genvar class_c;
    generate
        for (class_c = 0; class_c < K; class_c = class_c + 1) begin : g_mac
            wire signed [W_CLS-1:0]  w_c = $signed(w_row[class_c*W_CLS +: W_CLS]);
            reg  signed [W_PROD-1:0] prod_q;                              // prodotto registrato
            wire signed [W_ACC-1:0]  prod_ext = {{(W_ACC-W_PROD){prod_q[W_PROD-1]}}, prod_q};

            always @(posedge clk)                                         // C1
                if (c0_valid) prod_q <= w_c * xq;

            always @(posedge clk)                                         // C2
                if (c1_valid)
                    acc[class_c] <= (c1_first ? b_rom[class_c] : acc[class_c]) + prod_ext;
        end
    endgenerate

    // il ciclo dopo l'ultimo accumulo tutti gli acc sono completi
    reg mac_done;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) mac_done <= 1'b0;
        else        mac_done <= c1_valid && c1_last;
    end

    //-----------------------------------------------------------------
    // E: emissione dei K logit, uno per ciclo
    //-----------------------------------------------------------------
    reg             emit;                     // 1 = emissione in corso
    reg [IDX_W-1:0] ecnt;                     // classe in emissione

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            emit      <= 1'b0;
            ecnt      <= {IDX_W{1'b0}};
            out_valid <= 1'b0;
            out_idx   <= {IDX_W{1'b0}};
            out_logit <= {W_ACC{1'b0}};
            out_last  <= 1'b0;
        end else begin
            out_valid <= 1'b0;                // default: impulsi di 1 ciclo
            out_last  <= 1'b0;
            if (!emit) begin
                if (mac_done) begin
                    emit <= 1'b1;
                    ecnt <= {IDX_W{1'b0}};
                end
            end else begin
                out_valid <= 1'b1;
                out_idx   <= ecnt;
                out_logit <= acc[ecnt];       // mux K:1 registrato
                out_last  <= (ecnt == K-1);
                if (ecnt == K-1) emit <= 1'b0;
                else             ecnt <= ecnt + 1'b1;
            end
        end
    end

endmodule

`default_nettype wire
