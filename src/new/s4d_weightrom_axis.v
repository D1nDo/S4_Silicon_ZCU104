`timescale 1ns / 1ps
`default_nettype none

// ============================================================================
// s4d_weightrom_axis -- pesi di TUTTI i layer fissi in una ROM on-chip (BRAM),
// inviati come master AXI-Stream a 64 bit, un beat alla volta, verso
// l'ingresso s_axis_w_* di s4d_core_axis. Sostituisce la DMA dei pesi.
//
// CONTENUTO DELLA ROM: w_stream_ECG_nmac256.mem (stream con i pesi del mixing
// nel formato a fase singola, NMAC=256), NLAYER*NBEAT righe da 64 bit:
//     righe [0       .. NBEAT-1]      -> layer 0
//     righe [NBEAT   .. 2*NBEAT-1]    -> layer 1
//     ...
//     righe [5*NBEAT .. 6*NBEAT-1]    -> layer 5
// La riga i e' esattamente il tdata che la DMA avrebbe messo sul beat i.
//
// FUNZIONAMENTO (versione 2026-10-02, pipeline per il timing):
//
//     puntatore --> ROM (BRAM) --> rom_q1 --> rom_q2 --> coda (8 posti) --> m_axis
//     rd_rom_addr   lettura        reg BRAM   reg uscita   fifo_*            tdata = testa
//                   SEMPRE attiva             BRAM (DOB)
//
//   - La ROM viene letta a ogni ciclo, senza clock enable: prima l'enable
//     era (!tvalid || tready), cioe' il tready del core arrivava a tutte le
//     176 BRAM (slack misurato 0.57 ns a 177.8 MHz). Ora tready comanda
//     solo l'estrazione dalla coda (pochi registri).
//   - Una lettura viene "emessa" (il puntatore avanza e il dato verra'
//     messo in coda 2 cicli dopo) solo se c'e' posto garantito in coda:
//     contatore read_credits = letture in volo + parole in coda < 8.
//     Cosi' la coda non trabocca mai, anche se il core ferma tready.
//   - A regime: un beat per ciclo (giro emissione -> coda -> estrazione di
//     4 cicli, 8 crediti).
//   - rd_rom_addr ha max_fanout: la sintesi ne fa copie per gruppi di BRAM
//     (prima: indirizzo -> 176 BRAM, slack 0.64 ns).
//
// RIALLINEAMENTO PER LAYER (wload_start / wload_layer):
//   Il core segnala con un impulso wload_start (stesso ciclo di
//   u_wload.start) l'inizio del caricamento di un layer, e con wload_layer
//   QUALE layer. Su quell'impulso lo streamer:
//     - porta il puntatore all'inizio di quel layer (wload_layer*NBEAT);
//     - svuota la coda e scarta le letture in volo;
//     - non emette letture in quel ciclo.
//   Timing: nel ciclo dell'impulso s4d_wload ha ancora tready=0 (busy sale
//   al ciclo dopo), quindi nessun beat viene preso dal core durante lo
//   svuotamento; dal ciclo dopo il core ha tready=1 con tvalid=0 (legale,
//   aspetta) e la prima riga del layer e' in testa alla coda 4 cicli dopo
//   l'impulso.
//   Fallback: senza impulsi, dopo l'ultimo beat del layer 5 il puntatore
//   torna a 0 da solo (stesso ordine con cui il core itera).
//
// RESET: ponte di reset locale (attivazione asincrona, rilascio sincrono
// dopo 2 cicli), identico ai ponti di s4d_core_axis: core e streamer escono
// dal reset nello stesso ciclo.
//
// Stima risorse: 89862 x 64 bit ~ 5.75 Mbit ~ 176 BRAM36 (ZCU104: 312).
// URAM non utilizzabile qui: non e' inizializzabile da bitstream.
// Versione precedente (registro di uscita unico): s4d_weight_injector.v.bak2
// ============================================================================
module s4d_weightrom_axis #(
    parameter integer NBEAT        = 14977,   // beat per layer (= NBEAT di s4d_wload)
    parameter integer NLAYER       = 6,       // protetto in CLAUDE.md
    parameter integer AXI_DW       = 64,      // protetto in CLAUDE.md
    parameter         WEIGHTS_FILE = "w_stream_nmac256_ECG.mem"   // formato NMAC=256
)(
    input  wire                  clk,          // stesso clock del core
    input  wire                  rst_n,        // stesso reset del core

    // ---- riallineamento dal core (s4d_core_axis) ----
    input  wire                        wload_start,  // impulso: il core inizia a caricare un layer
    input  wire [$clog2(NLAYER)-1:0]   wload_layer,  // layer che il core sta per caricare (0..NLAYER-1)

    // ---- master AXI-Stream verso s4d_core_axis.s_axis_w_* ----
    output wire                  m_axis_tvalid,
    input  wire                  m_axis_tready,
    output wire [AXI_DW-1:0]     m_axis_tdata,
    output wire                  m_axis_tlast,  // 1 sull'ultimo beat di ogni layer
                                                // (il core lo ignora: solo diagnostica)

    // ---- diagnostica (possono restare scollegate) ----
    output wire                        layer_done,  // impulso: ultimo beat di un layer consumato
    output wire [$clog2(NLAYER)-1:0]   layer_idx    // layer a cui appartiene il beat in testa
);

    // ------------------------------------------------------------------
    // Dimensioni derivate
    // ------------------------------------------------------------------
    localparam integer TOTAL_BEATS   = NLAYER * NBEAT;          // righe della ROM (89862)
    localparam integer ROM_ADDR_W    = $clog2(TOTAL_BEATS);     // 17 bit
    localparam integer BEAT_IN_LAY_W = $clog2(NBEAT);           // 14 bit
    localparam integer LAYER_W       = $clog2(NLAYER);          // 3 bit

    localparam integer FIFO_DEPTH    = 8;                       // posti della coda d'uscita
    localparam integer FIFO_IDX_W    = 3;                       // bit degli indici della coda
    localparam integer COUNT_W       = 4;                       // bit dei contatori (0..8)

    // ------------------------------------------------------------------
    // Ponte di reset locale (vedi intestazione)
    // ------------------------------------------------------------------
    (* keep = "true" *) reg [1:0] rst_bridge;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) rst_bridge <= 2'b00;
        else        rst_bridge <= {rst_bridge[0], 1'b1};
    end
    wire rst_local_n = rst_bridge[1];

    // ------------------------------------------------------------------
    // ROM dei pesi: inizializzata da file, forzata su BRAM
    // ------------------------------------------------------------------
    (* rom_style = "block" *)
    reg [AXI_DW-1:0] weight_rom [0:TOTAL_BEATS-1];
    initial $readmemh(WEIGHTS_FILE, weight_rom);

    // ------------------------------------------------------------------
    // Puntatore di lettura: il prossimo beat da emettere. Tre contatori
    // allineati (indirizzo assoluto, posizione nel layer, layer) cosi'
    // tlast/layer_idx non richiedono divisioni.
    // ------------------------------------------------------------------
    //
    // 2026-10-05 (timing): il puntatore NON ha piu' il reset asincrono.
    // Le copie di rd_rom_addr create da max_fanout stanno vicino alle 176
    // BRAM; il reset doveva raggiungerle tutte (slack misurato +0.007 ns a
    // 266.7 MHz). Non serve: prima che il core chieda i pesi arriva sempre
    // un wload_start, che riallinea il puntatore. Il valore iniziale (0)
    // e' dato dall'inizializzazione dei registri alla configurazione
    // dell'FPGA (e vale anche in simulazione).
    (* max_fanout = 32 *)
    reg [ROM_ADDR_W-1:0]    rd_rom_addr      = {ROM_ADDR_W{1'b0}};      // 0 .. TOTAL_BEATS-1
    reg [BEAT_IN_LAY_W-1:0] rd_beat_in_layer = {BEAT_IN_LAY_W{1'b0}};   // 0 .. NBEAT-1
    reg [LAYER_W-1:0]       rd_layer         = {LAYER_W{1'b0}};         // 0 .. NLAYER-1

    wire rd_is_last_of_layer = (rd_beat_in_layer == NBEAT - 1);
    wire rd_is_last_of_rom   = (rd_rom_addr == TOTAL_BEATS - 1);

    // Indirizzo della prima riga del layer richiesto dal core:
    // wload_layer * NBEAT (moltiplicazione per costante -> pochi LUT).
    wire [ROM_ADDR_W-1:0] requested_layer_base = wload_layer * NBEAT;

    // Un indice di layer fuori range (6 o 7 su 3 bit) non deve spostare il
    // puntatore fuori dalla ROM: in quel caso l'impulso viene ignorato.
    wire resync_to_layer = wload_start && (wload_layer < NLAYER);

    // ------------------------------------------------------------------
    // Coda d'uscita
    // ------------------------------------------------------------------
    reg [AXI_DW-1:0]     fifo_data  [0:FIFO_DEPTH-1];   // beat
    reg                  fifo_last  [0:FIFO_DEPTH-1];   // ultimo beat del suo layer
    reg [LAYER_W-1:0]    fifo_layer [0:FIFO_DEPTH-1];   // layer del beat
    reg [FIFO_IDX_W-1:0] fifo_wr_idx, fifo_rd_idx;      // puntatori della coda
    reg [COUNT_W-1:0]    fifo_count;                    // parole in coda (0..8)
    reg [COUNT_W-1:0]    read_credits;                  // letture in volo + parole in coda (0..8)

    wire out_valid     = (fifo_count != {COUNT_W{1'b0}});   // c'e' un beat in testa
    wire out_handshake = out_valid && m_axis_tready;        // il core lo prende ora

    // si emette una lettura solo con posto garantito in coda e mai nel
    // ciclo del riallineamento (il puntatore sta cambiando)
    wire issue_read = (read_credits < FIFO_DEPTH) && !resync_to_layer;

    // ------------------------------------------------------------------
    // Lettura della ROM: 2 stadi registrati, SEMPRE abilitati, niente reset
    //   rom_q1: registro di lettura della BRAM (indirizzo = rd_rom_addr)
    //   rom_q2: registro d'uscita della BRAM (DOB_REG)
    // ------------------------------------------------------------------
    reg [AXI_DW-1:0] rom_q1, rom_q2;
    always @(posedge clk) begin
        rom_q1 <= weight_rom[rd_rom_addr];
        rom_q2 <= rom_q1;
    end

    // etichette che viaggiano insieme al dato nei 2 stadi
    reg               read_valid_1, read_valid_2;     // il dato nello stadio e' una lettura emessa
    reg               read_last_1,  read_last_2;      // ultimo beat del layer
    reg [LAYER_W-1:0] read_layer_1, read_layer_2;     // layer del beat

    wire fifo_push = read_valid_2 && !resync_to_layer;   // il dato di rom_q2 entra in coda

    // ------------------------------------------------------------------
    // Controllo: puntatore, stadi di lettura, coda, crediti
    // ------------------------------------------------------------------
    always @(posedge clk or negedge rst_local_n) begin
        if (!rst_local_n) begin
            read_valid_1     <= 1'b0;
            read_valid_2     <= 1'b0;
            read_last_1      <= 1'b0;
            read_last_2      <= 1'b0;
            read_layer_1     <= {LAYER_W{1'b0}};
            read_layer_2     <= {LAYER_W{1'b0}};
            fifo_wr_idx      <= {FIFO_IDX_W{1'b0}};
            fifo_rd_idx      <= {FIFO_IDX_W{1'b0}};
            fifo_count       <= {COUNT_W{1'b0}};
            read_credits     <= {COUNT_W{1'b0}};
        end else if (resync_to_layer) begin
            // Riallineamento: puntatore alla prima riga del layer richiesto,
            // letture in volo scartate, coda svuotata (il core in questo
            // ciclo ha tready=0, quindi non sta prendendo nessun beat).
            // Il puntatore si riallinea nel blocco senza reset, piu' sotto.
            read_valid_1     <= 1'b0;
            read_valid_2     <= 1'b0;
            fifo_wr_idx      <= {FIFO_IDX_W{1'b0}};
            fifo_rd_idx      <= {FIFO_IDX_W{1'b0}};
            fifo_count       <= {COUNT_W{1'b0}};
            read_credits     <= {COUNT_W{1'b0}};
        end else begin

            // ---- etichette lungo i 2 stadi di lettura ----
            read_valid_1 <= issue_read;
            read_last_1  <= rd_is_last_of_layer;
            read_layer_1 <= rd_layer;
            read_valid_2 <= read_valid_1;
            read_last_2  <= read_last_1;
            read_layer_2 <= read_layer_1;

            // ---- coda: inserimento (dal secondo stadio) ed estrazione (dal core) ----
            if (fifo_push)     fifo_wr_idx <= fifo_wr_idx + 1'b1;
            if (out_handshake) fifo_rd_idx <= fifo_rd_idx + 1'b1;

            fifo_count   <= fifo_count   + (fifo_push  ? 1'b1 : 1'b0) - (out_handshake ? 1'b1 : 1'b0);
            read_credits <= read_credits + (issue_read ? 1'b1 : 1'b0) - (out_handshake ? 1'b1 : 1'b0);
        end
    end

    // ------------------------------------------------------------------
    // Puntatore di lettura (SENZA reset, vedi dichiarazione):
    //   riallineamento -> prima riga del layer richiesto;
    //   emissione      -> beat successivo, con ritorno a 0 dopo l'ultimo.
    // ------------------------------------------------------------------
    always @(posedge clk) begin
        if (resync_to_layer) begin
            rd_rom_addr      <= requested_layer_base;
            rd_beat_in_layer <= {BEAT_IN_LAY_W{1'b0}};
            rd_layer         <= wload_layer;
        end else if (issue_read) begin
            rd_rom_addr <= rd_is_last_of_rom ? {ROM_ADDR_W{1'b0}}
                                             : rd_rom_addr + 1'b1;
            if (rd_is_last_of_layer) begin
                // fine layer: posizione a 0, layer successivo (5 -> 0)
                rd_beat_in_layer <= {BEAT_IN_LAY_W{1'b0}};
                rd_layer         <= (rd_layer == NLAYER - 1) ? {LAYER_W{1'b0}}
                                                             : rd_layer + 1'b1;
            end else begin
                rd_beat_in_layer <= rd_beat_in_layer + 1'b1;
            end
        end
    end

    // dati della coda (niente reset: contano solo con fifo_count)
    always @(posedge clk)
        if (fifo_push) begin
            fifo_data[fifo_wr_idx]  <= rom_q2;
            fifo_last[fifo_wr_idx]  <= read_last_2;
            fifo_layer[fifo_wr_idx] <= read_layer_2;
        end

    // ------------------------------------------------------------------
    // Uscite: testa della coda
    // ------------------------------------------------------------------
    assign m_axis_tvalid = out_valid;
    assign m_axis_tdata  = fifo_data[fifo_rd_idx];
    assign m_axis_tlast  = out_valid && fifo_last[fifo_rd_idx];

    // impulso quando il core prende l'ultimo beat di un layer
    assign layer_done = out_handshake && fifo_last[fifo_rd_idx];
    assign layer_idx  = fifo_layer[fifo_rd_idx];

endmodule

`default_nettype wire
