`timescale 1ns / 1ps
`default_nettype none
//=====================================================================
// s4d_stim_src -- sorgente AXI-Stream (master) di stimoli per l'encoder
//                 (prima ecg_stim_src / data_injector_sim.v, solo ECG)
//
//   Manda a s4d_encoder_axis una sequenza fissa di REC_COUNT record
//   (sequenze di ingresso) letti da un file .mem, in loop continuo.
//   Serve in simulazione al posto della DMA, ma e' anche sintetizzabile:
//   la memoria diventa una ROM in BRAM inizializzata dal .mem, utile per
//   provare il core su scheda senza PS ne' DMA.
//
//   DATASET sceglie i valori di default dei parametri (tutti comunque
//   sovrascrivibili uno per uno, per esempio TIMESTEPS_PER_RECORD):
//
//     DATASET    NIN  SAMPLE_WIDTH        AXIS_DATA_WIDTH  TIMESTEPS  parole/record
//     "ECG"       12  16 (Q5.11, segno)   64               1024       3072
//     "SP"        12  16 (Q5.11, segno)   64               4096       12288
//     "sCIFAR"     1   8 (pixel 0..255)   32               1024       256
//     "RadioML"    2  16 (I, Q)           64               1024       512
//
//   ("SP" = Speech Commands, come le cartelle mem_SP. ECG, SP, sCIFAR:
//   formati di s4d_encoder_axis e dei .mem in mem_*; SP con L = 4096 dal commento storico di s4d_core_axis. RadioML NON ha
//   ancora encoder ne' pesi nel progetto: 2 ingressi I/Q a 16 bit con
//   L = 1024 (RadioML 2018.01A) e' un'ipotesi da confermare.)
//
//   FORMATO DEL .mem (STIM_FILE_NAME, default "stim_<DATASET>.mem"):
//   una parola da AXIS_DATA_WIDTH bit per riga, in esadecimale, esattamente
//   come la vuole s4d_encoder_axis: i campioni formano un nastro continuo
//   (timestep t = NIN campioni consecutivi, c0 per primo), il primo
//   campione negli LSB della parola; un timestep puo' iniziare a meta'
//   parola. Esempi:
//     ECG/SP : 3 parole da 64 bit per timestep (4 campioni int16 a parola)
//     sCIFAR : 4 timestep (pixel uint8) per parola da 32 bit
//     RadioML: 2 timestep {Q, I} per parola da 64 bit (I negli LSB)
//   Ogni record inizia a inizio parola e occupa
//     WORDS_PER_RECORD = ceil(TIMESTEPS * NIN * SAMPLE_WIDTH / AXIS_DATA_WIDTH)
//   parole; i bit in piu' dell'ultima parola sono riempimento (l'encoder
//   li scarta). tlast va sull'ultima parola di ogni record.
//   Dopo REC_COUNT record il puntatore torna a 0: loop continuo.
//
// Handshake (2026-10-20): tvalid non dipende da tready, come richiede
// AXI-Stream. Il registro d'uscita si ricarica quando e' vuoto (!tvalid)
// o quando la parola corrente viene accettata (tready); con tready fermo
// la parola resta stabile. Una parola per ciclo.
//=====================================================================
module s4d_stim_src #(
    parameter         DATASET              = "ECG",  // "ECG" | "SP" | "sCIFAR" | "RadioML"

    // ---- formato dei record (default dal DATASET) ----
    parameter integer NIN                  = (DATASET == "sCIFAR")  ? 1  :
                                             (DATASET == "RadioML") ? 2  : 12,     // campioni per timestep
    parameter integer SAMPLE_WIDTH         = (DATASET == "sCIFAR")  ? 8  : 16,     // bit di un campione
    parameter integer AXIS_DATA_WIDTH      = (DATASET == "sCIFAR")  ? 32 : 64,     // = AXI_DW dell'encoder
    parameter integer TIMESTEPS_PER_RECORD = (DATASET == "SP") ? 4096 : 1024,  // L
    parameter integer REC_COUNT            = 10,     // record contenuti nel file .mem

    // parole per record (ultima eventualmente con riempimento) e nell'intero file
    parameter integer WORDS_PER_RECORD     = (TIMESTEPS_PER_RECORD * NIN * SAMPLE_WIDTH
                                              + AXIS_DATA_WIDTH - 1) / AXIS_DATA_WIDTH,
    parameter integer TOTAL_WORDS          = WORDS_PER_RECORD * REC_COUNT,

    parameter         STIM_FILE_NAME       = {"stim_", DATASET, ".mem"}
)(
    input  wire                                aclk,
    input  wire                                aresetn,   // reset attivo basso

    // porta master: noi guidiamo tdata/tvalid/tlast, l'encoder risponde con tready
    output reg                                 m_axis_tvalid,
    input  wire                                m_axis_tready,
    output reg  [AXIS_DATA_WIDTH-1:0]          m_axis_tdata,
    output reg                                 m_axis_tlast
);

    // controllo dei parametri (in simulazione)
    initial begin
        if (!(DATASET == "ECG" || DATASET == "SP" || DATASET == "sCIFAR" || DATASET == "RadioML")) begin
            $display("s4d_stim_src: DATASET \"%0s\" non riconosciuto", DATASET);
            $finish;
        end
        if (AXIS_DATA_WIDTH % SAMPLE_WIDTH != 0) begin
            $display("s4d_stim_src: AXIS_DATA_WIDTH=%0d non multiplo di SAMPLE_WIDTH=%0d",
                     AXIS_DATA_WIDTH, SAMPLE_WIDTH);
            $finish;
        end
    end

    // memoria che contiene l'intero .mem (ROM in BRAM se sintetizzata)
    (* rom_style = "block" *)
    reg [AXIS_DATA_WIDTH-1:0] stim_data [0:TOTAL_WORDS-1];

    initial $readmemh(STIM_FILE_NAME, stim_data);

    // word_ptr: indice assoluto della prossima parola da leggere,
    // gira su tutto il file (0..TOTAL_WORDS-1) e poi ricomincia da 0 (loop)
    reg [31:0] word_ptr;

    // rec_word_idx: posizione della parola DENTRO al record corrente
    // (0..WORDS_PER_RECORD-1); serve solo a sapere quando alzare tlast
    reg [31:0] rec_word_idx;

    // vera quando la parola che stiamo per caricare e' l'ultima del record corrente
    wire is_last_word_of_record = (rec_word_idx == WORDS_PER_RECORD - 1);

    // il registro d'uscita si ricarica se e' vuoto o se la parola corrente
    // viene accettata in questo ciclo (tvalid && tready)
    wire load = !m_axis_tvalid || m_axis_tready;

    // controllo: con reset
    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn) begin
            word_ptr      <= 32'd0;
            rec_word_idx  <= 32'd0;
            m_axis_tvalid <= 1'b0;
            m_axis_tlast  <= 1'b0;
        end else if (load) begin
            m_axis_tvalid <= 1'b1;
            m_axis_tlast  <= is_last_word_of_record;

            // avanza il puntatore assoluto, con wrap a fine file (loop dei record)
            word_ptr <= (word_ptr == TOTAL_WORDS - 1) ? 32'd0 : word_ptr + 32'd1;

            // avanza la posizione nel record, con wrap a fine record (per il prossimo tlast)
            rec_word_idx <= is_last_word_of_record ? 32'd0 : rec_word_idx + 32'd1;
        end
    end

    // dato: niente reset (conta solo con tvalid), cosi' la lettura con
    // enable resta dentro la BRAM (registro di lettura della ROM)
    always @(posedge aclk)
        if (load) m_axis_tdata <= stim_data[word_ptr];

endmodule

`default_nettype wire
