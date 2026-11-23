`timescale 1ns / 1ps
`default_nettype none
//=====================================================================
// s4d_biquad_bank  --  convoluzione S4D di un frame (H canali)
//
//   Per ogni canale ch:
//     y[ch] = D[ch]*u[ch] + sum_{k<MODES} biquad_k(u[ch])
//   dove ogni biquad_k e' una coppia di poli complessi coniugati del
//   kernel S4D realizzata come filtro IIR del 2o ordine con stato.
//
//   Organizzazione:
//     - NUNITS unita' (s4d_biquad_unit) lavorano in parallelo e in
//       lock-step; l'unita' j serve i canali j*NCH .. j*NCH+NCH-1.
//     - Ogni unita' scorre NB = NCH*MODES biquad, uno per ciclo:
//       con NUNITS=32, NCH=4 -> NB = 128, una passata (un frame) dura
//       ~128 cicli + latenza (con NUNITS=8, NCH=16 erano 512).
//     - Ingresso a doppio buffer (ubuf): mentre le unita' elaborano un
//       frame, il successivo viene raccolto nell'altra meta'.
//     - I risultati (uno per canale) vengono raccolti in res; a fine
//       passata res viene copiato nel registro a scorrimento out_shift
//       ed emesso un canale per ciclo, con arrotondamento (F_Y -> F_OUT)
//       e saturazione a W_OUT bit. L'emissione si sovrappone alla
//       passata successiva (res_pending impedisce di sovrascrivere res
//       prima della copia). Copia della versione precedente: biquad_bank.v.bak
//
//   Reset dello stato: il primo frame di ogni nuova sequenza parte con
//   lo stato dei biquad a zero (clr), vedi clr_next / buf_lastseq.
//=====================================================================

module s4d_biquad_bank #(
    parameter integer H        = 128,   // canali
    parameter integer LOG2_H   = 7,
    parameter integer NUNITS   = 8,     // unita' in parallelo
    parameter integer LOG2_NU  = 3,
    parameter integer MODES    = 32,    // biquad (modi) per canale
    parameter integer LOG2_MOD = 5,
    parameter integer NCH      = 16,    // canali per unita' (NUNITS*NCH = H)
    parameter integer LOG2_NCH = 4,

    parameter integer LOG2_NB  = LOG2_MOD + LOG2_NCH,   // bit di indirizzo nell'unita'
    parameter integer W_U      = 16,    // ingresso u
    parameter integer F_U      = 11,
    parameter integer W_A      = 24,    // coefficienti a1,a2
    parameter integer F_A      = 22,
    parameter integer W_B      = 16,    // coefficienti b0,b1
    parameter integer F_B      = 15,
    parameter integer W_S      = 32,    // stato / uscita di un biquad
    parameter integer F_Y      = 19,
    parameter integer W_D      = 16,    // coefficiente D
    parameter integer F_D      = 14,

    parameter integer W_BSH    = 5,     // shift dei coefficienti b
    parameter integer W_ACC    = 40,    // accumulatore per canale (Q.F_Y)
    parameter integer W_OUT    = 16,    // uscita del banco
    parameter integer F_OUT    = 11
)(
    input  wire                    clk,
    input  wire                    rst_n,

    // ---- scrittura coefficienti {b1,b0,a2,a1} nell'unita' cfg_unit ----
    input  wire                    cfg_we,
    input  wire [LOG2_NU-1:0]      cfg_unit,
    input  wire [LOG2_NB-1:0]      cfg_addr,
    input  wire [2*W_A+2*W_B-1:0]  cfg_data,

    // ---- scrittura D e shift dei b del canale cfgd_ch ----
    input  wire                    cfgd_we,
    input  wire [LOG2_H-1:0]       cfgd_ch,
    input  wire signed [W_D-1:0]   cfgd_data,
    input  wire [W_BSH-1:0]        cfgd_bsh,

    // ---- ingresso: H campioni per frame ----
    input  wire                    in_valid,
    output wire                    in_ready,
    input  wire signed [W_U-1:0]   in_data,
    input  wire                    in_last_seq,  // il frame e' l'ultimo della sequenza

    // ---- uscita: H campioni per frame ----
    output reg                     out_valid,
    input  wire                    out_ready,
    output reg  signed [W_OUT-1:0] out_data,
    output reg  [LOG2_H-1:0]       out_ch,       // canale del campione in uscita
    output reg                     out_last,     // ultimo canale del frame

    output wire                    ovf           // overflow in almeno un'unita' (sticky)
);

    localparam integer SH_O = F_Y - F_OUT;   // shift di uscita: Q.F_Y -> Q.F_OUT

    // limiti di saturazione a W_OUT bit, estesi a W_ACC
    localparam signed [W_ACC-1:0] OMAX = {{(W_ACC-W_OUT+1){1'b0}}, {(W_OUT-1){1'b1}}};
    localparam signed [W_ACC-1:0] OMIN = {{(W_ACC-W_OUT+1){1'b1}}, {(W_OUT-1){1'b0}}};

    //-----------------------------------------------------------------
    // Doppio buffer d'ingresso: 2 frame da H campioni (in registri)
    //   meta' wsel : in scrittura
    //   meta' rsel : in lettura da parte delle unita'
    //-----------------------------------------------------------------
    reg [2*H*W_U-1:0] ubuf;
    reg               wsel, rsel;    // meta' in scrittura / in lettura
    reg [LOG2_H-1:0]  wcnt;          // canale in scrittura
    reg               buf_full;      // 1 = un frame completo aspetta di essere elaborato

    // ingresso bloccato finche' un frame completo e' in attesa
    assign in_ready = !buf_full;

    reg                     start;        // impulso: le unita' iniziano una passata

    reg                     clr;          // impulso: la passata parte con stato azzerato
    reg                     clr_next;     // la PROSSIMA passata dovra' azzerare lo stato
    reg                     buf_lastseq;  // il frame raccolto e' l'ultimo della sequenza
    wire [NUNITS-1:0]       u_busy;       // unita' ancora occupate
    wire [NUNITS-1:0]       u_ovalid;     // risultato di un canale pronto (tutte insieme)
    wire [NUNITS*W_ACC-1:0] u_oacc;       // risultati delle unita'
    wire [NUNITS-1:0]       u_ovf;

    assign ovf = |u_ovf;

    //=================================================================
    // Distribuzione a ALBERO dei segnali comuni a tutte le unita'
    // (timing 2026-10-02, seconda iterazione).
    //   livello 1: NGROUPS = NUNITS/UNITS_PER_GROUP copie (una ogni 8 unita')
    //   livello 2: una copia per unita' (dentro g_unit, piu' sotto)
    // Misurato a 177.8 MHz con la sola copia per unita': il tratto
    // s4d_wload -> 32 unita' restava ~5 ns di solo filo (slack 0.128 ns),
    // e anche clr -> clr_r delle unita' (0.41 ns). Con il livello
    // intermedio ogni tratto copre circa meta' della distanza.
    // Ritardo: +1 ciclo sulle scritture dei coefficienti (ininfluente) e
    // +2 cicli su start/clr, uguali per tutte le unita' (lock-step
    // preservato; il controllo usa pass_active, quindi non riparte prima).
    // dont_touch: impedisce alla sintesi di riunire le copie.
    //=================================================================
    localparam integer UNITS_PER_GROUP = 8;
    localparam integer NGROUPS         = (NUNITS + UNITS_PER_GROUP - 1) / UNITS_PER_GROUP;

    (* dont_touch = "true" *) reg                     grp_cfg_we   [0:NGROUPS-1];
    (* dont_touch = "true" *) reg [LOG2_NU-1:0]       grp_cfg_unit [0:NGROUPS-1];
    (* dont_touch = "true" *) reg [LOG2_NB-1:0]       grp_cfg_addr [0:NGROUPS-1];
    (* dont_touch = "true" *) reg [2*W_A+2*W_B-1:0]   grp_cfg_data [0:NGROUPS-1];
    (* dont_touch = "true" *) reg                     grp_cfgd_we  [0:NGROUPS-1];
    (* dont_touch = "true" *) reg [LOG2_H-1:0]        grp_cfgd_ch  [0:NGROUPS-1];
    (* dont_touch = "true" *) reg signed [W_D-1:0]    grp_cfgd_data[0:NGROUPS-1];
    (* dont_touch = "true" *) reg [W_BSH-1:0]         grp_cfgd_bsh [0:NGROUPS-1];
    (* dont_touch = "true" *) reg                     grp_start    [0:NGROUPS-1];
    (* dont_touch = "true" *) reg                     grp_clr      [0:NGROUPS-1];

    integer grp_idx;
    // abilitazioni e impulsi di controllo: con reset (nessun evento spurio)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (grp_idx = 0; grp_idx < NGROUPS; grp_idx = grp_idx + 1) begin
                grp_cfg_we[grp_idx]  <= 1'b0;
                grp_cfgd_we[grp_idx] <= 1'b0;
                grp_start[grp_idx]   <= 1'b0;
                grp_clr[grp_idx]     <= 1'b0;
            end
        end else begin
            for (grp_idx = 0; grp_idx < NGROUPS; grp_idx = grp_idx + 1) begin
                grp_cfg_we[grp_idx]  <= cfg_we;
                grp_cfgd_we[grp_idx] <= cfgd_we;
                grp_start[grp_idx]   <= start;
                grp_clr[grp_idx]     <= clr;
            end
        end
    end
    // indirizzi e dati: niente reset (contano solo con l'abilitazione)
    always @(posedge clk) begin
        for (grp_idx = 0; grp_idx < NGROUPS; grp_idx = grp_idx + 1) begin
            grp_cfg_unit[grp_idx]  <= cfg_unit;
            grp_cfg_addr[grp_idx]  <= cfg_addr;
            grp_cfg_data[grp_idx]  <= cfg_data;
            grp_cfgd_ch[grp_idx]   <= cfgd_ch;
            grp_cfgd_data[grp_idx] <= cfgd_data;
            grp_cfgd_bsh[grp_idx]  <= cfgd_bsh;
        end
    end

    //=================================================================
    // NUNITS unita' biquad
    //=================================================================
    genvar j;
    generate
        for (j = 0; j < NUNITS; j = j + 1) begin : g_unit

            localparam integer MY_GROUP = j / UNITS_PER_GROUP;   // gruppo dell'albero di questa unita'

            // i NCH campioni del frame in lettura che spettano all'unita' j
            wire [NCH*W_U-1:0] u_slice =
                ubuf[(rsel*H + j*NCH)*W_U +: NCH*W_U];

            //---------------------------------------------------------
            // Copia LOCALE (livello 2 dell'albero) del bus di
            // configurazione e di start/clr, presa dalla copia del gruppo.
            // Prima cfg_* / cfgd_* andavano direttamente alle LUTRAM di
            // tutte le 32 unita' (fan-out fino a 4032, slack 0.20-0.49 ns
            // a 150 MHz). Qui l'abilitazione viene gia' decodificata.
            // Costo totale: le scritture dei coefficienti arrivano 2 cicli
            // dopo (ininfluente: il banco li usa solo a passata avviata).
            //---------------------------------------------------------
            (* dont_touch = "true" *) reg                      unit_cfg_we_q;     // scrivi coefficienti in questa unita'
            (* dont_touch = "true" *) reg [LOG2_NB-1:0]        unit_cfg_addr_q;   // canale locale*MODES + modo
            (* dont_touch = "true" *) reg [2*W_A+2*W_B-1:0]    unit_cfg_data_q;   // {b1, b0, a2, a1}
            (* dont_touch = "true" *) reg                      unit_cfgd_we_q;    // scrivi D/shift in questa unita'
            (* dont_touch = "true" *) reg [LOG2_NCH-1:0]       unit_cfgd_addr_q;  // canale locale
            (* dont_touch = "true" *) reg signed [W_D-1:0]     unit_cfgd_data_q;  // D
            (* dont_touch = "true" *) reg [W_BSH-1:0]          unit_cfgd_bsh_q;   // shift dei b
            (* dont_touch = "true" *) reg                      unit_start_q;      // impulso di avvio passata
            (* dont_touch = "true" *) reg                      unit_clr_q;        // passata con stato azzerato

            // abilitazioni con reset: dopo il reset nessuna scrittura spuria
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    unit_cfg_we_q  <= 1'b0;
                    unit_cfgd_we_q <= 1'b0;
                    unit_start_q   <= 1'b0;
                    unit_clr_q     <= 1'b0;
                end else begin
                    // coefficienti: solo l'unita' selezionata scrive
                    unit_cfg_we_q  <= grp_cfg_we[MY_GROUP]
                                   && (grp_cfg_unit[MY_GROUP] == j[LOG2_NU-1:0]);
                    // D e shift: bit alti del canale = unita', bit bassi = canale locale
                    unit_cfgd_we_q <= grp_cfgd_we[MY_GROUP]
                                   && (grp_cfgd_ch[MY_GROUP][LOG2_H-1:LOG2_NCH] == j[LOG2_NU-1:0]);
                    unit_start_q   <= grp_start[MY_GROUP];
                    unit_clr_q     <= grp_clr[MY_GROUP];
                end
            end

            // indirizzi e dati: niente reset (contano solo con l'abilitazione)
            always @(posedge clk) begin
                unit_cfg_addr_q  <= grp_cfg_addr[MY_GROUP];
                unit_cfg_data_q  <= grp_cfg_data[MY_GROUP];
                unit_cfgd_addr_q <= grp_cfgd_ch[MY_GROUP][LOG2_NCH-1:0];
                unit_cfgd_data_q <= grp_cfgd_data[MY_GROUP];
                unit_cfgd_bsh_q  <= grp_cfgd_bsh[MY_GROUP];
            end

            s4d_biquad_unit #(
                .NCH(NCH), .MODES(MODES),
                .LOG2_NB(LOG2_NB), .LOG2_MOD(LOG2_MOD), .LOG2_CH(LOG2_NCH),
                .W_U(W_U), .F_U(F_U), .W_A(W_A), .F_A(F_A),
                .W_B(W_B), .F_B(F_B), .W_S(W_S), .F_Y(F_Y),
                .W_D(W_D), .F_D(F_D), .W_BSH(W_BSH), .W_ACC(W_ACC)
            ) u_bq (
                .clk       (clk),
                .rst_n     (rst_n),
                // configurazione dalla copia locale (vedi sopra)
                .cfg_we    (unit_cfg_we_q),
                .cfg_addr  (unit_cfg_addr_q),
                .cfg_data  (unit_cfg_data_q),
                .cfgd_we   (unit_cfgd_we_q),
                .cfgd_addr (unit_cfgd_addr_q),
                .cfgd_data (unit_cfgd_data_q),
                .cfgd_bsh  (unit_cfgd_bsh_q),
                .start     (unit_start_q),       // copia locale (+2 cicli, uguale per tutte)
                .clr       (unit_clr_q),
                .u_vec     (u_slice),
                .busy      (u_busy[j]),
                .out_valid (u_ovalid[j]),
                .out_acc   (u_oacc[j*W_ACC +: W_ACC]),
                .ovf       (u_ovf[j])
            );
        end
    endgenerate

    //-----------------------------------------------------------------
    // Raccolta dei risultati (invariata): a ogni res_ev ogni unita'
    // consegna il risultato del suo canale locale slot_cnt
    // -> canale jj*NCH + slot_cnt
    //-----------------------------------------------------------------
    reg [H*W_ACC-1:0]  res;            // un accumulatore per canale
    reg [LOG2_NCH-1:0] slot_cnt;       // canale locale del prossimo risultato

    wire res_ev    = u_ovalid[0];                           // unita' in lock-step: basta la 0
    wire pass_done = res_ev && (slot_cnt == NCH-1);         // arrivato l'ultimo canale

    integer jj;
    always @(posedge clk) begin
        if (res_ev)
            for (jj = 0; jj < NUNITS; jj = jj + 1)
                res[(jj*NCH + slot_cnt)*W_ACC +: W_ACC] <=
                    u_oacc[jj*W_ACC +: W_ACC];
    end

    reg pass_active;                   // 1 = le unita' stanno elaborando un frame
    reg res_pending;                   // 1 = res contiene un frame completo non ancora copiato

    //-----------------------------------------------------------------
    // Registro di EMISSIONE a scorrimento: copia di res, emessa un
    // canale per ciclo leggendo sempre la testa (indice 0).
    // Permette alla passata successiva di partire mentre si emette, e
    // toglie il multiplexer 128:1 da 40 bit (era res[ocnt]).
    //-----------------------------------------------------------------
    reg signed [W_ACC-1:0] out_shift [0:H-1];
    reg                    stream_active;   // 1 = out_shift contiene canali da emettere
    reg [LOG2_H-1:0]       stream_cnt;      // canale in testa (0..H-1)

    // handshake del registro d'uscita
    wire out_free  = !out_valid || out_ready;                  // il registro d'uscita puo' caricare
    wire beat      = stream_active && out_free;                // si emette un canale
    wire last_beat = beat && (stream_cnt == H-1);              // si emette l'ultimo canale

    // copia res -> out_shift: frame pronto e registro di emissione libero
    // (vuoto, o sta emettendo ORA il suo ultimo canale)
    wire res_copy  = res_pending && (!stream_active || last_beat);

    // dati di out_shift: carico parallelo oppure scorrimento (niente reset)
    integer shift_idx;
    always @(posedge clk) begin
        if (res_copy) begin
            for (shift_idx = 0; shift_idx < H; shift_idx = shift_idx + 1)
                out_shift[shift_idx] <= $signed(res[shift_idx*W_ACC +: W_ACC]);
        end else if (beat) begin
            for (shift_idx = 0; shift_idx < H-1; shift_idx = shift_idx + 1)
                out_shift[shift_idx] <= out_shift[shift_idx+1];
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wsel          <= 1'b0;
            rsel          <= 1'b0;
            wcnt          <= {LOG2_H{1'b0}};
            buf_full      <= 1'b0;
            start         <= 1'b0;
            pass_active   <= 1'b0;
            res_pending   <= 1'b0;
            stream_active <= 1'b0;
            stream_cnt    <= {LOG2_H{1'b0}};
            slot_cnt      <= {LOG2_NCH{1'b0}};
            clr           <= 1'b0;
            clr_next      <= 1'b1;        // la prima sequenza parte da stato nullo
            buf_lastseq   <= 1'b0;
        end else begin
            start <= 1'b0;
            clr   <= 1'b0;

            // ---- raccolta di un campione nella meta' in scrittura (invariata) ----
            if (in_valid && in_ready) begin
                ubuf[(wsel*H + wcnt)*W_U +: W_U] <= in_data;
                if (wcnt == H-1) begin
                    // frame completo: si passa all'altra meta'
                    wcnt        <= {LOG2_H{1'b0}};
                    buf_full    <= 1'b1;
                    wsel        <= ~wsel;
                    buf_lastseq <= in_last_seq;
                end else begin
                    wcnt <= wcnt + 1'b1;
                end
            end

            // ---- avvio di una passata ----
            // frame pronto, passata precedente finita (ultimo risultato
            // arrivato, pass_active = 0) e res gia' copiato (cosi' la
            // nuova passata non sovrascrive risultati non ancora emessi).
            // NON aspetta piu' la fine dell'emissione (!stream).
            // 2026-10-15: non aspetta piu' nemmeno !(|u_busy), cioe' lo
            // svuotamento della coda della pipeline delle unita' (stadi
            // S4, 8 cicli dopo l'ultimo risultato). Quella coda scrive solo
            // lo stato dei biquad: il biquad k viene scritto al piu' tardi
            // 15 cicli dopo la sua lettura, mentre la passata successiva lo
            // rilegge almeno ~135 cicli dopo, quindi non c'e' conflitto.
            // Cosi' la pipeline piu' lunga non allunga il frame. u_busy
            // resta collegato solo per diagnostica.
            if (buf_full && !pass_active && !res_pending) begin
                start       <= 1'b1;
                clr         <= clr_next;       // azzera lo stato se inizia una nuova sequenza
                clr_next    <= buf_lastseq;    // dopo l'ultimo frame, il prossimo azzerera'
                rsel        <= ~wsel;          // si legge la meta' appena riempita
                buf_full    <= 1'b0;
                pass_active <= 1'b1;
                slot_cnt    <= {LOG2_NCH{1'b0}};
            end

            if (res_ev) slot_cnt <= slot_cnt + 1'b1;

            // ---- fine passata: res completo, in attesa di copia ----
            // (pass_done e res_copy non possono coincidere: la passata e'
            // partita con res_pending = 0)
            if (pass_done) begin
                pass_active <= 1'b0;
                res_pending <= 1'b1;
            end else if (res_copy) begin
                res_pending <= 1'b0;
            end

            // ---- emissione ----
            if (res_copy) begin
                // nuovo frame (eventualmente nello stesso ciclo dell'ultimo canale del precedente)
                stream_active <= 1'b1;
                stream_cnt    <= {LOG2_H{1'b0}};
            end else if (last_beat) begin
                stream_active <= 1'b0;
                stream_cnt    <= {LOG2_H{1'b0}};
            end else if (beat) begin
                stream_cnt <= stream_cnt + 1'b1;
            end
        end
    end

    //-----------------------------------------------------------------
    // Arrotondamento (round-half-up), shift e saturazione del canale in
    // testa: out = sat( (acc + rnd) >>> SH_O ).
    // Timing (2026-10-15): prima somma e confronti erano in serie nello
    // stesso ciclo. Ora i confronti si fanno su acc, IN PARALLELO alla
    // somma, contro soglie costanti equivalenti:
    //   (acc + rnd) >>> SH_O >  OMAX  <=>  acc >= (OMAX+1)*2^SH_O - rnd
    //   (acc + rnd) >>> SH_O <  OMIN  <=>  acc <  OMIN*2^SH_O     - rnd
    // Esatto finche' acc + rnd non trabocca W_ACC bit: |acc| <= (MODES+1)
    // * 2^(W_S-1) (D*u e MODES valori y saturati a W_S bit), controllato
    // sotto in elaborazione. Stessi risultati della versione precedente.
    //-----------------------------------------------------------------
    wire signed [W_ACC-1:0] acc_now = out_shift[0];

    localparam signed [W_ACC-1:0] RND_O  = {{(W_ACC-1){1'b0}}, 1'b1} <<< (SH_O-1);   // mezzo LSB d'uscita
    localparam signed [W_ACC-1:0] THR_HI = ((OMAX + 1) <<< SH_O) - RND_O;   // da qui in su: OMAX
    localparam signed [W_ACC-1:0] THR_LO = (OMIN <<< SH_O) - RND_O;         // sotto: OMIN

    initial if (W_ACC < W_S + $clog2(MODES + 1) + 1) begin
        $display("s4d_biquad_bank: W_ACC=%0d troppo stretto per la saturazione a soglie", W_ACC);
        $finish;
    end

    wire signed [W_ACC-1:0] yrnd   = (acc_now + RND_O) >>> SH_O;   // somma (lo shift sono fili)
    wire                    sat_hi = (acc_now >= THR_HI);          // confronti in parallelo
    wire                    sat_lo = (acc_now <  THR_LO);

    // registro d'uscita con saturazione a W_OUT bit
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            out_data  <= {W_OUT{1'b0}};
            out_ch    <= {LOG2_H{1'b0}};
            out_last  <= 1'b0;
        end else if (out_free) begin
            out_valid <= beat;
            out_ch    <= stream_cnt;
            out_last  <= last_beat;

            if      (sat_hi) out_data <= OMAX[W_OUT-1:0];
            else if (sat_lo) out_data <= OMIN[W_OUT-1:0];
            else             out_data <= yrnd[W_OUT-1:0];
        end
    end

endmodule

`default_nettype wire
