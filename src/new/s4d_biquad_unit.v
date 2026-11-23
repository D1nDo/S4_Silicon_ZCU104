`timescale 1ns / 1ps
`default_nettype none
//=====================================================================
// s4d_biquad_unit  --  unita' biquad (NCH canali x MODES modi)
//
//   Ogni "modo" S4D (coppia di poli complessi coniugati) e' realizzato
//   come filtro IIR del 2o ordine in forma diretta II trasposta:
//
//       H(z) = (b0 + b1 z^-1) / (1 + a1 z^-1 + a2 z^-2)
//
//       y[n]  = b0*u[n]           + s1[n-1]
//       s1[n] = b1*u[n] - a1*y[n] + s2[n-1]
//       s2[n] =         - a2*y[n]
//
//   L'unita' contiene NB = NCH*MODES biquad e li elabora UNO PER CICLO
//   (indirizzo cnt = canale*MODES + modo). Per ogni canale l'uscita e'
//
//       out[ch] = D[ch]*u[ch] + sum_{modi} y_modo
//
//   e viene emessa (out_valid) quando si elabora l'ultimo modo del canale.
//   Lo stato {s2, s1} di ogni biquad e' in memoria: viene letto quando
//   tocca a quel biquad e riscritto 5 cicli dopo; lo stesso biquad
//   viene riletto solo al frame successivo (NB cicli dopo), quindi non
//   c'e' conflitto tra lettura e scrittura.
//
//   Shift dei b (bshift, per canale): fa da esponente comune dei
//   coefficienti b0,b1. Il contributo b*u viene diviso per 2^bsh
//   (con arrotondamento) prima di sommarlo allo stato in Q.F_Y.
//   Con bsh = F_B+F_U-F_Y (= 7 con i formati di default) i b sono in
//   Q.F_B "normale"; valori maggiori danno piu' precisione a b piccoli.
//
//   Pipeline (vpipe[k] = dato valido nei registri dello stadio indicato).
//   Al massimo UN'operazione complessa per ciclo (timing, 2026-10-15):
//     S1  (vpipe[0])  lettura di b0, b1, s1, u, D, bsh
//     S2  (vpipe[1])  prodotti b0*u, b1*u, D*u;  mezzo LSB 2^(bsh-1)
//     S3a (vpipe[2])  b0*u, b1*u: + mezzo LSB 2^(bsh-1);  D*u: + mezzo LSB fisso
//     S3b (vpipe[3])  b0*u, b1*u: shift variabile di bsh bit;  D*u: shift fisso
//     S3c (vpipe[4])  y = s1 + round(b0*u / 2^bsh)
//     S3d (vpipe[5])  saturazione a W_S bit di y, b1*u, D*u
//                     -> accumulo di y sul canale
//                     lettura di a1, a2 e s2 (con l'indirizzo che viaggia)
//     S4a (vpipe[6])  prodotti a1*y, a2*y (24 x 32 bit: due DSP in cascata);
//                     in parallelo c = b1*u + s2
//     S4b (vpipe[7])  ... registri di pipeline del prodotto (M/P dei DSP)
//     S4c (vpipe[8])  ...
//     S4d (vpipe[9])  a*y + mezzo LSB (riporto a Q.F_Y)
//     S4e (vpipe[10]) shift fisso (fili) e saturazione di a1*y, a2*y
//     S4f (vpipe[11]) nuovo s1 = c - a1*y;  nuovo s2 = -a2*y
//     S4g (vpipe[12]) saturazione del nuovo s1 -> scrittura dello stato
//
//   2026-10-20 (ottimizzazione, aritmetica identica bit per bit):
//     - y = (s1*2^bsh + b0*u + rnd) >>> bsh  diventa  s1 + ((b0*u + rnd) >>> bsh):
//       s1*2^bsh e' multiplo di 2^bsh ed esce intero dallo shift. Niente
//       piu' shift a sinistra di s1 (barrel shifter a PW bit) e niente
//       limite bsh <= PW-W_S (= 24), oltre il quale s1 << bsh traboccava
//       (bsh usati: ECG 9-17, sCIFAR 8-15, Speech Commands 9-22).
//     - b1*u + s2 calcolato in S4a, in parallelo ai prodotti a*y, invece
//       di trasportare b1*u e s2 separatamente fino a S4f/S4g (somme
//       intere su PW bit, senza traboccamenti: stesso risultato).
//     - a1, a2 e s2 letti dalle memorie in S3d, dove servono, invece che
//       in S1 e poi trasportati per 6-12 stadi. Lo stato e' diviso in due
//       memorie (state1 = s1, state2 = s2), ognuna con una sola porta di
//       lettura; clr_r viaggia con il biquad (clr_s*) fino alla lettura
//       di s2.
//     Pipeline da 15 a 13 stadi. Versione precedente:
//     S4D_RTL/new_pre_biquad_opt/biquad_unit.v.
//   Conflitti sullo stato: il biquad k viene letto (s1 in S1, s2 in S3d)
//   e riscritto alla fine della pipeline (S4g), cioe' DOPO entrambe le
//   letture; viene riletto solo alla passata successiva, che parte
//   almeno NB cicli dopo (start viene accettato solo con run = 0).
//   Versioni precedenti: S4D_RTL/new_pre_pipeline/biquad_unit.v,
//   biquad_unit.v.bak2 sul Desktop.
//=====================================================================

module s4d_biquad_unit #(
    parameter integer NCH       = 2,    // canali serviti dall'unita'
    parameter integer MODES     = 32,   // biquad per canale
    parameter integer LOG2_NB   = 6,    // bit di indirizzo (NCH*MODES)
    parameter integer LOG2_MOD  = 5,

    parameter integer W_U   = 16,       // ingresso u (Q.F_U)
    parameter integer F_U   = 11,
    parameter integer W_A   = 24,       // a1, a2 (Q.F_A)
    parameter integer F_A   = 22,
    parameter integer W_B   = 16,       // b0, b1 (Q.F_B, scalati da bsh)
    parameter integer F_B   = 15,
    parameter integer W_S   = 32,       // stato e uscita y (Q.F_Y)
    parameter integer F_Y   = 19,
    parameter integer W_D   = 16,       // D (Q.F_D)
    parameter integer F_D   = 14,

    parameter integer W_BSH = 5,        // shift dei b
    parameter integer LOG2_CH = 1,      // bit per indicizzare i canali dell'unita'
    parameter integer W_ACC = 40        // accumulatore del canale (Q.F_Y)
)(
    input  wire                       clk,
    input  wire                       rst_n,

    // ---- scrittura coefficienti: cfg_data = {b1, b0, a2, a1} ----
    input  wire                       cfg_we,
    input  wire [LOG2_NB-1:0]         cfg_addr,
    input  wire [2*W_A+2*W_B-1:0]     cfg_data,

    // ---- scrittura D e shift dei b del canale locale cfgd_addr ----
    input  wire                       cfgd_we,
    input  wire [LOG2_CH-1:0]         cfgd_addr,
    input  wire signed [W_D-1:0]      cfgd_data,
    input  wire [W_BSH-1:0]           cfgd_bsh,

    input  wire                       start,   // impulso: elabora un frame
    input  wire                       clr,     // con start: stato azzerato per questo frame
    input  wire [NCH*W_U-1:0]         u_vec,   // campioni u dei NCH canali (fermi per tutto il frame)
    output reg                        busy,    // 1 = elaborazione o pipeline non vuota

    output reg                        out_valid, // impulso: risultato di un canale
    output reg  signed [W_ACC-1:0]    out_acc,   // D*u + somma dei modi, Q.F_Y

    output reg                        ovf        // saturazione avvenuta (sticky)
);

    localparam integer NB   = NCH*MODES;          // biquad nell'unita'
    localparam integer PW   = W_A + W_S;          // larghezza dei prodotti / somme interne

    localparam integer SH_B0 = F_B + F_U - F_Y;   // shift "naturale" di b*u (non usato)
    localparam integer SH_A = F_A;                // a*y ha F_A+F_Y bit frazionari -> Q.F_Y
    localparam integer SH_D = F_D + F_U - F_Y;    // D*u ha F_D+F_U bit frazionari -> Q.F_Y

    // limiti di saturazione a W_S bit, estesi a PW
    localparam signed [PW-1:0] SMAX = {{(PW-W_S+1){1'b0}}, {(W_S-1){1'b1}}};
    localparam signed [PW-1:0] SMIN = {{(PW-W_S+1){1'b1}}, {(W_S-1){1'b0}}};

    //-----------------------------------------------------------------
    // Memorie per biquad (indirizzo = canale*MODES + modo), profonde
    // NB = 128 con NUNITS=32: forzate in LUTRAM (RAM128X1D, lettura e
    // scrittura a indirizzi diversi). In BRAM sarebbero ~48 BRAM36 su 32
    // unita', con il budget dello ZCU104 gia' occupato da ROM pesi (176)
    // e seqbuf (64). Stima: ~576 LUT per unita', ~18k LUT in totale.
    // ATTENZIONE: valido per NB = 128. Con NUNITS=8 (NB = 512) la LUTRAM
    // costerebbe 4 volte tanto: tornare a "block".
    // Ogni memoria ha UNA porta di lettura: coef_b e state1 lette in S1
    // (indirizzo cnt), coef_a e state2 lette in S3d (indirizzo a_s3c).
    // (prima: solo coef_a con ram_style "block", il resto scelto da Vivado)
    //-----------------------------------------------------------------
    (* ram_style = "distributed" *)
    reg [2*W_A-1:0]       coef_a [0:NB-1];   // {a2, a1}
    (* ram_style = "distributed" *)
    reg [2*W_B-1:0]       coef_b [0:NB-1];   // {b1, b0}
    (* ram_style = "distributed" *)
    reg [W_S-1:0]         state1 [0:NB-1];   // s1
    (* ram_style = "distributed" *)
    reg [W_S-1:0]         state2 [0:NB-1];   // s2

    always @(posedge clk)
        if (cfg_we) begin
            coef_a[cfg_addr] <= cfg_data[2*W_A-1:0];
            coef_b[cfg_addr] <= cfg_data[2*W_A+2*W_B-1:2*W_A];
        end

    // memorie per canale: coefficiente D e shift dei b
    reg signed [W_D-1:0] dcoef [0:NCH-1];
    reg [W_BSH-1:0]      bshift [0:NCH-1];
    always @(posedge clk)
        if (cfgd_we) begin
            dcoef[cfgd_addr]  <= cfgd_data;
            bshift[cfgd_addr] <= cfgd_bsh;
        end

    //-----------------------------------------------------------------
    // Contatore dei biquad: a ogni start scorre cnt = 0..NB-1
    //-----------------------------------------------------------------
    reg [LOG2_NB-1:0] cnt;      // biquad in lettura (S1)
    reg               run;      // 1 = scansione in corso
    localparam integer NSTAGE = 13;            // stadi della pipeline dopo il contatore
    reg [NSTAGE-1:0]  vpipe;    // valid degli stadi della pipeline (vedi intestazione)

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cnt <= {LOG2_NB{1'b0}};
            run <= 1'b0;
        end else if (start && !run) begin
            run <= 1'b1;
            cnt <= {LOG2_NB{1'b0}};
        end else if (run) begin
            if (cnt == NB-1) begin
                run <= 1'b0;
                cnt <= {LOG2_NB{1'b0}};
            end else begin
                cnt <= cnt + 1'b1;
            end
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) vpipe <= {NSTAGE{1'b0}};
        else        vpipe <= {vpipe[NSTAGE-2:0], run};
    end

    // clr_r = 1: per tutta la passata lo stato letto vale zero
    // (primo frame di una nuova sequenza). Dopo il reset vale 1.
    reg clr_r;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)                   clr_r <= 1'b1;
        else if (start && !run)       clr_r <= clr;
        else if (run && (cnt == NB-1)) clr_r <= 1'b0;
    end

    // occupata finche' la scansione gira o la pipeline contiene dati
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) busy <= 1'b0;
        else        busy <= run || (|vpipe);
    end

    //=================================================================
    // S1: letture (b0, b1, s1, u, D, bsh)
    //=================================================================
    reg [LOG2_NB-1:0]     a_s1;                     // indirizzo del biquad lungo la pipeline
    reg [2*W_B-1:0]       cfb_s1;                   // {b1, b0}
    reg signed [W_S-1:0]  s1_s1;                    // s1 (0 se clr_r)
    reg signed [W_U-1:0]  u_s1;                     // ingresso del canale
    reg                   clr_s1;                   // stato nullo: vale anche per s2, letto in S3d

    // canale locale del biquad corrente = bit alti dell'indirizzo
    wire [LOG2_NB-LOG2_MOD-1:0] ch_sel = cnt[LOG2_NB-1:LOG2_MOD];
    wire signed [W_U-1:0]       u_now  = u_vec[ch_sel*W_U +: W_U];

    reg signed [W_D-1:0] d_s1;                      // D del canale
    reg [W_BSH-1:0]      bsh_s1;                    // shift dei b del canale

    always @(posedge clk) begin
        cfb_s1 <= coef_b[cnt];
        s1_s1  <= clr_r ? {W_S{1'b0}} : state1[cnt];
        u_s1   <= u_now;
        d_s1   <= dcoef[ch_sel];
        bsh_s1 <= bshift[ch_sel];
        clr_s1 <= clr_r;
        a_s1   <= cnt;
    end

    // separazione dei campi
    wire signed [W_B-1:0] b0_s1 = $signed(cfb_s1[0             +: W_B]);
    wire signed [W_B-1:0] b1_s1 = $signed(cfb_s1[W_B           +: W_B]);

    //=================================================================
    // S2: prodotti con l'ingresso u e mezzo LSB 2^(bsh-1)
    //=================================================================
    reg signed [W_B+W_U-1:0] pb0_s2, pb1_s2;   // b0*u, b1*u (Q.F_B+F_U)
    reg signed [W_D+W_U-1:0] pd_s2;            // D*u (Q.F_D+F_U)
    reg signed [PW-1:0]      rnd_s2;           // 2^(bsh-1), 0 se bsh = 0
    reg signed [W_S-1:0]     s1_s2;
    reg [W_BSH-1:0]          bsh_s2;
    reg                      clr_s2;
    reg [LOG2_NB-1:0]        a_s2;

    always @(posedge clk) begin
        pb0_s2 <= b0_s1 * u_s1;
        pb1_s2 <= b1_s1 * u_s1;
        pd_s2  <= d_s1 * u_s1;
        rnd_s2 <= (bsh_s1 > 0) ? (1 <<< (bsh_s1 - 1)) : {PW{1'b0}};
        s1_s2  <= s1_s1;
        bsh_s2 <= bsh_s1;
        clr_s2 <= clr_s1;
        a_s2   <= a_s1;
    end

    // estensione di segno a PW bit
    wire signed [PW-1:0] pb0_ext = {{(PW-W_B-W_U){pb0_s2[W_B+W_U-1]}}, pb0_s2};
    wire signed [PW-1:0] pb1_ext = {{(PW-W_B-W_U){pb1_s2[W_B+W_U-1]}}, pb1_s2};
    wire signed [PW-1:0] pd_ext  = {{(PW-W_D-W_U){pd_s2[W_D+W_U-1]}}, pd_s2};

    //=================================================================
    // S3a: + mezzo LSB (arrotondamento delle divisioni successive)
    //   b0*u, b1*u: 2^(bsh-1)      D*u: 2^(SH_D-1)
    //=================================================================
    localparam signed [PW-1:0] RND_D = (SH_D > 0) ? (1 <<< (SH_D - 1)) : 0;

    reg signed [PW-1:0]      pb0_s3a, pb1_s3a, pd_s3a;
    reg signed [W_S-1:0]     s1_s3a;
    reg [W_BSH-1:0]          bsh_s3a;
    reg                      clr_s3a;
    reg [LOG2_NB-1:0]        a_s3a;

    always @(posedge clk) begin
        pb0_s3a <= pb0_ext + rnd_s2;
        pb1_s3a <= pb1_ext + rnd_s2;
        pd_s3a  <= pd_ext + RND_D;
        s1_s3a  <= s1_s2;
        bsh_s3a <= bsh_s2;
        clr_s3a <= clr_s2;
        a_s3a   <= a_s2;
    end

    //=================================================================
    // S3b: shift aritmetici (round-half-up insieme al mezzo LSB gia'
    //      sommato, come la vecchia funzione round_shift)
    //   b0*u, b1*u: variabile di bsh bit      D*u: fisso di SH_D bit
    //=================================================================
    reg signed [PW-1:0]      pb0_t_s3b, pb1_t_s3b, du_t_s3b;
    reg signed [W_S-1:0]     s1_s3b;
    reg                      clr_s3b;
    reg [LOG2_NB-1:0]        a_s3b;

    always @(posedge clk) begin
        pb0_t_s3b <= pb0_s3a >>> bsh_s3a;
        pb1_t_s3b <= pb1_s3a >>> bsh_s3a;
        du_t_s3b  <= pd_s3a >>> SH_D;
        s1_s3b    <= s1_s3a;
        clr_s3b   <= clr_s3a;
        a_s3b     <= a_s3a;
    end

    //=================================================================
    // S3c: y = s1 + round(b0*u / 2^bsh)
    //   (= round((s1*2^bsh + b0*u) / 2^bsh) della versione precedente:
    //   s1*2^bsh e' multiplo di 2^bsh ed esce intero dallo shift)
    //=================================================================
    wire signed [PW-1:0] s1_ext = {{(PW-W_S){s1_s3b[W_S-1]}}, s1_s3b};

    reg signed [PW-1:0]      y_t_s3c, pb1_t_s3c, du_t_s3c;
    reg                      clr_s3c;
    reg [LOG2_NB-1:0]        a_s3c;

    always @(posedge clk) begin
        y_t_s3c   <= pb0_t_s3b + s1_ext;
        pb1_t_s3c <= pb1_t_s3b;
        du_t_s3c  <= du_t_s3b;
        clr_s3c   <= clr_s3b;
        a_s3c     <= a_s3b;
    end

    //-----------------------------------------------------------------
    // sat_only: saturazione a W_S bit, restituisce {overflow, valore}
    //-----------------------------------------------------------------
    function [W_S:0] sat_only;
        input signed [PW-1:0] t;
        begin
            if      (t > SMAX) sat_only = {1'b1, SMAX[W_S-1:0]};
            else if (t < SMIN) sat_only = {1'b1, SMIN[W_S-1:0]};
            else               sat_only = {1'b0, t[W_S-1:0]};
        end
    endfunction

    //=================================================================
    // S3d: saturazione a W_S bit di y, b1*u, D*u; lettura di a1, a2 e
    //      s2 del biquad (servono da S4a in poi)
    //=================================================================
    wire [W_S:0] y_r   = sat_only(y_t_s3c);
    wire [W_S:0] pb1_r = sat_only(pb1_t_s3c);
    wire [W_S:0] du_r  = sat_only(du_t_s3c);

    reg signed [W_S-1:0] y_s3, pb1_s3, du_s3, s2_s3;
    reg [2*W_A-1:0]      cfa_s3;                // {a2, a1}
    reg                  ov_s3;                 // saturazione in questo stadio
    reg [LOG2_NB-1:0]    a_s3d;

    always @(posedge clk) begin
        y_s3   <= $signed(y_r[W_S-1:0]);
        pb1_s3 <= $signed(pb1_r[W_S-1:0]);
        du_s3  <= $signed(du_r[W_S-1:0]);
        ov_s3  <= y_r[W_S] | pb1_r[W_S] | du_r[W_S];
        cfa_s3 <= coef_a[a_s3c];
        s2_s3  <= clr_s3c ? {W_S{1'b0}} : state2[a_s3c];
        a_s3d  <= a_s3c;
    end

    wire signed [W_A-1:0] a1_s3 = $signed(cfa_s3[0             +: W_A]);
    wire signed [W_A-1:0] a2_s3 = $signed(cfa_s3[W_A           +: W_A]);

    //=================================================================
    // S4a-S4c: prodotti a1*y, a2*y (24 x 32 bit) con tre registri: la
    // sintesi li sposta dentro i DSP in cascata (registri M e P).
    // In S4a, in parallelo: c = b1*u + s2 (W_S+1 bit, senza saturazione)
    //=================================================================
    reg signed [PW-1:0]  pa1_s4a, pa2_s4a, pa1_s4b, pa2_s4b, pa1_s4c, pa2_s4c;
    reg signed [W_S:0]   c_s4a, c_s4b, c_s4c;
    reg                  ov_s4a, ov_s4b, ov_s4c;
    reg [LOG2_NB-1:0]    a_s4a, a_s4b, a_s4c;

    always @(posedge clk) begin
        pa1_s4a <= a1_s3 * y_s3;
        pa2_s4a <= a2_s3 * y_s3;
        pa1_s4b <= pa1_s4a;
        pa2_s4b <= pa2_s4a;
        pa1_s4c <= pa1_s4b;
        pa2_s4c <= pa2_s4b;
        c_s4a   <= {pb1_s3[W_S-1], pb1_s3} + {s2_s3[W_S-1], s2_s3};
        c_s4b   <= c_s4a;   c_s4c   <= c_s4b;
        ov_s4a  <= ov_s3;   ov_s4b  <= ov_s4a;   ov_s4c  <= ov_s4b;
        a_s4a   <= a_s3d;   a_s4b   <= a_s4a;    a_s4c   <= a_s4b;
    end

    //=================================================================
    // S4d: a*y + mezzo LSB (a*y ha F_A+F_Y bit frazionari -> Q.F_Y)
    //=================================================================
    localparam signed [PW-1:0] RND_A = (SH_A > 0) ? (1 <<< (SH_A - 1)) : 0;

    reg signed [PW-1:0]  pa1_s4d, pa2_s4d;
    reg signed [W_S:0]   c_s4d;
    reg                  ov_s4d;
    reg [LOG2_NB-1:0]    a_s4d;

    always @(posedge clk) begin
        pa1_s4d <= pa1_s4c + RND_A;
        pa2_s4d <= pa2_s4c + RND_A;
        c_s4d   <= c_s4c;
        ov_s4d  <= ov_s4c;
        a_s4d   <= a_s4c;
    end

    //=================================================================
    // S4e: shift fisso di SH_A bit (fili) e saturazione a W_S bit
    //=================================================================
    wire signed [PW-1:0] pa1_sh = pa1_s4d >>> SH_A;
    wire signed [PW-1:0] pa2_sh = pa2_s4d >>> SH_A;
    wire [W_S:0]         a1y_r  = sat_only(pa1_sh);
    wire [W_S:0]         a2y_r  = sat_only(pa2_sh);

    reg signed [W_S-1:0] a1y_s4e, a2y_s4e;
    reg signed [W_S:0]   c_s4e;
    reg                  ov_s4e;
    reg [LOG2_NB-1:0]    a_s4e;

    always @(posedge clk) begin
        a1y_s4e <= $signed(a1y_r[W_S-1:0]);
        a2y_s4e <= $signed(a2y_r[W_S-1:0]);
        c_s4e   <= c_s4d;
        ov_s4e  <= ov_s4d | a1y_r[W_S] | a2y_r[W_S];
        a_s4e   <= a_s4d;
    end

    //=================================================================
    // S4f: nuovo s1 non saturato = (b1*u + s2) - a1*y (su PW bit)
    //      nuovo s2 = -a2*y (su W_S bit: -SMIN torna SMIN come nella
    //      versione precedente)
    //=================================================================
    reg signed [PW-1:0]  s1_t_s4f;
    reg signed [W_S-1:0] s2_new_s4f;
    reg                  ov_s4f;
    reg [LOG2_NB-1:0]    a_s4f;

    always @(posedge clk) begin
        s1_t_s4f   <= {{(PW-W_S-1){c_s4e[W_S]}}, c_s4e}
                    - {{(PW-W_S){a1y_s4e[W_S-1]}}, a1y_s4e};
        s2_new_s4f <= -a2y_s4e;
        ov_s4f     <= ov_s4e;
        a_s4f      <= a_s4e;
    end

    //=================================================================
    // S4g: saturazione del nuovo s1, poi scrittura dello stato {s2, s1}
    //      (la memoria viene scritta da registri, nel ciclo successivo)
    //=================================================================
    wire [W_S:0] s1_r = sat_only(s1_t_s4f);

    reg signed [W_S-1:0] s1_new_s4g, s2_new_s4g;
    reg                  ov_s4g;
    reg [LOG2_NB-1:0]    a_s4g;

    always @(posedge clk) begin
        s1_new_s4g <= $signed(s1_r[W_S-1:0]);
        s2_new_s4g <= s2_new_s4f;
        ov_s4g     <= ov_s4f | s1_r[W_S];
        a_s4g      <= a_s4f;
    end

    always @(posedge clk)
        if (vpipe[12]) begin
            state1[a_s4g] <= s1_new_s4g;
            state2[a_s4g] <= s2_new_s4g;
        end

    // saturazione in un qualsiasi punto del calcolo di questo biquad
    wire stage4_ovf = vpipe[12] & ov_s4g;

    //=================================================================
    // Accumulo sul canale (usa y saturato di S3d)
    //   primo modo : acc = D*u + y
    //   altri modi : acc = acc + y
    //   ultimo modo: il risultato esce su out_acc
    //=================================================================
    wire [LOG2_MOD-1:0] mode_s3  = a_s3d[LOG2_MOD-1:0];  // modo = bit bassi dell'indirizzo
    wire                first_s3 = (mode_s3 == 0);
    wire                last_s3  = (mode_s3 == MODES-1);

    reg  signed [W_ACC-1:0] acc;
    wire signed [W_ACC-1:0] y_ext   = {{(W_ACC-W_S){y_s3[W_S-1]}}, y_s3};
    wire signed [W_ACC-1:0] du_ext  = {{(W_ACC-W_S){du_s3[W_S-1]}}, du_s3};

    wire signed [W_ACC-1:0] acc_nxt = (first_s3 ? du_ext : acc) + y_ext;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc       <= {W_ACC{1'b0}};
            out_valid <= 1'b0;
            out_acc   <= {W_ACC{1'b0}};
        end else begin
            out_valid <= 1'b0;
            if (vpipe[5]) begin
                acc <= acc_nxt;
                if (last_s3) begin
                    out_valid <= 1'b1;
                    out_acc   <= acc_nxt;
                end
            end
        end
    end

    // flag di overflow (resta a 1 fino al reset)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)          ovf <= 1'b0;
        else if (stage4_ovf) ovf <= 1'b1;
    end

endmodule

`default_nettype wire
