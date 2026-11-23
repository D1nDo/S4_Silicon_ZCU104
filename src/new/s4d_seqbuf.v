`timescale 1ns / 1ps
`default_nettype none
//=====================================================================
// s4d_seqbuf  --  buffer della sequenza
//
//   Memoria dual port da NWORDS parole x W bit, in Block RAM.
//   Contiene l'intera sequenza (L timestep x H canali) tra una
//   passata e l'altra del layer.
//     - porta di scrittura: 1 parola per ciclo
//     - porta di lettura  : sincrona, dato disponibile RD_LAT cicli dopo rd_en
//       RD_LAT = 1: uscita diretta della BRAM
//       RD_LAT = 2: + registro d'uscita (assorbito nel DOB_REG della BRAM)
//       RD_LAT = 3: + un registro DOPO il multiplexer fra le catene di BRAM
//                   (consigliato a 250+ MHz, vedi CASCATA sotto)
//
//   CASCATA (timing 2026-10-05): senza vincoli Vivado costruiva la memoria
//   come catene di 8 BRAM in cascata; la lettura attraversava 7 BRAM +
//   una LUT6 in un solo ciclo (2.56 ns di logica, slack misurato
//   -0.153 ns a 266.7 MHz, unico percorso in violazione). Ora:
//     cascade_height = 2 -> catene di al massimo 2 BRAM;
//     RD_LAT = 3         -> uscita BRAM, multiplexer fra le catene e
//                           registro finale in cicli separati.
//   Versione precedente: sqbuf.v.bak3 (sul Desktop).
//   Se nello stesso ciclo si legge e si scrive lo stesso indirizzo, la
//   lettura restituisce il valore VECCHIO (read-first).
//   Copia della versione precedente (solo latenza 1): sqbuf.v.bak
//
//   Controllo err_ovw: durante una passata la scrittura avviene "in
//   place" e deve restare sempre DIETRO alla lettura (si sovrascrive
//   solo cio' che e' gia' stato letto). Se wr_addr >= rd_addr si
//   starebbe sovrascrivendo un dato non ancora letto -> flag sticky.
//=====================================================================

module s4d_seqbuf #(
    parameter integer W      = 16,       // larghezza della parola
    parameter integer NWORDS = 131072,   // parole (L*H)
    parameter integer AW     = 17,       // bit di indirizzo
    parameter integer RD_LAT = 3         // latenza di lettura: 1 = BRAM sola, 2 = + registro d'uscita,
                                         // 3 = + registro dopo il multiplexer fra le catene
)(
    input  wire                clk,
    input  wire                rst_n,

    // porta di scrittura
    input  wire                wr_en,
    input  wire [AW-1:0]       wr_addr,
    input  wire signed [W-1:0] wr_data,

    // porta di lettura (latenza RD_LAT cicli)
    input  wire                rd_en,
    input  wire [AW-1:0]       rd_addr,
    output wire signed [W-1:0] rd_data,   // era output reg: valido RD_LAT cicli dopo rd_en

    // controllo sovrascrittura
    input  wire                chk_en,   // 1 = controllo attivo (durante le passate)
    output reg                 err_ovw   // sticky: scrittura >= lettura
);

    (* ram_style = "block", cascade_height = 2 *)   // catene di al massimo 2 BRAM
    reg [W-1:0] mem [0:NWORDS-1];
    reg [W-1:0] rd_mem_q;                 // uscita della BRAM (latenza 1)

    // scrittura e lettura sincrone nella stessa memoria
    always @(posedge clk) begin
        if (wr_en) mem[wr_addr] <= wr_data;
        if (rd_en) rd_mem_q <= mem[rd_addr];
    end

    generate
    if (RD_LAT == 3) begin : g_out_reg2
        reg [W-1:0] rd_out_q;             // registro d'uscita della BRAM (DOB_REG)
        reg [W-1:0] rd_out_q2;            // registro dopo il multiplexer fra le catene
        always @(posedge clk) begin
            rd_out_q  <= rd_mem_q;
            rd_out_q2 <= rd_out_q;
        end
        assign rd_data = $signed(rd_out_q2);
    end else if (RD_LAT == 2) begin : g_out_reg
        reg [W-1:0] rd_out_q;             // registro d'uscita, sempre abilitato
        always @(posedge clk) rd_out_q <= rd_mem_q;
        assign rd_data = $signed(rd_out_q);
    end else begin : g_no_out_reg
        assign rd_data = $signed(rd_mem_q);
    end
    endgenerate

    // flag di sovrascrittura (resta a 1 fino al reset)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)                                    err_ovw <= 1'b0;
        else if (chk_en && wr_en && (wr_addr >= rd_addr)) err_ovw <= 1'b1;
    end

endmodule

`default_nettype wire
