`timescale 1ns / 1ps
`default_nettype none
//=====================================================================
// s4d_wload_axis  --  caricatore dei pesi di UN layer da AXI-Stream
//
//   A ogni impulso di start consuma esattamente NBEAT beat da 64 bit
//   dallo stream s_axis_w e, in base alla posizione del beat (pos),
//   genera la scrittura verso la memoria giusta del layer.
//
//   Formato dello stream per un layer (beat consecutivi, in ordine):
//
//   | pos (da..a)        | beat        | contenuto di un beat               | destinazione     |
//   |--------------------|-------------|------------------------------------|------------------|
//   | 0                  | 1           | [5:0] shift del BatchNorm          | s4d_affine       |
//   | C_BEG..C_END-1     | 2*NCOEF     | coefficienti biquad, 2 beat l'uno: | s4d_biquad_unit  |
//   |                    |  (=8192)    |   pari   : bit [63:0] della parola |                  |
//   |                    |             |   dispari: bit [CW-1:64]           |                  |
//   | C_END..D_END-1     | H   (=128)  | [15:0] D, [20:16] shift dei b      | s4d_biquad_unit  |
//   | D_END..NW_END-1    | H   (=128)  | [15:0] peso BatchNorm              | s4d_affine       |
//   | NW_END..NB_END-1   | H   (=128)  | [31:0] bias BatchNorm              | s4d_affine       |
//   | NB_END..BO_END-1   | TH  (=256)  | [31:0] bias del mixing             | s4d_mixing_glu   |
//   | BO_END..WO_END-1   | NWO*WOBEAT  | pesi del mixing, 48 beat per riga  | s4d_mixing_glu   |
//   |                    |  (=6144)    |   da NMAC*W_W = 3072 bit; riga h = |                  |
//   |                    |             |   W[c][h], c=0..255 (c=0 negli LSB)|                  |
//
//   Totale NBEAT = 1 + 8192 + 128 + 128 + 128 + 256 + 6144 = 14977 beat
//   (valori con i parametri di default del top).
//
//   Coefficienti biquad: parola da CW = 2*W_A + 2*W_B = 80 bit
//     {b1, b0, a2, a1}  (a1 negli LSB). Indice della parola
//     k = (pos - C_BEG) / 2 = unita' * 2^LOG2_NB + indirizzo nell'unita'.
//
//   Tutte le uscite di scrittura sono registrate e durano 1 ciclo.
//=====================================================================

module s4d_wload_axis #(
    parameter integer H        = 128,    // canali del modello
    parameter integer LOG2_H   = 7,
    parameter integer NUNITS   = 8,      // unita' biquad
    parameter integer LOG2_NU  = 3,
    parameter integer LOG2_NB  = 9,      // bit di indirizzo coefficienti per unita' (512)
    parameter integer NMAC     = 256,    // MAC in parallelo nel mixing (era 64)
    parameter integer LOG2_NPH = 0,      // una sola fase (era 2: 4 fasi)
    parameter integer WO_SEL_W = 6,      // bit di ld_wo_sel: WOBEAT = 48 -> 6 bit (era 4)
    parameter integer W_A      = 24,     // coefficienti a1,a2
    parameter integer W_B      = 16,     // coefficienti b0,b1
    parameter integer W_D      = 16,     // coefficiente D
    parameter integer W_BSH    = 5,      // shift dei coefficienti b
    parameter integer BN_W_W   = 16,     // peso BatchNorm
    parameter integer BN_W_B   = 32,     // bias BatchNorm
    parameter integer W_MACC   = 32,     // bias del mixing
    parameter integer W_W      = 12,     // pesi del mixing
    parameter integer AXI_DW   = 64      // larghezza dello stream
)(
    input  wire                  clk,
    input  wire                  rst_n,

    input  wire                  start,  // impulso: inizia a caricare un layer
    output reg                   busy,   // 1 = caricamento in corso
    output reg                   done,   // impulso: layer caricato

    // stream dei pesi: NBEAT beat consecutivi per layer, in ordine.
    // tlast e' accettato ma IGNORATO: il DMA puo' inviare i 6 layer in un
    // unico trasferimento; il conteggio interno fa da confine fra layer.
    input  wire                  s_axis_w_tvalid,
    output wire                  s_axis_w_tready,
    input  wire [AXI_DW-1:0]     s_axis_w_tdata,
    input  wire                  s_axis_w_tlast,

    // ---- coefficienti dei biquad: parola {b1,b0,a2,a1} ----
    output reg                   cfg_we,
    output reg  [LOG2_NU-1:0]    cfg_unit,    // unita' destinataria
    output reg  [LOG2_NB-1:0]    cfg_addr,    // indirizzo nell'unita' (canale*MODES + modo)
    output reg  [2*W_A+2*W_B-1:0] cfg_data,
    // ---- coefficiente D e shift dei b, per canale ----
    output reg                   cfgd_we,
    output reg  [LOG2_H-1:0]     cfgd_ch,
    output reg  signed [W_D-1:0] cfgd_data,
    output reg  [W_BSH-1:0]      cfgd_bsh,

    // ---- BatchNorm ----
    output reg                   ld_bn_we_sh, // scrive lo shift (uno per layer)
    output reg  [5:0]            ld_bn_sh,
    output reg                   ld_bn_we_w,  // scrive un peso
    output reg                   ld_bn_we_b,  // scrive un bias
    output reg  [LOG2_H-1:0]     ld_bn_addr,  // canale
    output reg  signed [BN_W_W-1:0] ld_bn_w,
    output reg  signed [BN_W_B-1:0] ld_bn_b,

    // ---- mixing: pesi (una riga = WOBEAT parole da 64 bit) e bias ----
    output reg                   ld_wo_we,
    output reg  [LOG2_NPH+LOG2_H-1:0] ld_wo_addr,   // riga = ingresso h (una sola fase)
    output reg  [WO_SEL_W-1:0]   ld_wo_sel,         // banco da 64 bit della riga (0..WOBEAT-1), era [3:0]
    output reg  [AXI_DW-1:0]     ld_wo_data,
    output reg                   ld_bo_we,
    output reg  [LOG2_H:0]       ld_bo_addr,        // 0..2H-1
    output reg  signed [W_MACC-1:0] ld_bo_data
);

    localparam integer CW     = 2*W_A + 2*W_B;          // bit di una parola coefficienti (80)
    localparam integer NCOEF  = NUNITS * (1 << LOG2_NB);// parole coefficienti in totale (4096)
    localparam integer NWO    = (1 << LOG2_NPH) * H;    // righe di pesi del mixing (128 = H)
    localparam integer WOBEAT = (NMAC*W_W) / AXI_DW;    // beat per riga di pesi (3072/64 = 48)

    // controllo: ld_wo_sel/wo_sel devono poter contare fino a WOBEAT-1
    initial if ((1 << WO_SEL_W) < WOBEAT) begin
        $display("s4d_wload_axis: WO_SEL_W=%0d troppo stretto per WOBEAT=%0d", WO_SEL_W, WOBEAT);
        $finish;
    end
    localparam integer TH     = 2*H;                    // uscite del mixing (256)

    // confini delle regioni dello stream (pos di fine, esclusa)
    localparam integer H_END  = 1;                      // shift BatchNorm
    localparam integer C_BEG  = H_END;
    localparam integer C_END  = C_BEG + 2*NCOEF;        // coefficienti biquad
    localparam integer D_END  = C_END + H;              // D + shift dei b
    localparam integer NW_END = D_END + H;              // pesi BatchNorm
    localparam integer NB_END = NW_END + H;             // bias BatchNorm
    localparam integer BO_END = NB_END + TH;            // bias mixing
    localparam integer WO_END = BO_END + NWO*WOBEAT;    // pesi mixing
    localparam integer NBEAT  = WO_END;                 // beat totali per layer

    // si accettano beat solo mentre si sta caricando (ready da registro)
    assign s_axis_w_tready = busy;
    wire unused_tlast      = s_axis_w_tlast;            // tlast volutamente non usato

    //-----------------------------------------------------------------
    // Posizione nello stream (timing, 2026-10-15).
    // Prima: un contatore pos e, a ogni beat, una catena di confronti
    // pos < H_END, pos < C_END, ... in serie, piu' le sottrazioni pos -
    // inizio regione. Ora la regione corrente e' un REGISTRO (region) e
    // sub_pos conta i beat dentro la regione: si passa alla regione
    // successiva quando sub_pos raggiunge la sua lunghezza - 1 (confronti
    // di uguaglianza con costanti, in parallelo). Uscite identiche.
    //-----------------------------------------------------------------
    localparam [2:0] RG_BNSH = 3'd0,    // shift del BatchNorm (1 beat)
                     RG_COEF = 3'd1,    // coefficienti biquad (2*NCOEF beat)
                     RG_D    = 3'd2,    // D + shift dei b (H beat)
                     RG_BNW  = 3'd3,    // pesi BatchNorm (H beat)
                     RG_BNB  = 3'd4,    // bias BatchNorm (H beat)
                     RG_BO   = 3'd5,    // bias del mixing (TH beat)
                     RG_WO   = 3'd6;    // pesi del mixing (NWO*WOBEAT beat)

    localparam integer SW = 15;                         // bit di sub_pos (regione piu' lunga < 2^15)

    reg [2:0]    region;                // regione del beat atteso
    reg [SW-1:0] sub_pos;               // indice del beat dentro la regione

    // ultimo beat della regione corrente: uguaglianze con costanti, una
    // per regione, calcolate in parallelo e scelte da region
    reg region_last;
    always @(*) begin
        case (region)
            RG_BNSH: region_last = (sub_pos == (H_END  - 1));
            RG_COEF: region_last = (sub_pos == (C_END  - C_BEG  - 1));
            RG_D:    region_last = (sub_pos == (D_END  - C_END  - 1));
            RG_BNW:  region_last = (sub_pos == (NW_END - D_END  - 1));
            RG_BNB:  region_last = (sub_pos == (NB_END - NW_END - 1));
            RG_BO:   region_last = (sub_pos == (BO_END - NB_END - 1));
            default: region_last = (sub_pos == (WO_END - BO_END - 1));
        endcase
    end

    reg [63:0] coef_lo;             // prima meta' (64 bit bassi) di una parola coefficienti

    wire       r_hs = s_axis_w_tvalid & s_axis_w_tready;   // handshake: beat ricevuto

    reg [LOG2_NPH+LOG2_H-1:0] wo_row;   // riga corrente dei pesi del mixing
    reg [WO_SEL_W-1:0]        wo_sel;   // parola da 64 bit dentro la riga (0..WOBEAT-1)

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy          <= 1'b0;
            done          <= 1'b0;
            region        <= RG_BNSH;
            sub_pos       <= {SW{1'b0}};
            wo_row        <= {(LOG2_NPH+LOG2_H){1'b0}};
            wo_sel        <= {WO_SEL_W{1'b0}};
            coef_lo       <= 64'd0;
            cfg_we        <= 1'b0;
            cfgd_we       <= 1'b0;
            ld_bn_we_sh   <= 1'b0;
            ld_bn_sh      <= 6'd0;
            ld_bn_we_w    <= 1'b0;
            ld_bn_we_b    <= 1'b0;
            ld_wo_we      <= 1'b0;
            ld_bo_we      <= 1'b0;
        end else begin
            // tutti gli strobe di scrittura sono impulsi di 1 ciclo
            done       <= 1'b0;
            cfg_we     <= 1'b0;
            cfgd_we    <= 1'b0;
            ld_bn_we_sh <= 1'b0;
            ld_bn_we_w <= 1'b0;
            ld_bn_we_b <= 1'b0;
            ld_wo_we   <= 1'b0;
            ld_bo_we   <= 1'b0;

            if (!busy) begin
                // fermo: aspetta start e azzera i contatori
                if (start) begin
                    busy    <= 1'b1;
                    region  <= RG_BNSH;
                    sub_pos <= {SW{1'b0}};
                    wo_row  <= {(LOG2_NPH+LOG2_H){1'b0}};
                    wo_sel  <= {WO_SEL_W{1'b0}};
                end
            end else begin

                if (r_hs) begin
                    // avanzamento della posizione: regione successiva o beat successivo
                    if (region_last) begin
                        region  <= region + 3'd1;
                        sub_pos <= {SW{1'b0}};
                    end else begin
                        sub_pos <= sub_pos + 1'b1;
                    end

                    // smistamento del beat in base alla regione
                    case (region)
                        RG_BNSH: begin
                            // shift del BatchNorm
                            ld_bn_sh    <= s_axis_w_tdata[5:0];
                            ld_bn_we_sh <= 1'b1;
                        end
                        RG_COEF: begin
                            // coefficienti biquad: beat pari = 64 bit bassi,
                            // beat dispari = bit alti -> scrittura della parola intera
                            if (!sub_pos[0]) begin
                                coef_lo <= s_axis_w_tdata;
                            end else begin
                                cfg_data <= {s_axis_w_tdata[CW-64-1:0], coef_lo};
                                cfg_unit <= sub_pos[LOG2_NB+LOG2_NU:LOG2_NB+1];   // bit alti di k
                                cfg_addr <= sub_pos[LOG2_NB:1];                   // bit bassi di k
                                cfg_we   <= 1'b1;
                            end
                        end
                        RG_D: begin
                            // D e shift dei b del canale sub_pos
                            cfgd_ch   <= sub_pos[LOG2_H-1:0];
                            cfgd_data <= $signed(s_axis_w_tdata[W_D-1:0]);
                            cfgd_bsh  <= s_axis_w_tdata[W_D+W_BSH-1:W_D];
                            cfgd_we   <= 1'b1;
                        end
                        RG_BNW: begin
                            // peso BatchNorm del canale sub_pos
                            ld_bn_addr <= sub_pos[LOG2_H-1:0];
                            ld_bn_w    <= $signed(s_axis_w_tdata[BN_W_W-1:0]);
                            ld_bn_we_w <= 1'b1;
                        end
                        RG_BNB: begin
                            // bias BatchNorm del canale sub_pos
                            ld_bn_addr <= sub_pos[LOG2_H-1:0];
                            ld_bn_b    <= $signed(s_axis_w_tdata[BN_W_B-1:0]);
                            ld_bn_we_b <= 1'b1;
                        end
                        RG_BO: begin
                            // bias del mixing, uscita sub_pos (0..2H-1)
                            ld_bo_addr <= sub_pos[LOG2_H:0];
                            ld_bo_data <= $signed(s_axis_w_tdata[W_MACC-1:0]);
                            ld_bo_we   <= 1'b1;
                        end
                        default: begin
                            // pesi del mixing: WOBEAT beat consecutivi per riga,
                            // il beat wo_sel va nel banco wo_sel della riga wo_row
                            ld_wo_addr <= wo_row;
                            ld_wo_sel  <= wo_sel;
                            ld_wo_data <= s_axis_w_tdata;
                            ld_wo_we   <= 1'b1;
                            // confronto a WO_SEL_W bit: con [3:0] WOBEAT=48 valeva 0
                            // e la riga si sarebbe spezzata ogni 16 beat
                            if (wo_sel == WOBEAT[WO_SEL_W-1:0] - 1'b1) begin
                                wo_sel <= {WO_SEL_W{1'b0}};
                                wo_row <= wo_row + 1'b1;
                            end else begin
                                wo_sel <= wo_sel + 1'b1;
                            end
                        end
                    endcase

                    // ultimo beat del layer: fine caricamento
                    if ((region == RG_WO) && region_last) begin
                        busy <= 1'b0;
                        done <= 1'b1;
                    end
                end
            end
        end
    end

endmodule

`default_nettype wire
