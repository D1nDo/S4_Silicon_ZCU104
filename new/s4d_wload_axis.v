`timescale 1ns / 1ps
`default_nettype none
//=====================================================================
// s4d_wload_axis  --  loader of the weights of ONE layer from AXI-Stream
//
//   At every start pulse it consumes exactly NBEAT beats of 64 bits
//   from the s_axis_w stream and, depending on the position of the beat
//   (pos), it generates the write to the right memory of the layer.
//
//   Stream format for one layer (consecutive beats, in order):
//
//   | pos (from..to)     | beats       | content of one beat                | destination      |
//   |--------------------|-------------|------------------------------------|------------------|
//   | 0                  | 1           | [5:0] BatchNorm shift              | s4d_affine       |
//   | C_BEG..C_END-1     | 2*NCOEF     | biquad coefficients, 2 beats each: | s4d_biquad_unit  |
//   |                    |  (=8192)    |   even: bits [63:0] of the word    |                  |
//   |                    |             |   odd : bits [CW-1:64]             |                  |
//   | C_END..D_END-1     | H   (=128)  | [15:0] D, [20:16] shift of the b   | s4d_biquad_unit  |
//   | D_END..NW_END-1    | H   (=128)  | [15:0] BatchNorm weight            | s4d_affine       |
//   | NW_END..NB_END-1   | H   (=128)  | [31:0] BatchNorm bias              | s4d_affine       |
//   | NB_END..BO_END-1   | TH  (=256)  | [31:0] mixing bias                 | s4d_mixing_glu   |
//   | BO_END..WO_END-1   | NWO*WOBEAT  | mixing weights, 48 beats per row   | s4d_mixing_glu   |
//   |                    |  (=6144)    |   of NMAC*W_W = 3072 bits; row h = |                  |
//   |                    |             |   W[c][h], c=0..255 (c=0 in LSBs)  |                  |
//
//   Total NBEAT = 1 + 8192 + 128 + 128 + 128 + 256 + 6144 = 14977 beats
//   (values with the default parameters of the top).
//
//   Biquad coefficients: word of CW = 2*W_A + 2*W_B = 80 bits
//     {b1, b0, a2, a1}  (a1 in the LSBs). Index of the word
//     k = (pos - C_BEG) / 2 = unit * 2^LOG2_NB + address in the unit.
//
//   All the write outputs are registered and last 1 cycle.
//=====================================================================

module s4d_wload_axis #(
    parameter integer H        = 128,    // channels of the model
    parameter integer LOG2_H   = 7,
    parameter integer NUNITS   = 8,      // biquad units
    parameter integer LOG2_NU  = 3,
    parameter integer LOG2_NB  = 9,      // coefficient address bits per unit (512)
    parameter integer NMAC     = 256,    // parallel MACs in the mixing (was 64)
    parameter integer LOG2_NPH = 0,      // a single phase (was 2: 4 phases)
    parameter integer WO_SEL_W = 6,      // bits of ld_wo_sel: WOBEAT = 48 -> 6 bits (was 4)
    parameter integer W_A      = 24,     // coefficients a1,a2
    parameter integer W_B      = 16,     // coefficients b0,b1
    parameter integer W_D      = 16,     // coefficient D
    parameter integer W_BSH    = 5,      // shift of the b coefficients
    parameter integer BN_W_W   = 16,     // BatchNorm weight
    parameter integer BN_W_B   = 32,     // BatchNorm bias
    parameter integer W_MACC   = 32,     // mixing bias
    parameter integer W_W      = 12,     // mixing weights
    parameter integer AXI_DW   = 64      // stream width
)(
    input  wire                  clk,
    input  wire                  rst_n,

    input  wire                  start,  // pulse: start loading a layer
    output reg                   busy,   // 1 = loading in progress
    output reg                   done,   // pulse: layer loaded

    // weight stream: NBEAT consecutive beats per layer, in order.
    // tlast is accepted but IGNORED: the DMA can send the 6 layers in a
    // single transfer; the internal count acts as the boundary between layers.
    input  wire                  s_axis_w_tvalid,
    output wire                  s_axis_w_tready,
    input  wire [AXI_DW-1:0]     s_axis_w_tdata,
    input  wire                  s_axis_w_tlast,

    // ---- biquad coefficients: word {b1,b0,a2,a1} ----
    output reg                   cfg_we,
    output reg  [LOG2_NU-1:0]    cfg_unit,    // destination unit
    output reg  [LOG2_NB-1:0]    cfg_addr,    // address in the unit (channel*MODES + mode)
    output reg  [2*W_A+2*W_B-1:0] cfg_data,
    // ---- coefficient D and shift of the b, per channel ----
    output reg                   cfgd_we,
    output reg  [LOG2_H-1:0]     cfgd_ch,
    output reg  signed [W_D-1:0] cfgd_data,
    output reg  [W_BSH-1:0]      cfgd_bsh,

    // ---- BatchNorm ----
    output reg                   ld_bn_we_sh, // writes the shift (one per layer)
    output reg  [5:0]            ld_bn_sh,
    output reg                   ld_bn_we_w,  // writes a weight
    output reg                   ld_bn_we_b,  // writes a bias
    output reg  [LOG2_H-1:0]     ld_bn_addr,  // channel
    output reg  signed [BN_W_W-1:0] ld_bn_w,
    output reg  signed [BN_W_B-1:0] ld_bn_b,

    // ---- mixing: weights (one row = WOBEAT words of 64 bits) and biases ----
    output reg                   ld_wo_we,
    output reg  [LOG2_NPH+LOG2_H-1:0] ld_wo_addr,   // row = input h (a single phase)
    output reg  [WO_SEL_W-1:0]   ld_wo_sel,         // 64-bit bank of the row (0..WOBEAT-1), was [3:0]
    output reg  [AXI_DW-1:0]     ld_wo_data,
    output reg                   ld_bo_we,
    output reg  [LOG2_H:0]       ld_bo_addr,        // 0..2H-1
    output reg  signed [W_MACC-1:0] ld_bo_data
);

    localparam integer CW     = 2*W_A + 2*W_B;          // bits of a coefficient word (80)
    localparam integer NCOEF  = NUNITS * (1 << LOG2_NB);// coefficient words in total (4096)
    localparam integer NWO    = (1 << LOG2_NPH) * H;    // rows of mixing weights (128 = H)
    localparam integer WOBEAT = (NMAC*W_W) / AXI_DW;    // beats per row of weights (3072/64 = 48)

    // check: ld_wo_sel/wo_sel must be able to count up to WOBEAT-1
    initial if ((1 << WO_SEL_W) < WOBEAT) begin
        $display("s4d_wload_axis: WO_SEL_W=%0d too narrow for WOBEAT=%0d", WO_SEL_W, WOBEAT);
        $finish;
    end
    localparam integer TH     = 2*H;                    // mixing outputs (256)

    // boundaries of the stream regions (end pos, excluded)
    localparam integer H_END  = 1;                      // BatchNorm shift
    localparam integer C_BEG  = H_END;
    localparam integer C_END  = C_BEG + 2*NCOEF;        // biquad coefficients
    localparam integer D_END  = C_END + H;              // D + shift of the b
    localparam integer NW_END = D_END + H;              // BatchNorm weights
    localparam integer NB_END = NW_END + H;             // BatchNorm biases
    localparam integer BO_END = NB_END + TH;            // mixing biases
    localparam integer WO_END = BO_END + NWO*WOBEAT;    // mixing weights
    localparam integer NBEAT  = WO_END;                 // total beats per layer

    // beats are accepted only while loading (ready from a register)
    assign s_axis_w_tready = busy;
    wire unused_tlast      = s_axis_w_tlast;            // tlast deliberately unused

    //-----------------------------------------------------------------
    // Position in the stream (timing, 2026-10-15).
    // Before: a pos counter and, at every beat, a chain of comparisons
    // pos < H_END, pos < C_END, ... in series, plus the subtractions pos -
    // region start. Now the current region is a REGISTER (region) and
    // sub_pos counts the beats inside the region: the next region is
    // entered when sub_pos reaches its length - 1 (equality comparisons
    // with constants, in parallel). Identical outputs.
    //-----------------------------------------------------------------
    localparam [2:0] RG_BNSH = 3'd0,    // BatchNorm shift (1 beat)
                     RG_COEF = 3'd1,    // biquad coefficients (2*NCOEF beats)
                     RG_D    = 3'd2,    // D + shift of the b (H beats)
                     RG_BNW  = 3'd3,    // BatchNorm weights (H beats)
                     RG_BNB  = 3'd4,    // BatchNorm biases (H beats)
                     RG_BO   = 3'd5,    // mixing biases (TH beats)
                     RG_WO   = 3'd6;    // mixing weights (NWO*WOBEAT beats)

    localparam integer SW = 15;                         // bits of sub_pos (longest region < 2^15)

    reg [2:0]    region;                // region of the expected beat
    reg [SW-1:0] sub_pos;               // index of the beat inside the region

    // last beat of the current region: equalities with constants, one
    // per region, computed in parallel and selected by region
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

    reg [63:0] coef_lo;             // first half (low 64 bits) of a coefficient word

    wire       r_hs = s_axis_w_tvalid & s_axis_w_tready;   // handshake: beat received

    reg [LOG2_NPH+LOG2_H-1:0] wo_row;   // current row of the mixing weights
    reg [WO_SEL_W-1:0]        wo_sel;   // 64-bit word inside the row (0..WOBEAT-1)

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
            // all the write strobes are 1-cycle pulses
            done       <= 1'b0;
            cfg_we     <= 1'b0;
            cfgd_we    <= 1'b0;
            ld_bn_we_sh <= 1'b0;
            ld_bn_we_w <= 1'b0;
            ld_bn_we_b <= 1'b0;
            ld_wo_we   <= 1'b0;
            ld_bo_we   <= 1'b0;

            if (!busy) begin
                // idle: wait for start and clear the counters
                if (start) begin
                    busy    <= 1'b1;
                    region  <= RG_BNSH;
                    sub_pos <= {SW{1'b0}};
                    wo_row  <= {(LOG2_NPH+LOG2_H){1'b0}};
                    wo_sel  <= {WO_SEL_W{1'b0}};
                end
            end else begin

                if (r_hs) begin
                    // position advance: next region or next beat
                    if (region_last) begin
                        region  <= region + 3'd1;
                        sub_pos <= {SW{1'b0}};
                    end else begin
                        sub_pos <= sub_pos + 1'b1;
                    end

                    // dispatch of the beat according to the region
                    case (region)
                        RG_BNSH: begin
                            // BatchNorm shift
                            ld_bn_sh    <= s_axis_w_tdata[5:0];
                            ld_bn_we_sh <= 1'b1;
                        end
                        RG_COEF: begin
                            // biquad coefficients: even beat = low 64 bits,
                            // odd beat = high bits -> write of the whole word
                            if (!sub_pos[0]) begin
                                coef_lo <= s_axis_w_tdata;
                            end else begin
                                cfg_data <= {s_axis_w_tdata[CW-64-1:0], coef_lo};
                                cfg_unit <= sub_pos[LOG2_NB+LOG2_NU:LOG2_NB+1];   // high bits of k
                                cfg_addr <= sub_pos[LOG2_NB:1];                   // low bits of k
                                cfg_we   <= 1'b1;
                            end
                        end
                        RG_D: begin
                            // D and shift of the b of channel sub_pos
                            cfgd_ch   <= sub_pos[LOG2_H-1:0];
                            cfgd_data <= $signed(s_axis_w_tdata[W_D-1:0]);
                            cfgd_bsh  <= s_axis_w_tdata[W_D+W_BSH-1:W_D];
                            cfgd_we   <= 1'b1;
                        end
                        RG_BNW: begin
                            // BatchNorm weight of channel sub_pos
                            ld_bn_addr <= sub_pos[LOG2_H-1:0];
                            ld_bn_w    <= $signed(s_axis_w_tdata[BN_W_W-1:0]);
                            ld_bn_we_w <= 1'b1;
                        end
                        RG_BNB: begin
                            // BatchNorm bias of channel sub_pos
                            ld_bn_addr <= sub_pos[LOG2_H-1:0];
                            ld_bn_b    <= $signed(s_axis_w_tdata[BN_W_B-1:0]);
                            ld_bn_we_b <= 1'b1;
                        end
                        RG_BO: begin
                            // mixing bias, output sub_pos (0..2H-1)
                            ld_bo_addr <= sub_pos[LOG2_H:0];
                            ld_bo_data <= $signed(s_axis_w_tdata[W_MACC-1:0]);
                            ld_bo_we   <= 1'b1;
                        end
                        default: begin
                            // mixing weights: WOBEAT consecutive beats per row,
                            // beat wo_sel goes into bank wo_sel of row wo_row
                            ld_wo_addr <= wo_row;
                            ld_wo_sel  <= wo_sel;
                            ld_wo_data <= s_axis_w_tdata;
                            ld_wo_we   <= 1'b1;
                            // comparison on WO_SEL_W bits: with [3:0] WOBEAT=48 was 0
                            // and the row would have been split every 16 beats
                            if (wo_sel == WOBEAT[WO_SEL_W-1:0] - 1'b1) begin
                                wo_sel <= {WO_SEL_W{1'b0}};
                                wo_row <= wo_row + 1'b1;
                            end else begin
                                wo_sel <= wo_sel + 1'b1;
                            end
                        end
                    endcase

                    // last beat of the layer: end of loading
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
