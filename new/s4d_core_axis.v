`timescale 1ns / 1ps
`default_nettype none
//=====================================================================
// s4d_core_axis  --  TOP del core S4D
//
//   Runs a whole S4D inference with NLAYER layers on a sequence of
//   L timesteps x H channels, using a SINGLE physical layer (s4d_layer)
//   reused NLAYER times (time-multiplexing). The weights of the layer are
//   reloaded from the weight stream before every pass.
//
//   Hierarchy:
//     s4d_core_axis
//       |- s4d_wload_axis   (u_wload)  : receives the weight stream and writes
//       |                                it into the memories of the layer
//       |- s4d_seqbuf       (u_buf)    : sequence buffer, L*H words
//       |- s4d_layer        (u_layer)  : the physical S4D layer
//            |- s4d_affine      (BatchNorm)        [s4d_affine.v]
//            |- s4d_biquad_bank (S4D convolution) [s4d_biquad_bank.v]
//            |     |- s4d_biquad_unit x NUNITS     [s4d_biquad_unit.v]
//            |- s4d_pwl         (GELU)             [s4d_pwl.v]
//            |- s4d_mixing_glu  (linear H->2H + GLU) [s4d_mixing_glu.v]
//                  |- s4d_pwl   (sigmoid)
//
//   Flow of an inference (state machine st):
//
//     ST_IDLE  waits for the first beat of the encoder
//     ST_ENC   writes the L*H beats of the encoder into the seqbuf
//              (address = t*H + channel, because the encoder emits
//              H channels for each timestep)
//     ST_LOAD  u_wload loads the weights of layer m from the s_axis_w stream
//     ST_PASS  reads the seqbuf from address 0 to the last one and sends it
//              into the layer:
//                - passes 0..NLAYER-2: the output of the layer is
//                  written back INTO THE SAME seqbuf ("in place"), always
//                  behind the read pointer;
//                - last pass: the output goes to m_axis_out.
//              At the end of the sequence: m <- m+1 and back to ST_LOAD,
//              or, if it was the last pass, ST_TAIL.
//     ST_TAIL  waits for the last beat to leave
//     ST_DONE  one cycle, then ST_IDLE again
//
//   Memory: the seqbuf is L*H = 131072 words x 16 bits = 2 Mbit
//   (~64 BRAM36).
//=====================================================================

module s4d_core_axis #(
    // ---- dimensions of the model ----
    parameter integer H        = 128,    // channels of the model
    parameter integer LOG2_H   = 7,
    parameter integer L        = 1024,   // sCIFAR (was 4096/Speech Commands)
    parameter integer LOG2_L   = 10,      // sCIFAR (was 12/Speech Commands)
    parameter integer NLAYER   = 6,      // S4D layers (all on the same physical layer)

    // ---- organization of the biquad bank ----
    parameter integer NUNITS   = 32,     // parallel biquad units (was 8; 2026-10-01 ZCU104)
    parameter integer LOG2_NU  = 5,      // (was 3)
    parameter integer MODES    = 32,     // modes (pole pairs) per channel
    parameter integer LOG2_MOD = 5,
    parameter integer NCH      = 4,      // channels served by each unit (NUNITS*NCH = H), was 16
    parameter integer LOG2_NCH = 2,      // (was 4)
    parameter integer LOG2_NB  = LOG2_MOD + LOG2_NCH,   // = 7: NCH*MODES = 128 biquads per unit (was 512)
    // ---- linear mixing ----
    parameter integer NMAC     = 256,  // 2026-10-01 (ZCU104): one MAC per mixing output,
                                        // a single phase (~131 cycles/frame instead of ~641).
                                        // Was 64 (4 phases, ~39ms/img on the Red Pitaya).
    parameter integer LOG2_NPH = 0,    // a single phase: TH/NMAC = 256/256 = 1 (was 2)
    parameter         WMEM_STYLE = "ultra",   // mixing weight memory (URAM on ZCU104)
    // GELU and sigmoid LUTs (read at synthesis): one file per dataset, because
    // the GELU depends on F_SSM (sCIFAR 8, ECG and Speech Commands 9)
    parameter         GELU_FILE  = "gelu_lut_ECG.mem",
    parameter         SIG_FILE   = "sigmoid_lut_ECG.mem",

    // ---- fixed-point formats (W_* = width, F_* = fractional bits) ----
    parameter integer W_ACT    = 16,     // activations between the layers (Q5.11)
    parameter integer F_ACT    = 11,
    parameter integer W_U      = 16,     // input of the biquad bank (BatchNorm output)
    parameter integer F_U      = 11,
    parameter integer W_A      = 24,     // biquad coefficients a1,a2 (denominator)
    parameter integer F_A      = 22,
    parameter integer W_B      = 16,     // biquad coefficients b0,b1 (numerator)
    parameter integer F_B      = 15,
    parameter integer W_S      = 32,     // biquad state and output y of each mode
    parameter integer F_Y      = 19,
    parameter integer W_D      = 16,     // coefficient D (skip connection u*D)
    parameter integer F_D      = 13,
    parameter integer W_BSH    = 5,      // per-channel shift of the b coefficients (common exponent)
    parameter integer W_BACC   = 40,     // accumulator of the sum over the modes
    parameter integer W_SSM    = 16,     // output of the biquad bank / GELU input
    parameter integer F_SSM    = 8,       // from the sCIFAR quantization
                                          // (quant_report.json);
                                          // re-check if the model is retrained/
                                          // requantized
    parameter integer W_W      = 12,     // weights of the linear mixing
    parameter integer F_W      = 11,
    parameter integer W_MACC   = 32,     // accumulators of the mixing (and mixing bias)
    parameter integer F_SIGI   = 11,     // sigmoid input (GLU gate)
    parameter integer F_SIGO   = 14,     // sigmoid output

    // ---- BatchNorm (s4d_affine) ----
    parameter integer BN_W_W   = 16,     // BN weight (scale)
    parameter integer BN_W_B   = 32,     // BN bias
    parameter integer BN_W_ACC = 34,     // BN accumulator
    parameter integer FIFO_LOG2 = 10,    // depth of the residual FIFO = 2^10

    parameter integer AXI_DW   = 64      // width of the weight stream
)(
    input  wire                    clk,
    input  wire                    rst_n,            // asynchronous active-low reset

    // ---- input sequence from the encoder: L*H beats, tlast on the last one ----
    input  wire                    s_axis_enc_tvalid,
    output wire                    s_axis_enc_tready,
    input  wire signed [W_ACT-1:0] s_axis_enc_tdata,
    input  wire                    s_axis_enc_tlast,
    // weight stream: 6 consecutive layers, in order
    input  wire                    s_axis_w_tvalid,
    output wire                    s_axis_w_tready,
    input  wire [AXI_DW-1:0]       s_axis_w_tdata,
    input  wire                    s_axis_w_tlast,

    // ---- output of the last layer: L*H beats, tlast on the last one ----
    output wire                    m_axis_out_tvalid,
    input  wire                    m_axis_out_tready,
    output wire signed [W_ACT-1:0] m_axis_out_tdata,
    output wire                    m_axis_out_tlast,

    // Status flags, sticky. They used to be internal reg/wire: an error that
    // nobody can read protects against nothing.
    //   [0] err_out_ovr  output overwritten (must NEVER fire now)
    //   [1] err_enc_len  wrong length of the input sequence
    //   [2] err_ovw      the write overtook the read in the seqbuf
    //   [3] layer_ovf    arithmetic overflow in the biquad bank
    //   [4] err_bank     ready broken between bank and mixing (sentinel)
    output wire [4:0]              status_err,

    // Realignment of the weight streamer (s4d_weightrom_axis).
    // wload_start is THE SAME wire that starts u_wload: the streamer jumps
    // to the start of layer wload_layer in the same cycle, before
    // s4d_wload raises tready (next cycle).
    output wire                    wload_start,   // pulse: the loading of a layer starts
    output wire [2:0]              wload_layer    // layer about to be loaded (0..NLAYER-1)
);

    localparam integer NWORDS = L * H;              // words in the seqbuf (131072)
    localparam integer AW     = LOG2_L + LOG2_H;    // address bits of the seqbuf (17)

    localparam [AW-1:0] ADDR_LAST = NWORDS - 1;     // last address of the sequence

    //=================================================================
    // LOCAL RESET (timing 2026-10-02)
    //   Before, rst_n (peripheral_aresetn of the Processor System Reset) went
    //   directly to tens of thousands of flip-flops: Vivado replicated the
    //   output registers of the reset IP and those paths were among the
    //   slowest (measured slack 0.47 ns at 177.8 MHz).
    //   Now rst_n enters only 4 reset "bridges", one per block:
    //     - immediate ASYNCHRONOUS assertion (as before: rst_n low
    //       clears everything at once);
    //     - SYNCHRONOUS release, 2 cycles after rst_n rises, the same
    //       for all the bridges (so all the blocks restart together,
    //       and together with the weight streamer, which has the same bridge).
    //   keep: prevents synthesis from merging the 4 bridges into one.
    //=================================================================
    (* keep = "true" *) reg [1:0] rst_bridge_top;     // logic of this file
    (* keep = "true" *) reg [1:0] rst_bridge_wload;   // u_wload
    (* keep = "true" *) reg [1:0] rst_bridge_buf;     // u_buf (seqbuf)
    (* keep = "true" *) reg [1:0] rst_bridge_layer;   // u_layer

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rst_bridge_top   <= 2'b00;
            rst_bridge_wload <= 2'b00;
            rst_bridge_buf   <= 2'b00;
            rst_bridge_layer <= 2'b00;
        end else begin
            rst_bridge_top   <= {rst_bridge_top[0],   1'b1};
            rst_bridge_wload <= {rst_bridge_wload[0], 1'b1};
            rst_bridge_buf   <= {rst_bridge_buf[0],   1'b1};
            rst_bridge_layer <= {rst_bridge_layer[0], 1'b1};
        end
    end

    wire rst_top_n   = rst_bridge_top[1];     // local reset, active low
    wire rst_wload_n = rst_bridge_wload[1];
    wire rst_buf_n   = rst_bridge_buf[1];
    wire rst_layer_n = rst_bridge_layer[1];

    // states of the main machine (see the header)
    localparam [2:0] ST_IDLE = 3'd0,
                     ST_ENC  = 3'd1,
                     ST_LOAD = 3'd2,
                     ST_PASS = 3'd3,
                     ST_TAIL = 3'd4,
                     ST_DONE = 3'd5;

   //(* mark_debug = "true" *) reg [2:0]     st;
   //(* mark_debug = "true" *) reg [2:0]     m;

   reg [2:0]     st;                 // current state
   reg [2:0]     m;                  // index of the pass/layer in progress (0..NLAYER-1)
   reg [AW-1:0]  rptr, wptr;         // read / write pointers in the seqbuf
   reg           rd_run;             // 1 = words remain to be read in the pass

   //(* mark_debug = "true" *) reg [2:0]     st;
   //(* mark_debug = "true" *) reg [2:0]     m;
   //(* mark_debug = "true" *) reg [AW-1:0]  rptr, wptr;
   //(* mark_debug = "true" *) reg           rd_run;



    // 1 = we are in the last pass: the output of the layer goes to m_axis_out
    wire last_pass = (m == NLAYER[2:0] - 3'd1);
    //(* mark_debug = "true" *) wire                     wl_busy, wl_done;

    //-----------------------------------------------------------------
    // Signals between u_wload and u_layer (writing of the weights into the memories)
    //-----------------------------------------------------------------
    wire                     wl_busy, wl_done;   // loading in progress / finished (pulse)
    reg                      wl_start;           // pulse: load the weights of the next layer
    // biquad coefficients (a1,a2,b0,b1) -> s4d_biquad_bank
    wire                     cfg_we;
    wire [LOG2_NU-1:0]       cfg_unit;
    wire [LOG2_NB-1:0]       cfg_addr;
    wire [2*W_A+2*W_B-1:0]   cfg_data;
    // coefficient D and shift of the b, per channel -> s4d_biquad_bank
    wire                     cfgd_we;
    wire [LOG2_H-1:0]        cfgd_ch;
    wire signed [W_D-1:0]    cfgd_data;
    wire [W_BSH-1:0]         cfgd_bsh;
    // BatchNorm: shift, weights and biases per channel -> s4d_affine
    wire                     ld_bn_we_sh;
    wire [5:0]               ld_bn_sh;
    wire                     ld_bn_we_w, ld_bn_we_b;
    wire [LOG2_H-1:0]        ld_bn_addr;
    wire signed [BN_W_W-1:0] ld_bn_w;
    wire signed [BN_W_B-1:0] ld_bn_b;
    // linear mixing: weights (in 64-bit rows) and biases -> s4d_mixing_glu
    wire                     ld_wo_we, ld_bo_we;
    wire [LOG2_NPH+LOG2_H-1:0] ld_wo_addr;
    wire [5:0]               ld_wo_sel;   // was [3:0]: 48 banks of 64 bits
    wire [AXI_DW-1:0]        ld_wo_data;
    wire [LOG2_H:0]          ld_bo_addr;
    wire signed [W_MACC-1:0] ld_bo_data;

    //=================================================================
    // Weight loader: at every wl_start it consumes from the s_axis_w
    // stream the beats of ONE layer and dispatches them to the memories of the layer
    //=================================================================
    s4d_wload_axis #(
        .H(H), .LOG2_H(LOG2_H), .NUNITS(NUNITS), .LOG2_NU(LOG2_NU),
        .LOG2_NB(LOG2_NB), .NMAC(NMAC), .LOG2_NPH(LOG2_NPH),
        .W_A(W_A), .W_B(W_B), .W_D(W_D), .W_BSH(W_BSH),
        .BN_W_W(BN_W_W), .BN_W_B(BN_W_B), .W_MACC(W_MACC), .W_W(W_W), .AXI_DW(AXI_DW),
        .WO_SEL_W(6)                    // 48 beats per row of the mixing weights
    ) u_wload (
        .clk(clk), .rst_n(rst_wload_n),        // local reset (see LOCAL RESET)
        .start(wl_start),
        .busy(wl_busy), .done(wl_done),.s_axis_w_tvalid(s_axis_w_tvalid),
        .s_axis_w_tready(s_axis_w_tready),
        .s_axis_w_tdata (s_axis_w_tdata),
        .s_axis_w_tlast (s_axis_w_tlast),
        .cfg_we(cfg_we), .cfg_unit(cfg_unit), .cfg_addr(cfg_addr),
        .cfg_data(cfg_data),
        .cfgd_we(cfgd_we), .cfgd_ch(cfgd_ch), .cfgd_data(cfgd_data),
        .cfgd_bsh(cfgd_bsh),
        .ld_bn_we_sh(ld_bn_we_sh), .ld_bn_sh(ld_bn_sh),
        .ld_bn_we_w(ld_bn_we_w), .ld_bn_we_b(ld_bn_we_b),
        .ld_bn_addr(ld_bn_addr), .ld_bn_w(ld_bn_w), .ld_bn_b(ld_bn_b),
        .ld_wo_we(ld_wo_we), .ld_wo_addr(ld_wo_addr), .ld_wo_sel(ld_wo_sel),
        .ld_wo_data(ld_wo_data),
        .ld_bo_we(ld_bo_we), .ld_bo_addr(ld_bo_addr), .ld_bo_data(ld_bo_data)
    );

    wire en = 1'b1;                 // the core starts when the data arrives
    reg  img_done;                  // internal use only
    //(* mark_debug = "true" *) wire layer_ovf, err_bank, err_ovw;
    wire layer_ovf, err_bank, err_ovw;   // error flags from the submodules
    wire lay_out_ready;                  // ready towards the output of the layer
    reg  err_out_ovr, err_enc_len;       // error flags generated here

    // beat accepted from the encoder; accepted only during ST_ENC
    // (s_axis_enc_tready depends only on the state register)
    wire enc_beat = s_axis_enc_tvalid && s_axis_enc_tready;
    assign s_axis_enc_tready = (st == ST_ENC);

    //-----------------------------------------------------------------
    // Ports of the seqbuf
    //-----------------------------------------------------------------
    wire                    buf_wr_en;
    wire [AW-1:0]           buf_wr_addr;
    wire signed [W_ACT-1:0] buf_wr_data;
    wire                    buf_rd_en;
    wire signed [W_ACT-1:0] buf_rd_data;   // available RD_LAT cycles after buf_rd_en

    //-----------------------------------------------------------------
    // Read of the seqbuf towards the layer: 1 word per cycle.
    //   do_rd       : launches a read at address rptr
    //   rd_vpipe/   : valid and "last word" flag along the latency of the
    //   rd_lpipe      seqbuf (RD_LAT cycles)
    //   rdq_*       : queue of RDQ_DEPTH words that presents the data to the layer
    //   rd_credits  : reads in flight + words in queue; a read is
    //                 launched only if a slot in the queue is guaranteed
    //                 -> the queue never overflows, even if the layer stalls.
    //   Depth >= RD_LAT+2: the full loop read -> queue -> layer
    //   takes RD_LAT+1 cycles, so in steady state the reads never stop.
    //   (2026-10-05: RD_LAT 2 -> 3 for the BRAM cascade of the seqbuf, queue
    //   4 -> 8 slots.)
    //   (Before: rd_inflight + sk register, one read every 2 cycles, which
    //   was enough only while the biquad bank took 512 cycles/frame.)
    //-----------------------------------------------------------------
    localparam integer RD_LAT    = 3;     // latency of the seqbuf (BRAM output + mux + register)
    localparam integer RDQ_DEPTH = 8;     // slots of the queue (>= RD_LAT + 2)
    localparam integer RDQ_IDX_W = 3;     // bits of the queue indices (8 slots)
    localparam integer RDQ_CNT_W = 4;     // bits of the counters (0..8)

    initial if ((RD_LAT < 2) || (RDQ_DEPTH < RD_LAT + 2) || ((1 << RDQ_IDX_W) != RDQ_DEPTH)) begin
        $display("s4d_core_axis: inconsistent seqbuf read parameters (RD_LAT=%0d, RDQ_DEPTH=%0d)", RD_LAT, RDQ_DEPTH);
        $finish;
    end

    reg  [RD_LAT-1:0]       rd_vpipe;     // reads in flight (bit RD_LAT-1 = arrives now)
    reg  [RD_LAT-1:0]       rd_lpipe;     // ... and whether they are the last word of the sequence

    reg  signed [W_ACT-1:0] rdq_data [0:RDQ_DEPTH-1];
    reg                     rdq_last [0:RDQ_DEPTH-1];
    reg  [RDQ_IDX_W-1:0]    rdq_wr_idx, rdq_rd_idx;   // queue pointers
    reg  [RDQ_CNT_W-1:0]    rdq_count;                // words in the queue (0..RDQ_DEPTH)
    reg  [RDQ_CNT_W-1:0]    rd_credits;               // in flight + in queue (0..RDQ_DEPTH)

    wire lay_in_ready;                                // the layer accepts a sample

    wire rdq_valid  = (rdq_count != {RDQ_CNT_W{1'b0}});   // there is a sample for the layer
    wire rdq_pop    = rdq_valid && lay_in_ready;      // the layer takes it
    wire rd_arrive  = rd_vpipe[RD_LAT-1];             // the data of a read arrives now

    // read if words remain and a slot in the queue is guaranteed
    wire do_rd = rd_run && (rd_credits < RDQ_DEPTH);

    assign buf_rd_en = do_rd;

    //=================================================================
    // Sequence buffer (Block RAM, dual port, read in RD_LAT cycles)
    //   chk_en enables the check "the write must not reach
    //   the read" only during the passes
    //=================================================================
    s4d_seqbuf #(.W(W_ACT), .NWORDS(NWORDS), .AW(AW), .RD_LAT(RD_LAT)) u_buf (
        .clk(clk), .rst_n(rst_buf_n),        // reset locale (vedi RESET LOCALE)
        .wr_en(buf_wr_en), .wr_addr(buf_wr_addr), .wr_data(buf_wr_data),
        .rd_en(buf_rd_en), .rd_addr(rptr), .rd_data(buf_rd_data),
        .chk_en((st == ST_PASS) && rd_run), .err_ovw(err_ovw)
    );

    // management of the reads in flight and of the queue
    always @(posedge clk or negedge rst_top_n) begin
        if (!rst_top_n) begin
            rd_vpipe   <= {RD_LAT{1'b0}};
            rd_lpipe   <= {RD_LAT{1'b0}};
            rdq_wr_idx <= {RDQ_IDX_W{1'b0}};
            rdq_rd_idx <= {RDQ_IDX_W{1'b0}};
            rdq_count  <= {RDQ_CNT_W{1'b0}};
            rd_credits <= {RDQ_CNT_W{1'b0}};
        end else if (st != ST_PASS) begin
            // outside the passes no pending reads (as before)
            rd_vpipe   <= {RD_LAT{1'b0}};
            rd_lpipe   <= {RD_LAT{1'b0}};
            rdq_wr_idx <= {RDQ_IDX_W{1'b0}};
            rdq_rd_idx <= {RDQ_IDX_W{1'b0}};
            rdq_count  <= {RDQ_CNT_W{1'b0}};
            rd_credits <= {RDQ_CNT_W{1'b0}};
        end else begin
            // valid and "last word" advance along the latency of the seqbuf
            rd_vpipe <= {rd_vpipe[RD_LAT-2:0], do_rd};
            rd_lpipe <= {rd_lpipe[RD_LAT-2:0], do_rd && (rptr == ADDR_LAST)};

            // the arrived data enters the queue
            if (rd_arrive) rdq_wr_idx <= rdq_wr_idx + 1'b1;
            // the layer takes the head
            if (rdq_pop)   rdq_rd_idx <= rdq_rd_idx + 1'b1;

            rdq_count  <= rdq_count  + (rd_arrive ? 1'b1 : 1'b0) - (rdq_pop ? 1'b1 : 1'b0);
            rd_credits <= rd_credits + (do_rd     ? 1'b1 : 1'b0) - (rdq_pop ? 1'b1 : 1'b0);
        end
    end

    // queue data (no reset)
    always @(posedge clk)
        if (rd_arrive) begin
            rdq_data[rdq_wr_idx] <= buf_rd_data;
            rdq_last[rdq_wr_idx] <= rd_lpipe[RD_LAT-1];
        end

    // towards the layer: head of the queue (same names as before, now wires)
    wire                    sk_valid = rdq_valid;
    wire signed [W_ACT-1:0] sk_data  = rdq_data[rdq_rd_idx];
    wire                    sk_last  = rdq_last[rdq_rd_idx];

    // output of the layer
    wire                    lay_valid;
    wire signed [W_ACT-1:0] lay_data;
    wire                    lay_last_seq;

    //=================================================================
    // The physical S4D layer, reused for all the NLAYER passes.
    //   LOADABLE=1: weights written at runtime by u_wload (no .mem)
    //   BN_SH=0   : the BatchNorm shift also comes from the stream
    //   The GELU and sigmoid LUTs are the same for all the layers: .mem
    //=================================================================
    s4d_layer #(
        .H(H), .LOG2_H(LOG2_H), .NUNITS(NUNITS), .LOG2_NU(LOG2_NU),
        .MODES(MODES), .LOG2_MOD(LOG2_MOD), .NCH(NCH), .LOG2_NCH(LOG2_NCH),
        .LOG2_NB(LOG2_NB), .NMAC(NMAC), .LOG2_NPH(LOG2_NPH),
        .W_ACT(W_ACT), .F_ACT(F_ACT),
        .W_U(W_U), .F_U(F_U), .W_A(W_A), .F_A(F_A), .W_B(W_B), .F_B(F_B),
        .W_S(W_S), .F_Y(F_Y), .W_D(W_D), .F_D(F_D), .W_BSH(W_BSH),
        .W_BACC(W_BACC), .W_SSM(W_SSM), .F_SSM(F_SSM),
        .W_W(W_W), .F_W(F_W), .W_MACC(W_MACC),
        .F_SIGI(F_SIGI), .F_SIGO(F_SIGO),
        .FIFO_LOG2(FIFO_LOG2),
        .LOADABLE(1), .LD_W(AXI_DW), .BN_SH(0),
        .WMEM_STYLE(WMEM_STYLE),
        .BN_W_W(BN_W_W), .BN_W_B(BN_W_B), .BN_W_ACC(BN_W_ACC),
        .GELU_FILE(GELU_FILE),
        .SIG_FILE (SIG_FILE)
    ) u_layer (
        .clk(clk), .rst_n(rst_layer_n),        // reset locale (vedi RESET LOCALE)
        .cfg_we(cfg_we), .cfg_unit(cfg_unit), .cfg_addr(cfg_addr),
        .cfg_data(cfg_data),
        .cfgd_we(cfgd_we), .cfgd_ch(cfgd_ch), .cfgd_data(cfgd_data),
        .cfgd_bsh(cfgd_bsh),
        .ld_bn_we_sh(ld_bn_we_sh), .ld_bn_sh(ld_bn_sh),
        .ld_bn_we_w(ld_bn_we_w), .ld_bn_we_b(ld_bn_we_b),
        .ld_bn_addr(ld_bn_addr), .ld_bn_w(ld_bn_w), .ld_bn_b(ld_bn_b),
        .ld_wo_we(ld_wo_we), .ld_wo_addr(ld_wo_addr), .ld_wo_sel(ld_wo_sel),
        .ld_wo_data(ld_wo_data),
        .ld_bo_we(ld_bo_we), .ld_bo_addr(ld_bo_addr), .ld_bo_data(ld_bo_data),
        .in_valid(sk_valid), .in_ready(lay_in_ready), .in_data(sk_data),
        .in_last_seq(sk_valid && sk_last),
        .out_ready(lay_out_ready),
        .out_valid(lay_valid), .out_data(lay_data),
        .out_last_ch(), .out_last_seq(lay_last_seq),
        .ovf(layer_ovf), .err_bank_ovr(err_bank)
    );

    //-----------------------------------------------------------------
    // Write into the seqbuf
    //   ST_ENC : data from the encoder
    //   ST_PASS: output of the layer (not in the last pass), written
    //            "in place" at wptr, which always stays behind rptr
    //-----------------------------------------------------------------
    assign buf_wr_en   = (st == ST_ENC)  ? enc_beat
                       : (st == ST_PASS) ? (lay_valid && !last_pass)
                       : 1'b0;
    assign buf_wr_addr = wptr;
    assign buf_wr_data = (st == ST_ENC) ? s_axis_enc_tdata : lay_data;

    //(* mark_debug = "true" *) wire dec_fire = lay_valid && last_pass && (st == ST_PASS);
    // in the last pass the layer has valid data for the output
    wire dec_fire = lay_valid && last_pass && (st == ST_PASS);

    // ------------------------------------------------------------------
    // Output SKID BUFFER (2 positions).
    //
    // It replaces the single register, which overwrote the data when
    // the consumer stalled (silent loss, signalled only by
    // err_out_ovr).
    //
    // sk_ready is REGISTERED (= !skid_v), not combinational from tready:
    // it avoids lengthening the critical path from the DMA to the layer.
    // Full throughput when there is no stall: with skid_v=0 the beat goes
    // directly to the output, 1 per cycle.
    // ------------------------------------------------------------------

    //(* mark_debug = "true" *) reg                    out_v,  skid_v;
    reg                    out_v,  skid_v;   // valid: output register / spare register
    reg signed [W_ACT-1:0] out_d,  skid_d;   // data
    reg                    out_l,  skid_l;   // tlast

    // beats from the layer are accepted while the spare register is empty
    wire sk_out_ready = !skid_v;

    // The layer is stalled ONLY in the last pass: in the others the output
    // ends up in the seqbuf, which always accepts.
    assign lay_out_ready = last_pass ? sk_out_ready : 1'b1;

    assign m_axis_out_tvalid = out_v;
    assign m_axis_out_tdata  = out_d;
    assign m_axis_out_tlast  = out_l;

    always @(posedge clk or negedge rst_top_n) begin
        if (!rst_top_n) begin
            out_v       <= 1'b0;
            out_d       <= {W_ACT{1'b0}};
            out_l       <= 1'b0;
            skid_v      <= 1'b0;
            skid_d      <= {W_ACT{1'b0}};
            skid_l      <= 1'b0;
            err_out_ovr <= 1'b0;
            err_enc_len <= 1'b0;
        end else begin
            if (!out_v || m_axis_out_tready) begin
                // the output stage frees up: it picks from the skid if full,
                // otherwise directly from the input
                if (skid_v) begin
                    out_v  <= 1'b1;
                    out_d  <= skid_d;
                    out_l  <= skid_l;
                    skid_v <= 1'b0;
                end else begin
                    out_v  <= dec_fire;
                    out_d  <= lay_data;
                    out_l  <= lay_last_seq;
                end
            end else if (dec_fire && !skid_v) begin
                // output blocked: the incoming beat goes into the skid
                skid_v <= 1'b1;
                skid_d <= lay_data;
                skid_l <= lay_last_seq;
            end

            // Overwrite sentinel.
            // WARNING: dec_fire is a VALID, not a fire: the layer
            // keeps out_valid high for the whole duration of the stall, as
            // AXI-Stream requires. The beat is really ACCEPTED only with
            // lay_out_ready high, so the loss condition is
            // "accepted while both stages are full".
            // With lay_out_ready = !skid_v this is structurally
            // impossible: if it fires, someone broke the ready.
            if (dec_fire && lay_out_ready && skid_v
                         && out_v && !m_axis_out_tready)
                err_out_ovr <= 1'b1;

            // tlast of the encoder must arrive exactly on the last
            // address (L*H-th beat): otherwise the sequence has the wrong length
            if (enc_beat && (s_axis_enc_tlast != (wptr == ADDR_LAST)))
                err_enc_len <= 1'b1;
        end
    end

    assign status_err = {err_bank, layer_ovf, err_ovw,
                         err_enc_len, err_out_ovr};

    // Outputs towards the weight streamer.
    // wl_start is a register (1-cycle pulse); m is updated
    // on the same edge on which wl_start rises (ST_PASS: m <= m+1,
    // wl_start <= 1; ST_ENC: m is already 0), so in the cycle in which
    // wl_start is 1, m already indicates the layer to load.
    assign wload_start = wl_start;
    assign wload_layer = m;

    //=================================================================
    // Main state machine
    //=================================================================
    always @(posedge clk or negedge rst_top_n) begin
        if (!rst_top_n) begin
            st       <= ST_IDLE;
            m        <= 3'd0;
            rptr     <= {AW{1'b0}};
            wptr     <= {AW{1'b0}};
            rd_run   <= 1'b0;
            wl_start <= 1'b0;
            img_done <= 1'b0;
        end else begin
            // one-cycle pulses
            wl_start <= 1'b0;
            img_done <= 1'b0;

            case (st)

                // wait: restarts from layer 0 as soon as the encoder sends data
                ST_IDLE: begin
                    wptr <= {AW{1'b0}};
                    m    <= 3'd0;
                    if (en && s_axis_enc_tvalid) st <= ST_ENC;
                end

                // write of the encoder sequence into the seqbuf;
                // at the last beat the loading of the weights of layer 0 starts
                ST_ENC: begin
                    if (enc_beat) begin
                        if (wptr == ADDR_LAST) begin
                            wptr     <= {AW{1'b0}};
                            wl_start <= 1'b1;
                            st       <= ST_LOAD;
                        end else begin
                            wptr <= wptr + 1'b1;
                        end
                    end
                end

                // wait for the weight loading, then the pass is prepared
                ST_LOAD: begin
                    if (wl_done) begin
                        rptr   <= {AW{1'b0}};
                        wptr   <= {AW{1'b0}};
                        rd_run <= 1'b1;
                        st     <= ST_PASS;
                    end
                end

                // one pass of the layer over the whole sequence
                ST_PASS: begin
                    // read advance
                    if (do_rd) begin
                        if (rptr == ADDR_LAST) rd_run <= 1'b0;
                        else                   rptr <= rptr + 1'b1;
                    end

                    // in-place write advance (not in the last pass)
                    if (lay_valid && !last_pass) begin
                        if (wptr != ADDR_LAST) wptr <= wptr + 1'b1;
                    end

                     // the layer emitted the last sample of the sequence
						if (lay_last_seq && lay_out_ready) begin
							if (last_pass) begin
								st <= ST_TAIL;
							end else begin
								m        <= m + 3'd1;      // next layer
								wl_start <= 1'b1;          // load its weights
								st       <= ST_LOAD;
							end
						end
					end

                // wait for the last beat to leave the output register
                ST_TAIL: if (!out_v) st <= ST_DONE;

                ST_DONE: begin
                    img_done <= 1'b1;
                    st       <= ST_IDLE;
                end

                default: st <= ST_IDLE;

            endcase
        end
    end

endmodule

`default_nettype wire
