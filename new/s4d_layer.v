`timescale 1ns / 1ps
`default_nettype none
//=====================================================================
// s4d_layer  --  one complete S4D layer
//
//   It processes the sequence one timestep at a time; each timestep is a
//   "frame" of H samples (channel 0..H-1). Processing chain:
//
//     x --+--> s4d_affine (BatchNorm) --> s4d_biquad_bank (S4D + D*u)
//         |        u_norm                      u_bank
//         |                                       |
//         |                                       v
//         |                               s4d_pwl (GELU)
//         |                                   u_gelu
//         |                                       |
//         |                                       v
//         |                               s4d_mixing_glu (linear H->2H, GLU)
//         |                                   u_mix
//         |                                       |
//         +--> residual FIFO ----------------> ( + ) --> saturation --> y
//
//     y[t][ch] = sat16( x[t][ch] + GLU( Mix( GELU( S4D( BN(x) ) ) ) )[t][ch] )
//
//   The residual branch is a FIFO that keeps x until the main branch
//   produces the result of the same sample (latency of several
//   frames: double buffers in the biquad bank and in the mixing).
//
//   2026-10-15 (fan-out of the output stall): the GLU branch and the
//   residual run freely; the output stall acts only on the OUTPUT TAIL
//   QUEUE (TAIL_DEPTH slots, with a head register) and, through the
//   credits (tail_pop), on the emission from the head of the mixing
//   snapshot. Before, out_ready (oen) directly enabled ~800 registers
//   here and in the mixing, plus ~8000 of the snapshot.
//
//   All the default parameters are overridden by the top
//   (s4d_core_axis): the values below are only fallbacks.
//=====================================================================

module s4d_layer #(
    // ---- dimensions (see s4d_core_axis) ----
    parameter integer H        = 128,
    parameter integer LOG2_H   = 7,
    parameter integer NUNITS   = 8,
    parameter integer LOG2_NU  = 3,
    parameter integer MODES    = 32,
    parameter integer LOG2_MOD = 5,
    parameter integer NCH      = 16,
    parameter integer LOG2_NCH = 4,
    parameter integer LOG2_NB  = LOG2_MOD + LOG2_NCH,
    parameter integer NMAC     = 256,       // was 64: one MAC per output (single-phase mixing)
    parameter integer LOG2_NPH = 0,         // was 2: a single phase -> 0 bits
    parameter         WMEM_STYLE = "ultra", // memory of the mixing weights: "ultra" | "block" | "distributed"

    // ---- fixed-point formats (see s4d_core_axis) ----
    parameter integer W_ACT   = 16,
    parameter integer F_ACT   = 11,
    parameter integer W_U     = 16,
    parameter integer F_U     = 11,
    parameter integer W_A     = 24,
    parameter integer F_A     = 22,
    parameter integer W_B     = 16,
    parameter integer F_B     = 15,
    parameter integer W_S     = 32,
    parameter integer F_Y     = 19,
    parameter integer W_D     = 16,
    parameter integer F_D     = 14,
    parameter integer W_BSH   = 5,
    parameter integer W_BACC  = 40,
    parameter integer W_SSM   = 16,
    parameter integer F_SSM   = 11,
    parameter integer W_W     = 12,
    parameter integer F_W     = 11,
    parameter integer W_MACC  = 32,
    parameter integer F_SIGI  = 11,
    parameter integer F_SIGO  = 14,
    parameter integer FIFO_LOG2 = 10,   // depth of the residual FIFO = 2^FIFO_LOG2
    parameter integer TAIL_LOG2 = 5,    // output tail queue: 2^TAIL_LOG2 slots (mixing credits)

    parameter integer LOADABLE = 0,     // 1 = weights written at runtime by the ld_* ports
    parameter integer LD_W    = 64,     // width of the weight-load words
    parameter integer BN_SH       = 11, // fixed BatchNorm shift (if not loadable)
    parameter integer BN_W_W      = 16,
    parameter integer BN_W_B      = 32,
    parameter integer BN_W_ACC    = 34,

    // s4d_layer is ONE physical layer reused for NLAYER roles (time-multiplexed):
    // there is no "the" right file here, whoever instantiates it always overrides. The
    // defaults point to layer 0 only because it is a file that really exists,
    // not an invented placeholder.
    parameter BN_W_FILE  = "l0_bn_w.mem",
    parameter BN_B_FILE  = "l0_bn_b.mem",
    parameter GELU_FILE  = "gelu_lut.mem",
    parameter SIG_FILE   = "sigmoid_lut.mem",
    parameter WOUT_FILE  = "l0_s4d_wout.mem",
    parameter BOUT_FILE  = "l0_s4d_bout.mem"
)(
    input  wire                    clk,
    input  wire                    rst_n,

    // ---- load of the biquad coefficients (from s4d_wload_axis) ----
    input  wire                    cfg_we,
    input  wire [LOG2_NU-1:0]      cfg_unit,
    input  wire [LOG2_NB-1:0]      cfg_addr,
    input  wire [2*W_A+2*W_B-1:0]  cfg_data,
    input  wire                    cfgd_we,
    input  wire [LOG2_H-1:0]       cfgd_ch,
    input  wire signed [W_D-1:0]   cfgd_data,
    input  wire [W_BSH-1:0]        cfgd_bsh,

    // ---- BatchNorm load ----
    input  wire                    ld_bn_we_sh,
    input  wire [5:0]              ld_bn_sh,
    input  wire                    ld_bn_we_w,
    input  wire                    ld_bn_we_b,
    input  wire [LOG2_H-1:0]       ld_bn_addr,
    input  wire signed [BN_W_W-1:0] ld_bn_w,
    input  wire signed [BN_W_B-1:0] ld_bn_b,

    // ---- mixing load ----
    input  wire                    ld_wo_we,
    input  wire [LOG2_NPH+LOG2_H-1:0] ld_wo_addr,
    input  wire [5:0]              ld_wo_sel,   // was [3:0]: 48 banks of 64 bits per row
    input  wire [LD_W-1:0]         ld_wo_data,
    input  wire                    ld_bo_we,
    input  wire [LOG2_H:0]         ld_bo_addr,
    input  wire signed [W_MACC-1:0] ld_bo_data,

    // ---- input: one sample per beat, frame of H channels ----
    input  wire                    in_valid,
    output wire                    in_ready,
    input  wire signed [W_ACT-1:0] in_data,
    input  wire                    in_last_seq,  // last sample of the sequence

    // ---- output ----
    input  wire                    out_ready,
    output reg                     out_valid,
    output reg  signed [W_ACT-1:0] out_data,
    output reg                     out_last_ch,  // last channel of the frame
    output reg                     out_last_seq, // last sample of the sequence

    // ---- diagnostics ----
    output wire                    ovf,          // overflow in the biquad bank (sticky)
    output reg                     err_bank_ovr  // sentinel on the GELU -> mixing ready (sticky)
);

    // ------------------------------------------------------------------
    // CREDIT-BASED BACKPRESSURE
    // The mixing emits a sample from the head of its snapshot only if it
    // has a credit; from there the sample goes through GLU and residual without
    // stopping and ends up in the output tail queue (TAIL_DEPTH slots). Each
    // sample that leaves the queue (tail_pop) gives a credit back.
    // The pop of the residual FIFO happens when the GLU result
    // arrives (gl_valid): GLU and residual advance together by construction,
    // so the residual cannot become misaligned.
    // ------------------------------------------------------------------
    localparam integer FDEPTH     = (1 << FIFO_LOG2);   // slots of the residual FIFO (1024)
    localparam integer TAIL_DEPTH = (1 << TAIL_LOG2);   // slots of the output tail queue (32)

    wire tail_pop;                                      // a sample leaves the output tail queue

    //-----------------------------------------------------------------
    // Residual FIFO: it keeps the input sample x (and the end-of-sequence
    // flag) until the main branch produces its result. Asynchronous
    // read (fifo[frd]) -> implemented in LUTRAM.
    //-----------------------------------------------------------------
    reg signed [W_ACT-1:0] fifo    [0:FDEPTH-1];   // samples x
    reg                    fifo_ls [0:FDEPTH-1];   // in_last_seq flag of each sample
    reg [FIFO_LOG2-1:0]    fwr, frd;               // write / read pointers
    reg [FIFO_LOG2:0]      fcnt;                   // samples present

    // inputs are accepted only if more than H free slots remain
    // (margin of one whole frame)
    wire fifo_ready = (fcnt < FDEPTH - H);

    // BatchNorm output
    wire                   ln_valid;
    wire signed [W_U-1:0]  ln_data;
    wire                   ln_last_ch, ln_last_seq;
    wire                   ln_ready;     // the BatchNorm accepts an input
    wire                   bk_ready;     // the biquad bank accepts an input

    //=================================================================
    // Per-channel BatchNorm: y = (w[ch]*x + b[ch]) >> shift
    // (module s4d_affine, file s4d_affine.v)
    //=================================================================
    s4d_affine #(
        .H(H), .LOG2_H(LOG2_H),
        .W_IN(W_ACT), .W_OUT(W_U),
        .W_W(BN_W_W), .W_B(BN_W_B), .W_ACC(BN_W_ACC), .SH(BN_SH),
        .LOADABLE(LOADABLE),
        .W_FILE(BN_W_FILE), .B_FILE(BN_B_FILE)
    ) u_norm (
        .clk(clk), .rst_n(rst_n),
        .ld_we_w(ld_bn_we_w), .ld_we_b(ld_bn_we_b),
        .ld_addr(ld_bn_addr), .ld_w(ld_bn_w), .ld_b(ld_bn_b),
        .ld_we_sh(ld_bn_we_sh), .ld_sh(ld_bn_sh),
        .in_valid(in_valid && fifo_ready), .in_ready(ln_ready),
        .in_data(in_data), .in_last_seq(in_last_seq),
        .out_ready(bk_ready),
        .out_valid(ln_valid), .out_data(ln_data),
        .out_last_ch(ln_last_ch), .out_last_seq(ln_last_seq)
    );

    // biquad bank output
    wire                     bk_valid;
    wire signed [W_SSM-1:0]  bk_data;
    wire [LOG2_H-1:0]        bk_ch;
    wire                     bk_last;     // last channel of the frame

    wire ge_in_ready;                     // the GELU accepts an input
    wire mix_ready;                       // the mixing accepts an input

    //=================================================================
    // S4D convolution: for each channel the sum of MODES biquads + D*u
    //=================================================================
    s4d_biquad_bank #(
        .H(H), .LOG2_H(LOG2_H), .NUNITS(NUNITS), .LOG2_NU(LOG2_NU),
        .MODES(MODES), .LOG2_MOD(LOG2_MOD),
        .NCH(NCH), .LOG2_NCH(LOG2_NCH), .LOG2_NB(LOG2_NB),
        .W_U(W_U), .F_U(F_U), .W_A(W_A), .F_A(F_A),
        .W_B(W_B), .F_B(F_B), .W_S(W_S), .F_Y(F_Y),
        .W_D(W_D), .F_D(F_D), .W_BSH(W_BSH), .W_ACC(W_BACC),
        .W_OUT(W_SSM), .F_OUT(F_SSM)
    ) u_bank (
        .clk(clk), .rst_n(rst_n),
        .cfg_we(cfg_we), .cfg_unit(cfg_unit), .cfg_addr(cfg_addr), .cfg_data(cfg_data),
        .cfgd_we(cfgd_we), .cfgd_ch(cfgd_ch), .cfgd_data(cfgd_data), .cfgd_bsh(cfgd_bsh),
        .in_valid(ln_valid), .in_ready(bk_ready), .in_data(ln_data),
        .in_last_seq(ln_last_seq),
        .out_valid(bk_valid), .out_ready(ge_in_ready), .out_data(bk_data),
        .out_ch(bk_ch), .out_last(bk_last),
        .ovf(ovf)
    );

    // GELU output
    wire                    ge_valid;
    wire signed [W_SSM-1:0] ge_data;
    wire                    ge_last;      // bk_last arrived together with the data (PWL tag)

    // With s4d_pwl equipped with in_ready/out_ready the chain
    // bank -> gelu -> mixing is continuous: this case can no longer
    // occur. The flag stays as a sentinel: if it fires, it means
    // that someone broke the propagation of the ready.
    // NOTE (analysis): ge_valid && !mix_ready is also the state of a
    // normal AXI stall (GELU with data ready, mixing with both buffers
    // full). No data is lost, but the flag can rise when
    // the output of the core applies backpressure in the last pass.
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)                       err_bank_ovr <= 1'b0;
        else if (ge_valid && !mix_ready)  err_bank_ovr <= 1'b1;
    end

    //=================================================================
    // Piecewise-linear approximated GELU (256 segments in [-8, 8)),
    // outside the interval: 0 on the left, x on the right (MODE_HI=1)
    //=================================================================
    s4d_pwl #(
        .W_IN(W_SSM), .F_IN(F_SSM), .W_OUT(W_SSM), .F_OUT(F_SSM),
        .NSEG(256), .LOG2_NSEG(8), .XH(3),
        .MODE_HI(1), .Y_LO_Q(0), .Y_HI_Q(0),
        .TAG_W(1), .LUT_FILE(GELU_FILE)
    ) u_gelu (
        .clk(clk), .rst_n(rst_n),
        .in_valid(bk_valid), .in_ready(ge_in_ready), .in_data(bk_data),
        .in_tag(bk_last),                     // the end-of-frame flag travels with the data
        .out_ready(mix_ready),
        .out_valid(ge_valid), .out_data(ge_data), .out_tag(ge_last)
    );

    // mixing/GLU output
    wire                    gl_valid;
    wire signed [W_ACT-1:0] gl_data;
    wire                    gl_last;

    //=================================================================
    // Linear mixing H -> 2H followed by GLU: out = a * sigmoid(g)
    //=================================================================
    s4d_mixing_glu #(
        .H(H), .LOG2_H(LOG2_H),
        .NMAC(NMAC), .LOG2_NPH(LOG2_NPH),
        .W_IN(W_SSM), .F_IN(F_SSM),
        .W_W(W_W), .F_W(F_W), .W_ACC(W_MACC),
        .W_SIG(16), .F_SIGI(F_SIGI), .F_SIGO(F_SIGO),
        .W_A(16), .F_A(F_ACT),
        .W_OUT(W_ACT), .F_OUT(F_ACT),
        .LOADABLE(LOADABLE), .LD_W(LD_W),
        .LOG2_NBANK(6), .WMEM_STYLE(WMEM_STYLE),   // 48 banks -> 6 selection bits
        .TAIL_DEPTH(TAIL_DEPTH),                     // credits = slots of the output tail queue
        .W_FILE(WOUT_FILE), .B_FILE(BOUT_FILE), .SIG_FILE(SIG_FILE)
    ) u_mix (
        .clk(clk), .rst_n(rst_n),
        .ld_we_w(ld_wo_we), .ld_addr_w(ld_wo_addr), .ld_sel_w(ld_wo_sel),
        .ld_data_w(ld_wo_data),
        .ld_we_b(ld_bo_we), .ld_addr_b(ld_bo_addr), .ld_data_b(ld_bo_data),
        .in_valid(ge_valid), .in_ready(mix_ready),
        .in_data(ge_data), .in_last(ge_last),
        .tail_pop(tail_pop),
        .out_valid(gl_valid), .out_data(gl_data), .out_last(gl_last)
    );

    // the input passes only if the BatchNorm and the residual FIFO have room
    assign in_ready = ln_ready && fifo_ready;

    wire fifo_push = in_valid && in_ready;   // x enters the layer -> also into the FIFO
    wire fifo_pop  = gl_valid;               // the result arrives -> its x is consumed

    // pointers and count of the residual FIFO
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fwr  <= {FIFO_LOG2{1'b0}};
            frd  <= {FIFO_LOG2{1'b0}};
            fcnt <= {(FIFO_LOG2+1){1'b0}};
        end else begin
            if (fifo_push) fwr <= fwr + 1'b1;
            if (fifo_pop)  frd <= frd + 1'b1;
            fcnt <= fcnt + (fifo_push ? 1'b1 : 1'b0) - (fifo_pop ? 1'b1 : 1'b0);
        end
    end

    // write into the residual FIFO
    always @(posedge clk) if (rst_n && fifo_push) fifo[fwr] <= in_data;

    always @(posedge clk) if (fifo_push) fifo_ls[fwr] <= in_last_seq;

    //-----------------------------------------------------------------
    // Residual sum: x (from the FIFO) + GLU output, on W_ACT+1 bits
    // so as not to lose the carry, then saturation to W_ACT bits.
    // Timing (2026-10-15): before, the read of the FIFO in LUTRAM (1024
    // slots), the sum and the saturation were in the same cycle. Now:
    //   R1  read of the FIFO (x and flag) and copy of gl_data
    //   R2  sum x + gl
    //   R3  saturation -> output tail queue
    // Free running (2026-10-15): the pop of the FIFO happens in R1,
    // in the same cycle in which the GLU result is registered,
    // so the residual does not become misaligned.
    //-----------------------------------------------------------------
    localparam integer RW = W_ACT + 1;
    localparam signed [RW-1:0] RMAX = {{(RW-W_ACT+1){1'b0}}, {(W_ACT-1){1'b1}}};   // +32767
    localparam signed [RW-1:0] RMIN = {{(RW-W_ACT+1){1'b1}}, {(W_ACT-1){1'b0}}};   // -32768

    reg [LOG2_H-1:0] och;    // channel of the next result in R1 (for out_last_ch)

    // ---- R1: read of the residual FIFO ----
    reg                    r1_valid, r1_last_ch, r1_last_seq;
    reg signed [W_ACT-1:0] r1_x, r1_gl;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r1_valid    <= 1'b0;
            r1_last_ch  <= 1'b0;
            r1_last_seq <= 1'b0;
            och         <= {LOG2_H{1'b0}};
        end else begin
            r1_valid    <= gl_valid;
            r1_last_ch  <= gl_valid && (och == H-1);
            // end of sequence: last channel of a frame whose x was the last
            r1_last_seq <= gl_valid && (och == H-1) && fifo_ls[frd];
            if (gl_valid) och <= (och == H-1) ? {LOG2_H{1'b0}} : (och + 1'b1);
        end
    end

    always @(posedge clk) begin
        r1_x  <= fifo[frd];
        r1_gl <= gl_data;
    end

    // ---- R2: sum on RW bits ----
    reg                    r2_valid, r2_last_ch, r2_last_seq;
    reg signed [RW-1:0]    r2_sum;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r2_valid    <= 1'b0;
            r2_last_ch  <= 1'b0;
            r2_last_seq <= 1'b0;
        end else begin
            r2_valid    <= r1_valid;
            r2_last_ch  <= r1_last_ch;
            r2_last_seq <= r1_last_seq;
        end
    end

    always @(posedge clk)
        r2_sum <= {r1_x[W_ACT-1], r1_x} + {r1_gl[W_ACT-1], r1_gl};

    //-----------------------------------------------------------------
    // R3: saturation (free running) -> output tail queue
    //-----------------------------------------------------------------
    reg                    r3_valid, r3_last_ch, r3_last_seq;
    reg signed [W_ACT-1:0] r3_data;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r3_valid    <= 1'b0;
            r3_last_ch  <= 1'b0;
            r3_last_seq <= 1'b0;
        end else begin
            r3_valid    <= r2_valid;
            r3_last_ch  <= r2_last_ch;
            r3_last_seq <= r2_last_seq;
        end
    end

    always @(posedge clk) begin
        if      (r2_sum > RMAX) r3_data <= RMAX[W_ACT-1:0];
        else if (r2_sum < RMIN) r3_data <= RMIN[W_ACT-1:0];
        else                    r3_data <= r2_sum[W_ACT-1:0];
    end

    //-----------------------------------------------------------------
    // OUTPUT TAIL QUEUE: memory of TAIL_DEPTH slots (LUTRAM) + head
    // register, which is directly the output of the layer (out_valid/out_data/
    // out_last_*: all from registers).
    //   - the head is freed (tail_head_take) if empty or if it is read;
    //     then it picks from the memory, or, if the memory is empty,
    //     directly from the r3 input (no cycle lost at full rate);
    //   - otherwise the incoming data goes into the memory.
    // It cannot overflow: the samples in the queue (memory + head) are at
    // most those "in flight" counted by the mixing credits, <= TAIL_DEPTH-1.
    // out_ready drives only this logic (~40 loads).
    //-----------------------------------------------------------------
    localparam integer TAIL_W = W_ACT + 2;              // {last_seq, last_ch, data}

    reg [TAIL_W-1:0]    tail_mem [0:TAIL_DEPTH-1];     // queue memory (asynchronous read)
    reg [TAIL_LOG2-1:0] tail_wr, tail_rd;              // write / read pointers
    reg [TAIL_LOG2:0]   tail_cnt;                      // elements in memory (head excluded)

    wire [TAIL_W-1:0] tail_in        = {r3_last_seq, r3_last_ch, r3_data};
    wire              tail_mem_empty = (tail_cnt == {(TAIL_LOG2+1){1'b0}});

    assign tail_pop = out_valid && out_ready;           // the head is consumed
    wire tail_head_take = !out_valid || out_ready;      // the head can be reloaded

    // the incoming data goes into memory, except when it passes directly to the head
    wire tail_push = r3_valid && !(tail_head_take && tail_mem_empty);
    // the head picks from the memory
    wire tail_mem_pop = tail_head_take && !tail_mem_empty;

    always @(posedge clk) if (tail_push) tail_mem[tail_wr] <= tail_in;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tail_wr  <= {TAIL_LOG2{1'b0}};
            tail_rd  <= {TAIL_LOG2{1'b0}};
            tail_cnt <= {(TAIL_LOG2+1){1'b0}};
        end else begin
            if (tail_push)    tail_wr <= tail_wr + 1'b1;
            if (tail_mem_pop) tail_rd <= tail_rd + 1'b1;
            tail_cnt <= tail_cnt + (tail_push ? 1'b1 : 1'b0) - (tail_mem_pop ? 1'b1 : 1'b0);
        end
    end

    // head register = output of the layer
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid    <= 1'b0;
            out_data     <= {W_ACT{1'b0}};
            out_last_ch  <= 1'b0;
            out_last_seq <= 1'b0;
        end else if (tail_head_take) begin
            if (!tail_mem_empty) begin
                // the oldest in memory
                out_valid <= 1'b1;
                {out_last_seq, out_last_ch, out_data} <= tail_mem[tail_rd];
            end else begin
                // empty memory: directly from R3 (or empty head)
                out_valid <= r3_valid;
                {out_last_seq, out_last_ch, out_data} <= r3_valid ? tail_in : {TAIL_W{1'b0}};
            end
        end
    end

    // simulation-only check: the queue must never overflow
// synthesis translate_off
    always @(posedge clk)
        if (rst_n && tail_push && (tail_cnt == TAIL_DEPTH)) begin
            $display("s4d_layer: ERROR, output tail queue full (credits not respected) t=%0t", $time);
        end
// synthesis translate_on

endmodule

`default_nettype wire
