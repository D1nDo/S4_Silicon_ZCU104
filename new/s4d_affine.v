`timescale 1ns / 1ps
`default_nettype none
//=====================================================================
// s4d_affine  --  per-channel affine transformation (fused BatchNorm)
//
//   For each sample of channel ch:
//
//     y = sat( round( (w[ch]*x + b[ch]) / 2^sh ) )
//
//   The samples arrive in channel order 0, 1, ..., H-1, 0, 1, ...
//   and the internal counter ch keeps track of the current channel.
//
//   Weights and biases: from .mem files (LOADABLE = 0) or written at
//   runtime through the ld_* ports (LOADABLE = 1). The shift sh is SH after
//   reset and, with LOADABLE = 1, it can be rewritten by ld_sh.
//
//   6-stage pipeline, globally stallable with out_ready, at most ONE
//   complex operation per cycle (timing, 2026-10-15; before: 3 stages with
//   w*x+b in one cycle and addition + variable shift + saturation in
//   another):
//     S0 read of w[ch] and b[ch], register of x
//     S1 product w*x                                (M register of the DSP)
//     S2 bias addition: w*x + b                     (P register of the DSP)
//     S3 addition of the half LSB: acc + 2^(sh-1)
//     S4 variable arithmetic shift by sh bits
//     S5 saturation to W_OUT bits -> output
//   Arithmetic identical to the 3-stage version (same widths, same
//   truncations): only the position of the registers changes.
//   The half LSB 2^(sh-1) is a register computed from sh_r (which changes
//   only when the weights are loaded, many cycles before the data).
//=====================================================================

module s4d_affine #(
    parameter integer H      = 128,    // channels
    parameter integer LOG2_H = 7,
    parameter integer W_IN   = 16,     // input x
    parameter integer W_OUT  = 16,     // output y
    parameter integer W_W    = 16,     // weight w
    parameter integer W_B    = 32,     // bias b (in the scale of the product w*x)
    parameter integer W_ACC  = 34,     // accumulator w*x + b
    parameter integer SH     = 11,     // output shift after reset

    parameter integer LOADABLE = 0,    // 1 = weights/biases/shift written by the ld_* ports
    parameter         W_FILE = "l6_bn_w.mem",  // module reused for 6+1 different roles
    parameter         B_FILE = "l6_bn_b.mem"   // (no name is "the" right one): it points
                                                // to the final BN of the decoder, the only case
                                                // in which this default is actually read
                                                // (LOADABLE=0). Whoever uses it for a layer must
                                                // always override it explicitly.
)(
    input  wire                    clk,
    input  wire                    rst_n,

    // ---- load of weight and bias of channel ld_addr ----
    input  wire                    ld_we_w,
    input  wire                    ld_we_b,
    input  wire [LOG2_H-1:0]       ld_addr,
    input  wire signed [W_W-1:0]   ld_w,
    input  wire signed [W_B-1:0]   ld_b,

    // ---- load of the shift ----
    input  wire                    ld_we_sh,
    input  wire [5:0]              ld_sh,

    // ---- input ----
    input  wire                    in_valid,
    output wire                    in_ready,
    input  wire signed [W_IN-1:0]  in_data,
    input  wire                    in_last_seq,   // last sample of the sequence

    // ---- output ----
    input  wire                    out_ready,
    output reg                     out_valid,
    output reg  signed [W_OUT-1:0] out_data,
    output reg                     out_last_ch,   // last channel of the frame
    output reg                     out_last_seq   // last sample of the sequence
);

    // weight and bias memories, one entry per channel
    reg signed [W_W-1:0] w_rom [0:H-1];
    reg signed [W_B-1:0] b_rom [0:H-1];

    generate
    if (LOADABLE == 0) begin : g_init
        // fixed values from file
        initial begin
            $readmemh(W_FILE, w_rom);
            $readmemh(B_FILE, b_rom);
        end
    end else begin : g_load
        // values written at runtime
        always @(posedge clk) begin
            if (ld_we_w) w_rom[ld_addr] <= ld_w;
            if (ld_we_b) b_rom[ld_addr] <= ld_b;
        end
    end
    endgenerate

    // output shift: SH after reset, then possibly loaded
    reg [5:0] sh_r;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)                             sh_r <= SH[5:0];
        else if ((LOADABLE != 0) && ld_we_sh)   sh_r <= ld_sh;
    end

    // global stall: the whole pipeline advances only with out_ready
    // (in_ready is combinational from out_ready)
    wire en = out_ready;
    assign in_ready = out_ready;

    wire acc_in = in_valid & in_ready;     // sample accepted

    reg [LOG2_H-1:0] ch;                   // channel of the next input sample

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)          ch <= {LOG2_H{1'b0}};
        else if (acc_in)     ch <= (ch == H-1) ? {LOG2_H{1'b0}} : ch + 1'b1;
    end

    //=================================================================
    // S0: read of weight and bias of the channel, input register
    //=================================================================
    reg signed [W_IN-1:0] x0;
    reg signed [W_W-1:0]  w0;
    reg signed [W_B-1:0]  b0;
    reg [LOG2_H-1:0]      ch0;
    reg                   v0, ls0;         // valid, last of the sequence

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v0 <= 1'b0;
            ls0 <= 1'b0;
        end else if (en) begin
            v0  <= acc_in;
            ls0 <= acc_in & in_last_seq;
        end
    end

    always @(posedge clk) begin
        if (en) begin
            x0  <= in_data;
            w0  <= w_rom[ch];
            b0  <= b_rom[ch];
            ch0 <= ch;
        end
    end

    //=================================================================
    // Flags that travel with the data in stages S1..S4:
    //   v_pipe[k]    = valid data at stage S(k+1)
    //   last_pipe[k] = it is the last channel of the frame
    //   seq_pipe[k]  = it is the last sample of the sequence
    //=================================================================
    reg [3:0] v_pipe, last_pipe, seq_pipe;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v_pipe    <= 4'b0;
            last_pipe <= 4'b0;
            seq_pipe  <= 4'b0;
        end else if (en) begin
            v_pipe    <= {v_pipe[2:0],    v0};
            last_pipe <= {last_pipe[2:0], v0 && (ch0 == H-1)};
            seq_pipe  <= {seq_pipe[2:0],  v0 && (ch0 == H-1) && ls0};
        end
    end

    //=================================================================
    // S1: product w*x
    //=================================================================
    reg signed [W_W+W_IN-1:0] prod1;       // w*x
    reg signed [W_B-1:0]      b1;          // bias aligned with the product

    always @(posedge clk) begin
        if (en) begin
            prod1 <= w0 * x0;
            b1    <= b0;
        end
    end

    //=================================================================
    // S2: bias addition (extended to W_ACC bits)
    //=================================================================
    reg signed [W_ACC-1:0] acc2;           // w*x + b

    always @(posedge clk) begin
        if (en)
            acc2 <= $signed({{(W_ACC-W_W-W_IN){prod1[W_W+W_IN-1]}}, prod1})
                  + $signed({{(W_ACC-W_B){b1[W_B-1]}}, b1});
    end

    //=================================================================
    // S3: addition of the output half LSB (round-half-up)
    //   rnd_r = 2^(sh_r-1), or 0 if sh_r = 0: register recomputed every
    //   cycle from sh_r, which is constant during the processing
    //=================================================================
    reg signed [W_ACC-1:0] rnd_r;
    always @(posedge clk)
        rnd_r <= (sh_r > 0) ? ({{(W_ACC-1){1'b0}}, 1'b1} <<< (sh_r-1))
                            : {W_ACC{1'b0}};

    reg signed [W_ACC-1:0] acc3;           // w*x + b + 2^(sh-1)

    always @(posedge clk) begin
        if (en) acc3 <= acc2 + rnd_r;
    end

    //=================================================================
    // S4: variable arithmetic shift by sh_r bits
    //=================================================================
    reg signed [W_ACC-1:0] shifted4;

    always @(posedge clk) begin
        if (en) shifted4 <= acc3 >>> sh_r;
    end

    //=================================================================
    // S5: saturation to W_OUT bits -> output register
    //=================================================================
    localparam signed [W_ACC-1:0] SAT_MAX = (1 <<< (W_OUT-1)) - 1;
    localparam signed [W_ACC-1:0] SAT_MIN = -(1 <<< (W_OUT-1));

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid    <= 1'b0;
            out_data     <= {W_OUT{1'b0}};
            out_last_ch  <= 1'b0;
            out_last_seq <= 1'b0;
        end else if (en) begin
            out_valid    <= v_pipe[3];
            out_last_ch  <= last_pipe[3];
            out_last_seq <= seq_pipe[3];

            if      (shifted4 > SAT_MAX) out_data <= SAT_MAX[W_OUT-1:0];
            else if (shifted4 < SAT_MIN) out_data <= SAT_MIN[W_OUT-1:0];
            else                         out_data <= shifted4[W_OUT-1:0];
        end
    end

endmodule

`default_nettype wire
