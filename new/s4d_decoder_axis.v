`timescale 1ns / 1ps
`default_nettype none
//=====================================================================
// s4d_decoder_axis -- parametric S4D decoder, PURE AXI-STREAM
//                     (file: s4d_decoder_axis.v)
//
//   S_AXIS : W_ACT-bit activations from the last layer of the core,
//            t-major order (H channels per timestep, L timesteps);
//            tlast = last channel of the last timestep
//   M_AXIS : the K logits of the sequence, 32 bits per beat, tlast on the last one
//
//   Computation (identical to the previous modules when L is a power of 2):
//
//     y[t][h]  = sat16( round( (w[h]*x[t][h] + b[h]) / 2^BN_SH ) )   final BatchNorm
//     m[h]     = sat16( floor( (sum_t y[t][h] + floor(L/2)) / L ) )  mean over L
//     xq[h]    = sat8 ( round( m[h] / 2^RQ_SHIFT ) )                 requantization
//     logit[c] = b_cls[c] + sum_h W[h][c] * xq[h]                    classifier
//
//   round = round-half-up (+ half LSB, then arithmetic shift).
//   The mean is round-half-up of sum/L even for any L (for odd L
//   there are no ties).
//
// What changes with respect to decoder.v + decoder_1.v + mean_pool.v + classifier.v:
//   - the LOG2_* are no longer parameters: they are derived with $clog2
//     (in the block design LOG2_K, LOG2_L, LOG2_H must be removed);
//   - any L: power of 2 -> shift as before (same latency);
//     otherwise EXACT division by a constant (multiplication by the
//     reciprocal rounded up, Granlund-Montgomery method);
//   - any K >= 1 and any H >= 1;
//   - W_ACC <= 32 (before it had to be = 32): the logit is
//     sign-extended onto the 32-bit beat;
//   - elaboration checks on the widths of the accumulators.
//   s4d_affine (s4d_affine.v), shared with the core, is unchanged.
//
// Hierarchy:
//   s4d_decoder_axis                 (this file)
//     |- s4d_affine      u_norm      final BatchNorm        [s4d_affine.v]
//     |- s4d_meanpool    u_pool      mean over L            (this file)
//     |- s4d_classifier  u_cls       int8, logits           (this file)
//
// The decoder applies no backpressure at the input (tready always high).
// The output packet must be drained before the logits of the next
// sequence arrive (at least L*H cycles later): in simulation a
// check reports the violation. Consecutive sequences with no pause at
// the input are allowed (the drain of the mean pool reads every accumulator
// before the new frame rewrites it).
//=====================================================================

module s4d_decoder_axis #(
    parameter integer H        = 128,     // channels
    parameter integer L        = 1024,    // timesteps per sequence (any >= 1)
    parameter integer K        = 5,       // classes: 5 ECG, 10 sCIFAR, 35 Speech Commands

    parameter integer W_ACT    = 16,      // input activations (Q5.11)
    parameter integer W_CIN    = 8,       // classifier input (int8)
    parameter integer W_CLS    = 8,       // classifier weights (int8)
    parameter integer W_ACC    = 32,      // accumulators / logit (<= 32, 32-bit output beat)
    parameter integer W_POOL   = 32,      // per-channel accumulator of the mean pool

    parameter integer F_ACT    = 11,      // fractional bits of the activations (documentation)
    parameter integer RQ_SHIFT = 6,       // Q5.11 -> int8 shift before the classifier

    // ---- final BatchNorm (fixed shift, weights in ROM) ----
    // Value per project (quant_report.json, hw_generics.BN_SH_FINAL):
    // PTB-XL/ECG 13, sCIFAR 12. Default = ECG (current ZCU104 target).
    parameter integer BN_SH    = 13,
    parameter integer BN_W_W   = 16,
    parameter integer BN_W_B   = 32,
    parameter integer BN_W_ACC = 34,

    // .mem files read at synthesis ($readmemh)
    parameter         BN_W_FILE = "l6_bn_w_ECG.mem",
    parameter         BN_B_FILE = "l6_bn_b_ECG.mem",
    parameter         W_FILE    = "cls_w_ECG.mem",
    parameter         B_FILE    = "cls_b_ECG.mem"
)(
    input  wire                    aclk,
    input  wire                    aresetn,           // asynchronous active-low reset

    // ---- input from the core: L*H beats, tlast on the last one ----
    input  wire                    s_axis_tvalid,
    output wire                    s_axis_tready,     // always 1
    input  wire signed [W_ACT-1:0] s_axis_tdata,
    input  wire                    s_axis_tlast,

    // ---- logits: K beats of 32 bits, tlast on the last one ----
    output wire                    m_axis_tvalid,
    input  wire                    m_axis_tready,
    output wire [31:0]             m_axis_tdata,
    output wire                    m_axis_tlast
);

    // widths of the counters (at least 1 bit even for H = 1 or K = 1)
    localparam integer H_W   = (H > 1) ? $clog2(H) : 1;   // channel index
    localparam integer IDX_W = (K > 1) ? $clog2(K) : 1;   // class index

    //=================================================================
    // Elaboration checks: an inconsistent parameter stops synthesis
    //=================================================================
    initial begin
        if (H < 1 || L < 1 || K < 1) begin
            $display("s4d_decoder_axis: H=%0d L=%0d K=%0d must be >= 1", H, L, K);
            $finish;
        end
        if (W_ACC > 32) begin
            $display("s4d_decoder_axis: W_ACC=%0d, the output beat is 32 bits", W_ACC);
            $finish;
        end
    end

    //=================================================================
    // Final per-channel BatchNorm: y = (w*x + b) >> BN_SH
    //   LOADABLE = 0: weights from BN_W_FILE/BN_B_FILE, load ports at 0.
    //   out_ready = 1: the pipeline never stalls.
    //=================================================================
    wire                    norm_valid;
    wire signed [W_ACT-1:0] norm_data;
    wire                    norm_last_ch;      // last channel of the timestep
    wire                    norm_last_seq;     // last sample of the sequence

    s4d_affine #(
        .H(H), .LOG2_H(H_W),
        .W_IN(W_ACT), .W_OUT(W_ACT),
        .W_W(BN_W_W), .W_B(BN_W_B), .W_ACC(BN_W_ACC), .SH(BN_SH),
        .W_FILE(BN_W_FILE), .B_FILE(BN_B_FILE)
    ) u_norm (
        .clk          (aclk),
        .rst_n        (aresetn),
        .ld_we_w      (1'b0),                 // runtime load not used
        .ld_we_b      (1'b0),
        .ld_addr      ({H_W{1'b0}}),
        .ld_w         ({BN_W_W{1'b0}}),
        .ld_b         ({BN_W_B{1'b0}}),
        .ld_we_sh     (1'b0),
        .ld_sh        (6'd0),
        .in_valid     (s_axis_tvalid),
        .in_ready     (s_axis_tready),
        .in_data      (s_axis_tdata),
        .in_last_seq  (s_axis_tlast),
        .out_ready    (1'b1),                 // no backpressure
        .out_valid    (norm_valid),
        .out_data     (norm_data),
        .out_last_ch  (norm_last_ch),
        .out_last_seq (norm_last_seq)
    );

    //=================================================================
    // Temporal mean per channel
    //=================================================================
    wire                    pool_valid;
    wire signed [W_ACT-1:0] pool_data;
    wire                    pool_last;

    s4d_meanpool #(
        .H(H), .L(L), .W_IN(W_ACT), .W_ACC(W_POOL), .W_OUT(W_ACT)
    ) u_pool (
        .clk         (aclk),
        .rst_n       (aresetn),
        .in_valid    (norm_valid),
        .in_data     (norm_data),
        .in_last_ch  (norm_last_ch),
        .in_last_seq (norm_last_seq),
        .out_valid   (pool_valid),
        .out_data    (pool_data),
        .out_last    (pool_last)
    );

    //=================================================================
    // Int8 linear classifier (logits only, no argmax)
    //=================================================================
    wire                    logit_valid, logit_last;
    wire [IDX_W-1:0]        logit_idx;
    wire signed [W_ACC-1:0] logit_data;

    s4d_classifier #(
        .H(H), .K(K),
        .W_IN(W_ACT), .W_CIN(W_CIN), .W_CLS(W_CLS), .W_ACC(W_ACC),
        .RQ_SHIFT(RQ_SHIFT), .W_FILE(W_FILE), .B_FILE(B_FILE)
    ) u_cls (
        .clk         (aclk),
        .rst_n       (aresetn),
        .in_valid    (pool_valid),
        .in_data     (pool_data),
        .in_last     (pool_last),
        .out_valid   (logit_valid),
        .out_idx     (logit_idx),
        .out_logit   (logit_data),
        .out_last    (logit_last)
    );

    //=================================================================
    // Capture of the logits: they arrive 1 per cycle without backpressure, they must be
    // parked before they can be sent on M_AXIS
    //=================================================================
    reg signed [W_ACC-1:0] logit_buf [0:K-1];
    always @(posedge aclk) if (logit_valid) logit_buf[logit_idx] <= logit_data;

    //=================================================================
    // Sending of the packet: K beats read from logit_buf
    //   it starts in the cycle in which the last logit arrives (logit_buf[0..K-2]
    //   are already written; logit_buf[K-1] is written on this edge
    //   and read no earlier than K-1 beats later). With K = 1 the first beat
    //   is taken directly from logit_data.
    //=================================================================
    reg              sending;     // 1 = packet in progress
    reg [IDX_W-1:0]  send_idx;    // logit currently on the output register
    reg [IDX_W-1:0]  next_idx;    // = send_idx + 1, kept in a register: the mux
                                  //   logit_buf[next_idx] does not wait for the adder

    reg              o_valid;     // M_AXIS output register
    reg [31:0]       o_data;
    reg              o_last;

    assign m_axis_tvalid = o_valid;
    assign m_axis_tdata  = o_data;
    assign m_axis_tlast  = o_last;

    wire out_beat  = o_valid && m_axis_tready;        // output beat accepted
    wire pkt_start = logit_valid && logit_last;       // last logit of the sequence arrived

    // logit sign-extended to 32 bits
    function [31:0] sext32;
        input signed [W_ACC-1:0] value;
        begin
            sext32 = {{(33-W_ACC){value[W_ACC-1]}}, value[W_ACC-2:0]};
        end
    endfunction

    // first beat: with K = 1 the logit is arriving now and is not yet in logit_buf
    wire signed [W_ACC-1:0] first_logit = (K == 1) ? logit_data : logit_buf[0];

    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn) begin
            sending  <= 1'b0;
            send_idx <= {IDX_W{1'b0}};
            next_idx <= {IDX_W{1'b0}};
            o_valid  <= 1'b0;
            o_data   <= 32'd0;
            o_last   <= 1'b0;
        end else if (!sending) begin
            if (pkt_start) begin
                // first logit of the packet
                sending  <= 1'b1;
                send_idx <= {IDX_W{1'b0}};
                next_idx <= {{(IDX_W-1){1'b0}}, 1'b1};
                o_valid  <= 1'b1;
                o_data   <= sext32(first_logit);
                o_last   <= (K == 1);
            end
        end else if (out_beat) begin
            if (send_idx == K-1) begin
                // last logit accepted: packet finished
                sending <= 1'b0;
                o_valid <= 1'b0;
                o_last  <= 1'b0;
            end else begin
                // next logit
                send_idx <= next_idx;
                next_idx <= next_idx + 1'b1;
                o_data   <= sext32(logit_buf[next_idx]);
                o_last   <= (next_idx == K-1);
            end
        end
    end

    //=================================================================
    // Simulation-only checks (no logic in synthesis)
    //=================================================================
// synthesis translate_off
    localparam integer NBEAT = L * H;             // beats per sequence
    integer sim_beat_count = 0;                   // beats received in the current sequence
    always @(posedge aclk) if (aresetn) begin
        // tlast contract: exactly L*H beats per sequence
        if (s_axis_tvalid && s_axis_tready) begin
            if (s_axis_tlast) begin
                if (sim_beat_count != NBEAT-1)
                    $display("s4d_decoder_axis: tlast at beat %0d, expected %0d", sim_beat_count, NBEAT-1);
                sim_beat_count = 0;
            end else begin
                sim_beat_count = sim_beat_count + 1;
            end
        end
        // new logits while the previous packet has not yet left
        if (logit_valid && sending)
            $display("s4d_decoder_axis: new logits during the sending of the previous packet (overwrite)");
    end
// synthesis translate_on

endmodule


//=====================================================================
// s4d_meanpool -- temporal mean per channel (global average pool)
//
//   It receives L timesteps x H channels in t-major order and, for each channel,
//   accumulates acc[ch] = sum_t x[t][ch]. At the end of the sequence it emits
//   in streaming (one channel per cycle, H consecutive cycles):
//
//     out[ch] = sat( floor( (acc[ch] + floor(L/2)) / L ) )
//
//   i.e. the rounded mean (round-half-up) in the same format as
//   the input.
//
//   Accumulation in 2 stages (timing: LUTRAM read and sum in different cycles):
//     A1  read of acc[ch] (with forwarding of the value being written if the
//         channel is the same as that of the previous sample, i.e. H = 1)
//     A2  sum and write of acc[ch]
//   Drain (after the last sample of the sequence has been written):
//     D1          registered read of acc[ocnt]                  -> acc_rd
//     L = 2^n:    D2  acc_rd + L/2
//                 D3  shift by n bits (wires) and saturation    -> output
//     any L: exact division by the constant L:
//       D2  num  = acc_rd + floor(L/2) + 2^(W_IN-1)*L   (>= 0, < 2^NUM_W)
//       D3  prod = num * MAGIC                           (M register of the DSP)
//       D4  registered prod                              (P register of the DSP)
//       D5  quot = prod >> SHIFT (wires);  mean = quot - 2^(W_IN-1)
//       D6  saturation                                   -> output
//     with MAGIC = ceil(2^SHIFT / L), SHIFT = NUM_W + ceil(log2 L):
//     floor(num/L) = (num*MAGIC) >> SHIFT is exact for every num < 2^NUM_W
//     (Granlund-Montgomery, "Division by invariant integers", 1994).
//     The offset 2^(W_IN-1)*L is a multiple of L: it makes num non-negative
//     without changing the remainder, and it is removed from the quotient as 2^(W_IN-1).
//
//   Accumulation and drain can overlap (next sequence without
//   pause): D1 reads acc[k] on the same edge as, or before, the one on which
//   the first frame of the new sequence rewrites acc[k].
//=====================================================================
module s4d_meanpool #(
    parameter integer H      = 128,       // channels
    parameter integer L      = 1024,      // timesteps per sequence
    parameter integer W_IN   = 16,        // input (Q5.11)
    parameter integer W_ACC  = 32,        // per-channel accumulator
    parameter integer W_OUT  = 16         // output (same format as the input)
)(
    input  wire                    clk,
    input  wire                    rst_n,             // asynchronous active-low reset

    // ---- input: one sample per beat, frame of H channels ----
    input  wire                    in_valid,
    input  wire signed [W_IN-1:0]  in_data,
    input  wire                    in_last_ch,        // last channel of the frame (timestep)
    input  wire                    in_last_seq,       // last sample of the sequence

    // ---- output: H means, one per cycle ----
    output reg                     out_valid,
    output reg  signed [W_OUT-1:0] out_data,
    output reg                     out_last           // last channel (ch = H-1)
);

    localparam integer H_W     = (H > 1) ? $clog2(H) : 1;   // channel index
    localparam integer LOG2_L  = $clog2(L);                 // ceil(log2 L), 0 for L = 1
    localparam         L_POW2  = ((L & (L - 1)) == 0);      // 1 if L is a power of 2
    localparam integer HALF_L  = L / 2;                     // floor(L/2): half LSB of the mean

    // division by a constant (used only if L is not a power of 2)
    localparam integer NUM_W   = W_IN + LOG2_L;             // bits of the non-negative numerator
    localparam integer SHIFT   = NUM_W + LOG2_L;            // shift of the product
    localparam integer MAGIC_W = NUM_W + 1;                 // MAGIC < 2^(NUM_W+1)
    localparam integer PROD_W  = NUM_W + MAGIC_W;

    // MAGIC = ceil(2^SHIFT / L), computed at elaboration on 128 bits
    function [MAGIC_W-1:0] div_magic;
        input integer divisor;
        input integer shift_bits;
        reg [127:0] power_of_two;
        begin
            power_of_two = 128'd1 << shift_bits;
            div_magic    = (power_of_two + divisor - 1) / divisor;
        end
    endfunction
    localparam [MAGIC_W-1:0] MAGIC = div_magic(L, SHIFT);

    // saturation limits at W_OUT bits (in practice never reached: the
    // mean of W_IN-bit values fits in W_IN bits)
    localparam signed [W_ACC-1:0] SAT_MAX = (1 <<< (W_OUT-1)) - 1;
    localparam signed [W_ACC-1:0] SAT_MIN = -(1 <<< (W_OUT-1));

    initial begin
        // |acc| <= 2^(W_IN-1) * L: W_IN + ceil(log2 L) signed bits are needed
        if (W_ACC < W_IN + LOG2_L) begin
            $display("s4d_meanpool: W_ACC=%0d < W_IN+ceil(log2 L)=%0d, overflow possible",
                     W_ACC, W_IN + LOG2_L);
            $finish;
        end
        if (!L_POW2 && SHIFT > 120) begin
            $display("s4d_meanpool: SHIFT=%0d too large for div_magic", SHIFT);
            $finish;
        end
    end

    //-----------------------------------------------------------------
    // Accumulation: first timestep -> acc = x, other timesteps -> acc += x
    // (the first flag acts as a "clear" at the start of every sequence)
    //-----------------------------------------------------------------
    reg signed [W_ACC-1:0] acc [0:H-1];       // LUTRAM: 1 read/write port + 1 read port
    reg [H_W-1:0]          ch;                // channel of the next input sample
    reg                    first;             // 1 = current frame = first of the sequence

    wire signed [W_ACC-1:0] in_ext  = {{(W_ACC-W_IN){in_data[W_IN-1]}}, in_data};
    // ---- A1: read of the accumulator of the channel ----
    reg                    a1_valid, a1_first, a1_last_seq;
    reg [H_W-1:0]          a1_ch;
    reg signed [W_ACC-1:0] a1_in, a1_acc;

    // ---- A2: value to write (new accumulator) ----
    wire signed [W_ACC-1:0] a2_value = a1_first ? a1_in : (a1_acc + a1_in);

    // forwarding: the channel being read now is the one A2 is writing (only H = 1)
    wire forward = a1_valid && (a1_ch == ch);

    always @(posedge clk) begin
        a1_ch    <= ch;
        a1_in    <= in_ext;
        a1_first <= first;
        a1_acc   <= forward ? a2_value : acc[ch];
        if (a1_valid) acc[a1_ch] <= a2_value;
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            a1_valid    <= 1'b0;
            a1_last_seq <= 1'b0;
        end else begin
            a1_valid    <= in_valid;
            a1_last_seq <= in_valid && in_last_seq;
        end
    end

    // channel counter realigned on in_last_ch (generated by s4d_affine)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ch    <= {H_W{1'b0}};
            first <= 1'b1;              // the first sequence after reset starts from zero
        end else if (in_valid) begin
            if (in_last_ch) begin
                ch    <= {H_W{1'b0}};
                first <= in_last_seq;   // 1 only after the last frame of the sequence
            end else begin
                ch <= ch + 1'b1;
            end
        end
    end

    //-----------------------------------------------------------------
    // Drain, stage D1: scan ocnt = 0..H-1 and registered read
    //-----------------------------------------------------------------
    reg                    drain;             // 1 = D1 active
    reg [H_W-1:0]          ocnt;              // channel read in D1
    reg signed [W_ACC-1:0] acc_rd;            // registered acc[ocnt]
    reg                    rd_valid, rd_last; // acc_rd valid / last channel

    always @(posedge clk) acc_rd <= acc[ocnt];   // data: no reset

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            drain    <= 1'b0;
            ocnt     <= {H_W{1'b0}};
            rd_valid <= 1'b0;
            rd_last  <= 1'b0;
        end else begin
            rd_valid <= drain;
            rd_last  <= drain && (ocnt == H-1);
            if (!drain) begin
                if (a1_last_seq) begin               // last sample written now: reading starts
                    drain <= 1'b1;
                    ocnt  <= {H_W{1'b0}};
                end
            end else begin
                if (ocnt == H-1) drain <= 1'b0;      // all channels read
                else             ocnt  <= ocnt + 1'b1;
            end
        end
    end

    generate
    if (L_POW2) begin : g_shift
        //-------------------------------------------------------------
        // L = 2^n: D2 acc_rd + L/2;  D3 >>> n (wires) and saturation
        //-------------------------------------------------------------
        reg signed [W_ACC-1:0] sum_d2;
        reg                    valid_d2, last_d2;

        always @(posedge clk) sum_d2 <= acc_rd + HALF_L;

        wire signed [W_ACC-1:0] avg = sum_d2 >>> LOG2_L;

        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                valid_d2  <= 1'b0;
                last_d2   <= 1'b0;
                out_valid <= 1'b0;
                out_data  <= {W_OUT{1'b0}};
                out_last  <= 1'b0;
            end else begin
                valid_d2  <= rd_valid;
                last_d2   <= rd_last;
                out_valid <= valid_d2;
                out_last  <= last_d2;
                if      (avg > SAT_MAX) out_data <= SAT_MAX[W_OUT-1:0];
                else if (avg < SAT_MIN) out_data <= SAT_MIN[W_OUT-1:0];
                else                    out_data <= avg[W_OUT-1:0];
            end
        end
    end else begin : g_divide
        //-------------------------------------------------------------
        // any L: exact division by a constant (D2..D5)
        //-------------------------------------------------------------
        // offset, a multiple of L, that makes the numerator non-negative
        localparam signed [W_ACC+1:0] NUM_OFFSET = (64'sd1 <<< (W_IN-1)) * L + HALF_L;
        localparam signed [W_ACC+1:0] QUOT_BIAS  = (64'sd1 <<< (W_IN-1));   // offset / L

        reg [NUM_W-1:0]  num_d2;              // D2: non-negative numerator
        reg [PROD_W-1:0] prod_d3, prod_d4;    // D3/D4: product (M and P registers of the DSP)
        reg [3:0]        valid_pipe, last_pipe;   // valid/last of D2, D3, D4, D5

        // acc_rd extended by 2 bits to add the offset without overflow
        wire signed [W_ACC+1:0] num_full = {{2{acc_rd[W_ACC-1]}}, acc_rd} + NUM_OFFSET;

        always @(posedge clk) begin
            num_d2  <= num_full[NUM_W-1:0];   // 0 <= num_full < 2^NUM_W by construction
            prod_d3 <= num_d2 * MAGIC;
            prod_d4 <= prod_d3;
        end

        // D5: quotient (wires), with the offset part removed
        wire signed [W_ACC+1:0] quot = $signed({2'b00, prod_d4[PROD_W-1:SHIFT]});
        reg  signed [W_ACC+1:0] avg;
        always @(posedge clk) avg <= quot - QUOT_BIAS;
        // D6: saturation (below)

        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                valid_pipe <= 4'b0000;
                last_pipe  <= 4'b0000;
                out_valid  <= 1'b0;
                out_data   <= {W_OUT{1'b0}};
                out_last   <= 1'b0;
            end else begin
                valid_pipe <= {valid_pipe[2:0], rd_valid};
                last_pipe  <= {last_pipe[2:0],  rd_last};
                out_valid  <= valid_pipe[3];
                out_last   <= last_pipe[3];
                if      (avg > SAT_MAX) out_data <= SAT_MAX[W_OUT-1:0];
                else if (avg < SAT_MIN) out_data <= SAT_MIN[W_OUT-1:0];
                else                    out_data <= avg[W_OUT-1:0];
            end
        end
    end
    endgenerate

endmodule


//=====================================================================
// s4d_classifier -- int8 linear classifier (H -> K), logits only
//
//     xq[h]    = sat_int8( round( x[h] / 2^RQ_SHIFT ) )
//     logit[c] = b[c] + sum_{h<H} W[h][c] * xq[h]           c = 0..K-1
//
//   Formats: x W_IN bits; xq W_CIN bits; W W_CLS signed bits;
//   b, acc, logit W_ACC bits in the scale of the product W*xq.
//
//   ROM from .mem files:
//     W_FILE : H rows of K*W_CLS bits, class c in bits [c*W_CLS +: W_CLS]
//     B_FILE : K rows of W_ACC bits
//
//   Pipeline (classifier.v with one more stage, timing):
//     CA (in_valid)  x + half LSB of the requantization
//     C0 (ca_valid)  shift (wires) and saturation -> xq, read of row W[h]
//     C1 (c0_valid)  K registered products
//     C2 (c1_valid)  K accumulations (first channel: acc = b + prod)
//     E  (mac_done)  emission of the K logits, 1 per cycle
//=====================================================================
module s4d_classifier #(
    parameter integer H        = 128,     // inputs (channels)
    parameter integer K        = 5,       // classes
    parameter integer W_IN     = 16,      // input x (Q5.11)
    parameter integer W_CIN    = 8,       // requantized input xq
    parameter integer W_CLS    = 8,       // weights W
    parameter integer W_ACC    = 32,      // accumulators, biases and logits
    parameter integer RQ_SHIFT = 6,       // requantization shift x -> xq
    parameter         W_FILE   = "cls_w_ECG.mem",
    parameter         B_FILE   = "cls_b_ECG.mem"
)(
    input  wire                    clk,
    input  wire                    rst_n,             // asynchronous active-low reset

    // ---- input: H samples, in_last on the last one ----
    input  wire                    in_valid,
    input  wire signed [W_IN-1:0]  in_data,
    input  wire                    in_last,

    // ---- logits: K consecutive beats (index + value) ----
    output reg                     out_valid,
    output reg  [((K > 1) ? $clog2(K) : 1)-1:0] out_idx,   // class of the logit
    output reg  signed [W_ACC-1:0] out_logit,
    output reg                     out_last           // last logit (class K-1)
);

    localparam integer H_W   = (H > 1) ? $clog2(H) : 1;
    localparam integer IDX_W = (K > 1) ? $clog2(K) : 1;
    localparam integer W_PROD = W_CLS + W_CIN;            // bits of a product

    initial begin
        // sum of H products without bias: W_PROD + ceil(log2 H) bits are needed
        if (W_ACC < W_PROD + $clog2(H)) begin
            $display("s4d_classifier: W_ACC=%0d < W_CLS+W_CIN+ceil(log2 H)=%0d",
                     W_ACC, W_PROD + $clog2(H));
            $finish;
        end
        if (RQ_SHIFT < 0 || RQ_SHIFT >= W_IN) begin
            $display("s4d_classifier: RQ_SHIFT=%0d outside [0, W_IN)", RQ_SHIFT);
            $finish;
        end
    end

    // ROM of the weights (one row = the K weights of a channel) and of the biases
    reg [K*W_CLS-1:0]      w_rom [0:H-1];
    reg signed [W_ACC-1:0] b_rom [0:K-1];

    initial begin
        $readmemh(W_FILE, w_rom);
        $readmemh(B_FILE, b_rom);
    end

    //-----------------------------------------------------------------
    // Requantization x -> xq: (x + 2^(RQ_SHIFT-1)) >>> RQ_SHIFT, then
    // saturation to W_CIN bits; computed on W_IN+1 bits for the carry
    //-----------------------------------------------------------------
    localparam signed [W_IN:0] CSAT_MAX = (1 <<< (W_CIN-1)) - 1;
    localparam signed [W_IN:0] CSAT_MIN = -(1 <<< (W_CIN-1));

    wire signed [W_IN:0] rq_rnd = (RQ_SHIFT == 0)
                                ? {(W_IN+1){1'b0}}
                                : ({{W_IN{1'b0}}, 1'b1} <<< (RQ_SHIFT-1));   // half LSB
    wire signed [W_IN:0] rq_ext = {in_data[W_IN-1], in_data};

    //-----------------------------------------------------------------
    // CA: addition of the half LSB (W_IN+1 bits for the carry)
    //-----------------------------------------------------------------
    reg                    ca_valid, ca_first, ca_last;
    reg [H_W-1:0]          hcnt;              // channel of the next input
    reg [H_W-1:0]          ca_h;              // channel of the data in CA
    reg signed [W_IN:0]    ca_sum;            // x + 2^(RQ_SHIFT-1)

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ca_valid <= 1'b0;
            ca_first <= 1'b0;
            ca_last  <= 1'b0;
            hcnt     <= {H_W{1'b0}};
        end else begin
            ca_valid <= in_valid;
            ca_first <= in_valid && (hcnt == 0);
            ca_last  <= in_valid && in_last;
            if (in_valid) hcnt <= in_last ? {H_W{1'b0}} : (hcnt + 1'b1);
        end
    end

    always @(posedge clk) begin
        ca_h   <= hcnt;
        ca_sum <= rq_ext + rq_rnd;
    end

    //-----------------------------------------------------------------
    // C0: shift (wires) and saturation -> xq; read of the weight row
    //-----------------------------------------------------------------
    wire signed [W_IN:0] rq_val = ca_sum >>> RQ_SHIFT;

    reg signed [W_CIN-1:0] xq;
    reg                    c0_valid, c0_first, c0_last;
    reg [K*W_CLS-1:0]      w_row;             // weights W[h][0..K-1] of the current channel

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            c0_valid <= 1'b0;
            c0_first <= 1'b0;
            c0_last  <= 1'b0;
        end else begin
            c0_valid <= ca_valid;
            c0_first <= ca_first;
            c0_last  <= ca_last;
        end
    end

    always @(posedge clk) begin
        w_row <= w_rom[ca_h];
        if      (rq_val > CSAT_MAX) xq <= CSAT_MAX[W_CIN-1:0];
        else if (rq_val < CSAT_MIN) xq <= CSAT_MIN[W_CIN-1:0];
        else                        xq <= rq_val[W_CIN-1:0];
    end

    //-----------------------------------------------------------------
    // C1 (product) and C2 (accumulation), K classes in parallel
    //-----------------------------------------------------------------
    reg c1_valid, c1_first, c1_last;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            c1_valid <= 1'b0;
            c1_first <= 1'b0;
            c1_last  <= 1'b0;
        end else begin
            c1_valid <= c0_valid;
            c1_first <= c0_first;
            c1_last  <= c0_last;
        end
    end

    reg signed [W_ACC-1:0] acc [0:K-1];       // logits being accumulated

    genvar class_c;
    generate
        for (class_c = 0; class_c < K; class_c = class_c + 1) begin : g_mac
            wire signed [W_CLS-1:0]  w_c = $signed(w_row[class_c*W_CLS +: W_CLS]);
            reg  signed [W_PROD-1:0] prod_q;                              // registered product
            wire signed [W_ACC-1:0]  prod_ext = {{(W_ACC-W_PROD){prod_q[W_PROD-1]}}, prod_q};

            always @(posedge clk)                                         // C1
                if (c0_valid) prod_q <= w_c * xq;

            always @(posedge clk)                                         // C2
                if (c1_valid)
                    acc[class_c] <= (c1_first ? b_rom[class_c] : acc[class_c]) + prod_ext;
        end
    endgenerate

    // the cycle after the last accumulation all the acc are complete
    reg mac_done;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) mac_done <= 1'b0;
        else        mac_done <= c1_valid && c1_last;
    end

    //-----------------------------------------------------------------
    // E: emission of the K logits, one per cycle
    //-----------------------------------------------------------------
    reg             emit;                     // 1 = emission in progress
    reg [IDX_W-1:0] ecnt;                     // class being emitted

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            emit      <= 1'b0;
            ecnt      <= {IDX_W{1'b0}};
            out_valid <= 1'b0;
            out_idx   <= {IDX_W{1'b0}};
            out_logit <= {W_ACC{1'b0}};
            out_last  <= 1'b0;
        end else begin
            out_valid <= 1'b0;                // default: 1-cycle pulses
            out_last  <= 1'b0;
            if (!emit) begin
                if (mac_done) begin
                    emit <= 1'b1;
                    ecnt <= {IDX_W{1'b0}};
                end
            end else begin
                out_valid <= 1'b1;
                out_idx   <= ecnt;
                out_logit <= acc[ecnt];       // registered K:1 mux
                out_last  <= (ecnt == K-1);
                if (ecnt == K-1) emit <= 1'b0;
                else             ecnt <= ecnt + 1'b1;
            end
        end
    end

endmodule

`default_nettype wire
