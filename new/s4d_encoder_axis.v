`timescale 1ns / 1ps
`default_nettype none
//=====================================================================
// s4d_encoder_axis -- PARAMETRIC linear encoder
//   NIN inputs of N_BIT_IN bits per timestep  ->  H channels of the S4D model
//   AXI-Stream input of AXI_DW bits, AXI-Stream output of W_ACT bits.
//
//   It replaces, with the same bit-exact result:
//     - s4d_encoder_ecg_axis64  (ECG:              NIN=12, 16 bits signed,    AXI_DW=64)
//     - s4d_encoder_sc_axis64   (Speech Commands:  NIN=12, 16 bits signed,    AXI_DW=64)
//     - axis_32to8 + s4d_encoder_axis (sCIFAR:     NIN=1,   8 bits unsigned,  AXI_DW=32)
//
//   Target: Zynq UltraScale+ XCZU7EV-2 (ZCU104), clock >= 250 MHz (4 ns).
//
//   Function computed, for every timestep t and every channel ch:
//
//     y[t][ch] = sat_W_ACT( round( sum_{c<NIN} W[ch][c]*x[t][c] + b[ch] ) >> OUT_SHIFT )
//
//   ---------------------------------------------------------------
//   INPUT (S_AXIS, AXI_DW bits) -- continuous stream of samples
//   ---------------------------------------------------------------
//     Each word contains SAMPLES_PER_WORD = AXI_DW / N_BIT_IN samples,
//     the first one in the least significant bits:
//       s_axis_tdata[N_BIT_IN-1:0]            = sample k
//       s_axis_tdata[2*N_BIT_IN-1:N_BIT_IN]   = sample k+1   ...
//
//     The samples form a continuous tape: a timestep is made of NIN
//     consecutive samples of the tape (c0 first) and CAN start in the
//     middle of a word. This is exactly the byte order in DDR (little-endian):
//       - ECG/SC : int16 c0..c11 per timestep -> 3 words of 64 bits per timestep
//       - sCIFAR : 1 uint8 pixel per timestep -> 4 timesteps per 32-bit word
//
//     tlast goes on the LAST word of the frame. Inside that word the last
//     timestep is the one that completes when fewer than NIN samples
//     remain: the remaining samples are padding and are
//     discarded. If the word with tlast ends without completing a timestep,
//     the partial timestep is discarded and err_frame is raised.
//
//   ---------------------------------------------------------------
//   OUTPUT (M_AXIS, W_ACT bits)
//   ---------------------------------------------------------------
//     H beats per timestep (channel 0, 1, ..., H-1).
//     tlast on the last channel of the last timestep of the frame.
//
//   ---------------------------------------------------------------
//   STRUCTURE
//   ---------------------------------------------------------------
//     1. Tape loader: takes 1 sample per cycle from the current word
//        (shift register, no multiplexer) and pushes it into the
//        input buffer (also a shift register).
//        NIN cycles per timestep are needed, overlapped with the H cycles of
//        computation of the previous timestep: no slowdown as long as NIN <= H.
//     2. Double buffer: input (next timestep) / computation
//        (timestep in progress, stable for all the H channels).
//     3. Stall-free pipeline (PIPE_STAGES stages):
//          S1, S2      Block RAM read of weights/bias (+ output register)
//          S3, S4, S5  NIN products in the DSP48E2 (registers A/B, M, P)
//          S6 ..       binary sum tree over NIN products + bias:
//                      TREE_LEVELS = clog2(NIN+1) levels, one 2-input
//                      adder per register
//          last        rounding + shift towards F_ACT
//        --> saturation to W_ACT bits written into the output FIFO.
//     4. Credit-based flow control: a channel enters S1 only if it already
//        has a guaranteed slot in the output FIFO, so the pipeline
//        never stalls and m_axis_tready drives only the FIFO.
//
//     With NIN=12: TREE_LEVELS=4, PIPE_STAGES=10 (like the ECG encoder).
//     With NIN=1 : TREE_LEVELS=1, PIPE_STAGES=7.
//     The arithmetic is integer and overflow-free (checked below), so
//     the order of the sums does not change the result.
//
//   ---------------------------------------------------------------
//   FORMATS
//   ---------------------------------------------------------------
//     IN_SIGNED = 1 : samples in two's complement (ECG, SC: Q5.11)
//     IN_SIGNED = 0 : unsigned samples (sCIFAR: pixels 0..255)
//     F_WENC    = fractional bits of the accumulator (weights + input)
//     OUT_SHIFT = F_WENC - F_ACT
//     enc_w.mem : H rows of NIN*W_WENC bits, one per channel. With
//                 C0_AT_LSB=1 the weight of c0 is in the low W_WENC bits.
//     enc_b.mem : H rows of W_BENC bits, already in the F_WENC scale.
//     (same .mem as the previous encoders, no regeneration)
//=====================================================================

module s4d_encoder_axis #(
    parameter integer H          = 128,          // channels of the model (outputs per timestep)
    parameter integer NIN        = 12,           // inputs (leads/samples) per timestep
    parameter integer N_BIT_IN   = 16,           // bits of an input sample
    parameter integer IN_SIGNED  = 1,            // 1 = signed samples, 0 = unsigned
    parameter integer AXI_DW     = 64,           // width of the S_AXIS word

    parameter integer W_WENC     = 18,           // bits of a weight
    parameter integer F_WENC     = 31,           // fractional bits of the accumulator
    parameter integer W_BENC     = 32,           // bits of a bias
    parameter integer W_ENCACC   = 40,           // bits of the accumulator / sum tree

    parameter integer W_ACT      = 16,           // bits of the output activation
    parameter integer F_ACT      = 11,           // fractional bits of the output (Q5.11)

    parameter integer C0_AT_LSB  = 1,            // 1 = weight of c0 in the LSBs of the enc_w.mem row

    parameter         ENC_W_FILE = "enc_w_ECG.mem",  // weight file
    parameter         ENC_B_FILE = "enc_b_ECG.mem"   // bias file
)(
    input  wire                        aclk,
    input  wire                        aresetn,          // active-low reset

    // ---- sample input (from the DMA) ----
    input  wire                        s_axis_tvalid,
    output wire                        s_axis_tready,
    input  wire [AXI_DW-1:0]           s_axis_tdata,     // SAMPLES_PER_WORD samples per word
    input  wire                        s_axis_tlast,     // last word of the frame

    // ---- activation output towards the S4D core ----
    output wire                        m_axis_tvalid,
    input  wire                        m_axis_tready,
    output wire signed [W_ACT-1:0]     m_axis_tdata,
    output wire                        m_axis_tlast,

    // ---- diagnostic flags (stay at 1 until reset) ----
    output reg                         sat_sticky,       // at least one output was saturated
    output reg                         err_frame         // frame ended in the middle of a timestep
);

    //=================================================================
    // Constant function: nodes present at level 'level' of the tree
    //   level 0 = NIN products + 1 bias; each level halves,
    //   rounding up (the odd node passes forward).
    //=================================================================
    function integer tree_nodes;
        input integer level;
        integer node_count, step;
        begin
            node_count = NIN + 1;
            for (step = 0; step < level; step = step + 1)
                node_count = (node_count + 1) / 2;
            tree_nodes = node_count;
        end
    endfunction

    //=================================================================
    // Derived constants
    //=================================================================
    localparam integer LOG2H            = (H > 1) ? $clog2(H) : 1;          // bits of the channel index
    localparam integer SAMPLES_PER_WORD = AXI_DW / N_BIT_IN;                // samples in an AXI word
    localparam integer W_WORD_SMP_IDX   = (SAMPLES_PER_WORD > 1) ? $clog2(SAMPLES_PER_WORD) : 1;
    localparam integer W_STEP_SMP_IDX   = (NIN > 1) ? $clog2(NIN) : 1;     // bits of the samples/timestep counter
    localparam integer W_STEP           = NIN * N_BIT_IN;                   // bits of a complete timestep

    // Sample extended for the DSP: if signed it stays N_BIT_IN bits, if unsigned
    // it gets a leading 0 (so the DSP sees it as positive)
    localparam integer W_X              = IN_SIGNED ? N_BIT_IN : (N_BIT_IN + 1);
    localparam integer W_PROD           = W_WENC + W_X;                     // bits of a weight*sample product
    localparam integer OUT_SHIFT        = F_WENC - F_ACT;                   // accumulator -> F_ACT shift

    localparam integer TREE_ADDENDS     = NIN + 1;                          // products + bias
    localparam integer TREE_LEVELS      = $clog2(TREE_ADDENDS);             // registered levels of the tree
    localparam integer W_LEVEL_BUS      = TREE_ADDENDS * W_ENCACC;          // bits of a level (maximum space)

    localparam integer DSP_LAST_STAGE   = 5;                                // S5 = P register of the DSPs
    localparam integer PIPE_STAGES      = DSP_LAST_STAGE + TREE_LEVELS + 1; // + tree + rounding
    localparam integer W_FIFO_PTR       = $clog2(PIPE_STAGES + 3);          // FIFO > latency of a credit
    localparam integer OUT_FIFO_DEPTH   = (1 << W_FIFO_PTR);                // power of 2
    localparam integer W_CREDIT         = W_FIFO_PTR + 1;                   // counts up to OUT_FIFO_DEPTH inclusive

    // Bits needed by the tree: the widest of the addends + one bit per level
    localparam integer W_WIDEST_ADDEND  = (W_PROD > W_BENC) ? W_PROD : W_BENC;
    localparam integer W_TREE_NEEDED    = W_WIDEST_ADDEND + TREE_LEVELS;

    //=================================================================
    // Parameter checks: they stop simulation/elaboration
    //=================================================================
    initial begin
        if (OUT_SHIFT < 0) begin
            $display("s4d_encoder_axis: F_WENC (%0d) < F_ACT (%0d)", F_WENC, F_ACT);
            $finish;
        end
        if ((AXI_DW % N_BIT_IN) != 0) begin
            // each word must contain an integer number of samples
            $display("s4d_encoder_axis: AXI_DW (%0d) not a multiple of N_BIT_IN (%0d)", AXI_DW, N_BIT_IN);
            $finish;
        end
        if (W_X > 27) begin
            // port A of the DSP48E2: 27 bits
            $display("s4d_encoder_axis: sample extended to %0d bits, beyond the 27 of the DSP port A", W_X);
            $finish;
        end
        if (W_ENCACC < W_TREE_NEEDED) begin
            // the tree could overflow: the result would no longer be exact
            $display("s4d_encoder_axis: W_ENCACC (%0d) < %0d bits needed for %0d addends",
                     W_ENCACC, W_TREE_NEEDED, TREE_ADDENDS);
            $finish;
        end
        if (NIN > H)
            // not an error: the loader (NIN cycles/timestep) becomes
            // slower than the computation (H cycles/timestep)
            $display("s4d_encoder_axis: WARNING NIN (%0d) > H (%0d), throughput < 1 channel/cycle", NIN, H);
    end

    //=================================================================
    // Block RAM of weights and biases (one row per output channel)
    //   Used read-only (ROM), initialized from the .mem files.
    //=================================================================
    (* rom_style = "block" *)
    reg        [NIN*W_WENC-1:0] enc_weight_rom [0:H-1];   // row ch = NIN weights of channel ch
    (* rom_style = "block" *)
    reg signed [W_BENC-1:0]     enc_bias_rom   [0:H-1];   // bias of channel ch (F_WENC scale)

    initial begin
        $readmemh(ENC_W_FILE, enc_weight_rom);
        $readmemh(ENC_B_FILE, enc_bias_rom);
    end

    //=================================================================
    // 1. TAPE LOADER
    //=================================================================

    //-----------------------------------------------------------------
    // Current word: right-shift register. The sample to consume
    // is always in the low N_BIT_IN bits.
    //-----------------------------------------------------------------
    reg [AXI_DW-1:0]         word_shift_reg;              // word being consumed
    reg                      word_valid;                  // 1 = samples remain to be consumed
    reg                      word_is_frame_last;          // the word had tlast
    reg [W_WORD_SMP_IDX-1:0] word_sample_idx;             // index of the current sample in the word

    wire [N_BIT_IN-1:0]      current_sample = word_shift_reg[N_BIT_IN-1:0];

    //-----------------------------------------------------------------
    // Input buffer: shift register, the new sample enters at the top.
    // After NIN pushes c0 is in the low bits (slot 0).
    //-----------------------------------------------------------------
    reg [W_STEP-1:0]         in_step_buf;                 // NIN samples of the incoming timestep
    reg [W_STEP_SMP_IDX-1:0] in_sample_idx;               // samples already loaded in the timestep
    reg                      in_buf_full;                 // 1 = timestep complete, waiting for computation
    reg                      in_buf_is_last_step;         // 1 = it is the last timestep of the frame

    // A sample is consumed if there is one and the input buffer has room
    wire load_sample        = word_valid & ~in_buf_full;
    // With this sample the timestep is complete
    wire step_complete      = load_sample & (in_sample_idx == NIN-1);
    // After this sample fewer than NIN remain in the word:
    // they are padding if the word is the last of the frame
    wire word_tail_is_pad   = (word_sample_idx + NIN >= SAMPLES_PER_WORD);
    // End of the frame: last timestep completed inside the word with tlast
    wire frame_end_here     = step_complete & word_is_frame_last & word_tail_is_pad;
    // Last sample of the word consumed
    wire word_exhausted     = load_sample & (word_sample_idx == SAMPLES_PER_WORD-1);
    // The word is finished: exhausted, or the rest is padding
    wire word_done          = word_exhausted | frame_end_here;
    // Word with tlast exhausted in the middle of a timestep: frame too short
    wire partial_step_drop  = word_exhausted & word_is_frame_last & ~step_complete;

    // A word is accepted if there is none in progress, or if the one in
    // progress ends in this cycle (no cycle lost between two words)
    assign s_axis_tready = ~word_valid | word_done;
    wire   in_word_accept = s_axis_tvalid & s_axis_tready;   // successful AXIS handshake

    // State of the current word
    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn) begin
            word_valid         <= 1'b0;
            word_is_frame_last <= 1'b0;
            word_sample_idx    <= {W_WORD_SMP_IDX{1'b0}};
        end else if (in_word_accept) begin
            // new word: restart from the sample in the low bits
            word_valid         <= 1'b1;
            word_is_frame_last <= s_axis_tlast;
            word_sample_idx    <= {W_WORD_SMP_IDX{1'b0}};
        end else if (word_done) begin
            word_valid         <= 1'b0;                    // no new word ready
        end else if (load_sample) begin
            word_sample_idx    <= word_sample_idx + 1'b1;  // go to the next sample
        end
    end

    // Word data: shift by N_BIT_IN bits at every consumed sample
    always @(posedge aclk) begin
        if (in_word_accept)
            word_shift_reg <= s_axis_tdata;
        else if (load_sample)
            word_shift_reg <= word_shift_reg >> N_BIT_IN;
    end

    // Timestep data: the new sample enters at the top, the others move down
    generate
        if (NIN == 1) begin : g_step_single
            always @(posedge aclk)
                if (load_sample) in_step_buf <= current_sample;
        end else begin : g_step_shift
            always @(posedge aclk)
                if (load_sample) in_step_buf <= {current_sample, in_step_buf[W_STEP-1:N_BIT_IN]};
        end
    endgenerate

    //=================================================================
    // 2. DOUBLE BUFFER AND CHANNEL SCAN
    //=================================================================
    reg [W_STEP-1:0]     cur_step_buf;             // NIN samples of the timestep being processed
    reg                  cur_buf_busy;             // 1 = channels remain to be launched
    reg                  cur_buf_is_last_step;     // 1 = it is the last timestep of the frame
    reg [LOG2H-1:0]      out_channel;              // next channel to launch = BRAM address

    wire out_channel_is_last = (out_channel == H-1);

    // Credits: results "in flight" towards the output
    reg  [W_CREDIT-1:0]  in_flight_count;          // in pipeline + in FIFO + output register
    wire                 credit_available = (in_flight_count < OUT_FIFO_DEPTH);

    // A channel is launched in S1 if there is one to do and a slot is guaranteed
    wire launch_channel = cur_buf_busy & credit_available;

    // Input -> computation transfer: input full and computation free,
    // or the last channel of the current timestep is just starting
    // (S1 reads cur_step_buf on this same edge: it still takes the old one)
    wire move_in_to_cur = in_buf_full & (~cur_buf_busy | (launch_channel & out_channel_is_last));

    // Output handshake (defined further below, needed by the credits)
    wire out_beat_done;

    // Sample counter and state of the input buffer
    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn) begin
            in_sample_idx       <= {W_STEP_SMP_IDX{1'b0}};
            in_buf_full         <= 1'b0;
            in_buf_is_last_step <= 1'b0;
            err_frame           <= 1'b0;
        end else begin
            // timestep copied into the computation buffer -> input free again
            // (exclusive with load_sample, which requires ~in_buf_full)
            if (move_in_to_cur) in_buf_full <= 1'b0;

            if (load_sample) begin
                if (step_complete) begin
                    // timestep complete: it waits for the computation
                    in_sample_idx       <= {W_STEP_SMP_IDX{1'b0}};
                    in_buf_full         <= 1'b1;
                    in_buf_is_last_step <= frame_end_here;
                end else if (partial_step_drop) begin
                    // frame ended in the middle of a timestep: the partial one is dropped
                    in_sample_idx       <= {W_STEP_SMP_IDX{1'b0}};
                    err_frame           <= 1'b1;
                end else begin
                    in_sample_idx       <= in_sample_idx + 1'b1;
                end
            end
        end
    end

    // Scan of channels 0..H-1 for the timestep in the computation buffer
    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn) begin
            cur_buf_busy <= 1'b0;
            out_channel  <= {LOG2H{1'b0}};
        end else if (move_in_to_cur) begin
            // new timestep: restart from channel 0
            cur_buf_busy <= 1'b1;
            out_channel  <= {LOG2H{1'b0}};
        end else if (launch_channel) begin
            if (out_channel_is_last) cur_buf_busy <= 1'b0;         // all H channels launched
            else                     out_channel  <= out_channel + 1'b1;
        end
    end

    // Copy of the timestep from the input buffer to the computation one
    always @(posedge aclk) begin
        if (move_in_to_cur) begin
            cur_step_buf         <= in_step_buf;
            cur_buf_is_last_step <= in_buf_is_last_step;
        end
    end

    // Credits: +1 when a channel is launched, -1 when a beat leaves on M_AXIS
    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn)
            in_flight_count <= {W_CREDIT{1'b0}};
        else if ( launch_channel & ~out_beat_done)
            in_flight_count <= in_flight_count + 1'b1;
        else if (~launch_channel &  out_beat_done)
            in_flight_count <= in_flight_count - 1'b1;
        // both or neither: the count does not change
    end

    //=================================================================
    // 3. PIPELINE
    //=================================================================

    //-----------------------------------------------------------------
    // Valid/last flags along the pipeline (they always flow, no enable)
    //   stage_valid[k] = stage Sk contains valid data
    //   stage_last[k]  = stage Sk contains the last beat of the frame
    //-----------------------------------------------------------------
    reg [PIPE_STAGES:1] stage_valid, stage_last;

    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn) begin
            stage_valid <= {PIPE_STAGES{1'b0}};
            stage_last  <= {PIPE_STAGES{1'b0}};
        end else begin
            stage_valid <= {stage_valid[PIPE_STAGES-1:1], launch_channel};
            stage_last  <= {stage_last[PIPE_STAGES-1:1],
                            launch_channel & cur_buf_is_last_step & out_channel_is_last};
        end
    end

    //-----------------------------------------------------------------
    // S1 + S2: Block RAM read in 2 cycles (internal latch + DOB_REG).
    //   Registers without reset and without enable: Vivado puts them inside
    //   the RAMB primitive. The samples are delayed in step.
    //-----------------------------------------------------------------
    reg        [NIN*W_WENC-1:0] s1_weight_row, s2_weight_row;   // NIN weights of the channel
    reg signed [W_BENC-1:0]     s1_bias,       s2_bias;         // bias of the channel
    reg        [W_STEP-1:0]     s1_samples,    s2_samples;      // NIN samples of the timestep

    always @(posedge aclk) begin
        s1_weight_row <= enc_weight_rom[out_channel];
        s1_bias       <= enc_bias_rom[out_channel];
        s1_samples    <= cur_step_buf;
        s2_weight_row <= s1_weight_row;
        s2_bias       <= s1_bias;
        s2_samples    <= s1_samples;
    end

    // Bias delayed up to S5: it enters the tree together with the products
    reg signed [W_BENC-1:0] s3_bias, s4_bias, s5_bias;

    always @(posedge aclk) begin
        s3_bias <= s2_bias;
        s4_bias <= s3_bias;
        s5_bias <= s4_bias;
    end

    //-----------------------------------------------------------------
    // Levels of the tree, all in a single flat bus:
    //   tree_bus[level*W_LEVEL_BUS + node*W_ENCACC +: W_ENCACC]
    //   level 0 = NIN products (nodes 0..NIN-1) + bias (node NIN), in S5
    //   level l = registered output of stage S(5+l)
    //-----------------------------------------------------------------
    wire [(TREE_LEVELS+1)*W_LEVEL_BUS-1:0] tree_bus;

    //-----------------------------------------------------------------
    // S3 + S4 + S5: NIN parallel multipliers (weight * sample)
    //   DSP48E2 with full pipeline: S3 registers A/B, S4 M, S5 P.
    //   Weight of W_WENC bits on port B, sample of W_X bits on port A.
    //-----------------------------------------------------------------
    genvar input_idx;   // input index (0..NIN-1)
    generate
        for (input_idx = 0; input_idx < NIN; input_idx = input_idx + 1) begin : g_mul
            // position of the weight of this input inside the BRAM row
            localparam integer WEIGHT_SLOT = C0_AT_LSB ? input_idx : (NIN - 1 - input_idx);

            wire [N_BIT_IN-1:0] raw_sample = s2_samples[input_idx*N_BIT_IN +: N_BIT_IN];

            (* use_dsp = "yes" *)
            reg  signed [W_WENC-1:0] s3_weight_b;    // -> B register of the DSP
            reg  signed [W_X-1:0]    s3_sample_a;    // -> A register of the DSP
            reg  signed [W_PROD-1:0] s4_product_m;   // -> M register of the DSP
            reg  signed [W_PROD-1:0] s5_product_p;   // -> P register of the DSP

            always @(posedge aclk) begin
                s3_weight_b  <= s2_weight_row[WEIGHT_SLOT*W_WENC +: W_WENC];
                // signed: direct copy; unsigned: a leading 0
                s3_sample_a  <= IN_SIGNED ? $signed(raw_sample) : $signed({1'b0, raw_sample});
                s4_product_m <= s3_weight_b * s3_sample_a;
                s5_product_p <= s4_product_m;
            end

            // level 0 of the tree: product sign-extended to W_ENCACC
            assign tree_bus[input_idx*W_ENCACC +: W_ENCACC] =
                {{(W_ENCACC-W_PROD){s5_product_p[W_PROD-1]}}, s5_product_p};
        end
    endgenerate

    // level 0, last node: sign-extended bias
    assign tree_bus[NIN*W_ENCACC +: W_ENCACC] = {{(W_ENCACC-W_BENC){s5_bias[W_BENC-1]}}, s5_bias};

    //-----------------------------------------------------------------
    // S6 .. S(5+TREE_LEVELS): generic binary sum tree.
    //   Each node adds two adjacent nodes of the previous level; if the
    //   previous level has an odd number of nodes, the last one passes
    //   forward only registered. One adder per register.
    //-----------------------------------------------------------------
    genvar level, node;
    generate
        for (level = 1; level <= TREE_LEVELS; level = level + 1) begin : g_tree_level
            localparam integer PREV_NODES = tree_nodes(level - 1);   // nodes of the level below
            localparam integer THIS_NODES = tree_nodes(level);       // nodes of this level
            localparam integer PREV_BASE  = (level - 1) * W_LEVEL_BUS;
            localparam integer THIS_BASE  = level * W_LEVEL_BUS;

            for (node = 0; node < TREE_ADDENDS; node = node + 1) begin : g_tree_node
                if (node < THIS_NODES) begin : g_used
                    wire signed [W_ENCACC-1:0] left_child =
                        tree_bus[PREV_BASE + (2*node)*W_ENCACC +: W_ENCACC];
                    reg  signed [W_ENCACC-1:0] partial_sum;

                    if (2*node + 1 < PREV_NODES) begin : g_add
                        wire signed [W_ENCACC-1:0] right_child =
                            tree_bus[PREV_BASE + (2*node+1)*W_ENCACC +: W_ENCACC];
                        always @(posedge aclk)
                            partial_sum <= left_child + right_child;   // a single adder
                    end else begin : g_pass
                        always @(posedge aclk)
                            partial_sum <= left_child;                 // odd node: register only
                    end

                    assign tree_bus[THIS_BASE + node*W_ENCACC +: W_ENCACC] = partial_sum;
                end else begin : g_unused
                    // empty slot of the bus: constant, Vivado removes it
                    assign tree_bus[THIS_BASE + node*W_ENCACC +: W_ENCACC] = {W_ENCACC{1'b0}};
                end
            end
        end
    endgenerate

    // root of the tree: sum_c W[ch][c]*x[c] + b[ch]
    wire signed [W_ENCACC-1:0] tree_root = tree_bus[TREE_LEVELS*W_LEVEL_BUS +: W_ENCACC];

    //-----------------------------------------------------------------
    // Last stage: rounding + shift towards F_ACT (one adder)
    //-----------------------------------------------------------------
    // output half LSB: added before the shift it gives round-half-up
    localparam signed [W_ENCACC-1:0] ROUND_HALF = (OUT_SHIFT == 0) ? 0 : (1 <<< (OUT_SHIFT-1));

    reg signed [W_ENCACC-1:0] rounded_value;          // value with F_ACT fractional bits

    always @(posedge aclk) begin
        // the arithmetic shift is just a rewiring
        rounded_value <= (tree_root + ROUND_HALF) >>> OUT_SHIFT;
    end

    //-----------------------------------------------------------------
    // Saturation to W_ACT bits (combinational, written into the FIFO).
    //   The value fits in W_ACT bits if the bits from the sign of W_ACT up
    //   are all equal: a wide AND/NOR, no comparisons with carry.
    //-----------------------------------------------------------------
    localparam integer W_HIGH_PART = W_ENCACC - W_ACT + 1;   // bits [W_ENCACC-1 : W_ACT-1]

    wire [W_HIGH_PART-1:0]  high_part       = rounded_value[W_ENCACC-1 : W_ACT-1];
    wire                    fits_in_w_act   = (&high_part) | (~|high_part);   // all 1s or all 0s
    wire                    out_overflow    = ~fits_in_w_act;
    wire                    value_is_neg    = rounded_value[W_ENCACC-1];

    wire signed [W_ACT-1:0] sat_max_value   = {1'b0, {(W_ACT-1){1'b1}}};      // maximum positive
    wire signed [W_ACT-1:0] sat_min_value   = {1'b1, {(W_ACT-1){1'b0}}};      // minimum negative
    wire signed [W_ACT-1:0] saturated_value = fits_in_w_act ? rounded_value[W_ACT-1:0]
                                            : value_is_neg  ? sat_min_value
                                            :                 sat_max_value;

    // Saturation flag: it rises if a valid datum of the last stage saturated
    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn)                                     sat_sticky <= 1'b0;
        else if (stage_valid[PIPE_STAGES] & out_overflow) sat_sticky <= 1'b1;
    end

    //=================================================================
    // 4. OUTPUT FIFO (OUT_FIFO_DEPTH x (W_ACT+1) bits, in LUTRAM)
    //   Word = {tlast, data}. It cannot overflow: the credits
    //   guarantee every launched channel an already reserved slot.
    //=================================================================
    (* ram_style = "distributed" *)
    reg [W_ACT:0] out_fifo_mem [0:OUT_FIFO_DEPTH-1];

    // pointers with one extra bit: equal = FIFO empty
    reg [W_FIFO_PTR:0] out_fifo_wr_ptr;
    reg [W_FIFO_PTR:0] out_fifo_rd_ptr;
    wire               out_fifo_empty = (out_fifo_wr_ptr == out_fifo_rd_ptr);

    // write: every valid datum leaving the last stage, already saturated
    wire out_fifo_write = stage_valid[PIPE_STAGES];

    always @(posedge aclk) begin
        if (out_fifo_write)
            out_fifo_mem[out_fifo_wr_ptr[W_FIFO_PTR-1:0]] <= {stage_last[PIPE_STAGES], saturated_value};
    end

    //-----------------------------------------------------------------
    // M_AXIS output register: tdata/tvalid/tlast come out of flip-flops
    //-----------------------------------------------------------------
    reg                    out_reg_valid;
    reg signed [W_ACT-1:0] out_reg_data;
    reg                    out_reg_last;

    assign out_beat_done = out_reg_valid & m_axis_tready;               // beat delivered
    // the register reloads if it is empty or if its datum is leaving now
    wire out_reg_load    = ~out_fifo_empty & (~out_reg_valid | m_axis_tready);

    always @(posedge aclk) begin
        if (out_reg_load)
            {out_reg_last, out_reg_data} <= out_fifo_mem[out_fifo_rd_ptr[W_FIFO_PTR-1:0]];
    end

    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn) begin
            out_fifo_wr_ptr <= {(W_FIFO_PTR+1){1'b0}};
            out_fifo_rd_ptr <= {(W_FIFO_PTR+1){1'b0}};
            out_reg_valid   <= 1'b0;
        end else begin
            if (out_fifo_write) out_fifo_wr_ptr <= out_fifo_wr_ptr + 1'b1;
            if (out_reg_load)   out_fifo_rd_ptr <= out_fifo_rd_ptr + 1'b1;

            if (out_reg_load)       out_reg_valid <= 1'b1;   // new datum loaded
            else if (out_beat_done) out_reg_valid <= 1'b0;   // delivered, nothing to reload
        end
    end

    //=================================================================
    // AXI-Stream output
    //=================================================================
    assign m_axis_tvalid = out_reg_valid;
    assign m_axis_tdata  = out_reg_data;
    assign m_axis_tlast  = out_reg_last;

endmodule

`default_nettype wire
