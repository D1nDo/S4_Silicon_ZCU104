`timescale 1ns / 1ps
`default_nettype none
//=====================================================================
// s4d_mixing_glu  --  linear mixing H -> 2H followed by GLU
//                     (SINGLE-PHASE version: NMAC = 2H = 256)
//
//   For every frame (H samples x[0..H-1]):
//
//     acc[c] = bias[c] + sum_{h<H} W[c][h] * x[h]        c = 0..2H-1
//     a[i]   = acc[i]          (first half)      i = 0..H-1
//     g[i]   = acc[H+i]        (second half)
//     out[i] = sat( a[i] * sigmoid(g[i]) )                (GLU)
//
//   Structure:
//
//     input --> g_buf (2 frames) --> MAC ENGINE --> acc[0..2H-1] --> SNAPSHOT --> GLU --> output
//                                    H cycles,      one MAC for      a_snap/g_snap  1 per cycle
//                                    3 stages       each output c    (shifting)     (credit-based)
//
//   - One MAC per output: MAC c updates only acc[c] and has its own
//     bias in a register -> no per-phase/per-output multiplexer.
//   - Weights: row h contains the 2H weights W[c][h] (c = 0 in the LSBs,
//     NMAC*W_W = 3072 bits), split into NBANK = 48 banks of 64 bits,
//     each H deep. One row read per cycle.
//   - MAC pipeline (7 stages): t addresses to the bank groups and x[h] read
//     -> t+1 addresses to the banks (x delayed) -> t+2 weight row read
//     -> t+3 row in the output register of the memory and x copied per
//     group of 16 MACs -> t+4 weight and x in the input registers of the MAC
//     -> t+5 registered product -> t+6 accumulation
//     (shape suited to the DSP48E2: A/B = inputs, M = product, P = acc).
//   - Snapshot: at the end of the frame acc is copied into two shift
//     registers; the GLU always reads the head (index 0). Meanwhile the MAC
//     engine processes the next frame.
//   Throughput: ~H + 7 = 135 cycles per frame with no output stall
//   (previous 4-phase version with NMAC=64: ~641 cycles).
//
//   2026-10-15 (fan-out of the output stall and URAM):
//     - FREE-RUNNING GLU branch: no more oen enable on Q1..G3,
//       sigmoid and output register. The stall acts only on the EMISSION
//       from the head of the snapshot, through credits: a result is emitted only if the
//       results "in flight" (emitted and not yet out of the output
//       queue of the layer, signalled by tail_pop) are fewer than
//       TAIL_DEPTH-2. The queue of the layer (TAIL_DEPTH slots) never overflows.
//       Before: oen = out_ready drove ~8000 enables (snapshot) + ~800
//       registers of the GLU pipeline and of the residual.
//     - snapshot: control replicated per groups of SNAP_GROUP positions
//       (registered copies of glu_active, glu_last and issue_ok, dont_touch),
//       so every enable drives 2*SNAP_GROUP*W_ACC flip-flops instead of
//       2*H*W_ACC = 8192.
//     - weights: with WMEM_STYLE = "ultra" the banks are instances of
//       xpm_memory_sdpram with MEMORY_PRIMITIVE = "ultra" (Vivado rejected
//       ram_style = "ultra" on the banks of 128 rows: Synth 8-6849, fallback
//       to LUTRAM, 48 x 20 RAM64M8). Read latency 2 (OREG register of the
//       URAM) for ALL the styles; stage A3 added to realign
//       x with the weights. In simulation without Xilinx libraries (Icarus) define
//       S4D_BEHAV_MEM: an equivalent behavioral model is used.
//     Identical arithmetic.
//
//   2026-10-15 (timing, at most one complex operation per cycle):
//     - input registers of the MACs (A/B of the DSP) between the weight row and
//       the multiplier;
//     - GLU branch: requantization in 2 stages (sum, saturation), the
//       sigmoid carries along a and the end-of-frame flag (PWL tag),
//       product a*sigmoid(g) in 3 stages (product, half LSB, saturation).
//     Identical arithmetic.
//
//   2026-10-02 (timing): local copies of x[h], of the MAC flags and of the
//   addresses/data of the weight banks, to reduce the fan-out measured
//   at 150 MHz (x[h] -> 256 DSPs, mac_h -> ~6150 cells).
//=====================================================================

module s4d_mixing_glu #(
    parameter integer H          = 128,   // inputs (and outputs of the GLU)
    parameter integer LOG2_H     = 7,
    parameter integer NMAC       = 256,   // = 2H: one MAC for each output, a single phase
    parameter integer LOG2_NPH   = 0,     // phases = 1 -> 0 bits (kept for compatibility with s4d_layer)
    parameter integer W_IN       = 16,    // input x (Q.F_IN)
    parameter integer F_IN       = 11,
    parameter integer W_W        = 12,    // weights (Q.F_W)
    parameter integer F_W        = 11,
    parameter integer W_ACC      = 32,    // accumulators and biases (Q.F_W+F_IN)
    parameter integer W_SIG      = 16,    // input/output of the sigmoid
    parameter integer F_SIGI     = 11,    // fractional bits of the gate g at the sigmoid input
    parameter integer F_SIGO     = 14,    // fractional bits of the sigmoid output
    parameter integer W_A        = 16,    // value a (Q.F_A)
    parameter integer F_A        = 11,
    parameter integer W_OUT      = 16,    // output (Q.F_OUT)
    parameter integer F_OUT      = 11,

    parameter integer LOADABLE   = 0,     // 1 = weights/biases written by the ld_* ports
    parameter integer LD_W       = 64,    // width of a weight bank (= beat of the stream)
    parameter integer LOG2_NBANK = 6,     // bits of ld_sel_w: NBANK = NMAC*W_W/LD_W = 48
    parameter         WMEM_STYLE = "ultra",  // weight memory: "ultra" | "block" | "distributed"
    parameter integer TAIL_DEPTH = 32,    // slots of the output queue of the layer (emission credits)
    parameter integer SNAP_GROUP = 2,     // snapshot positions served by one copy of the control
    // reused for NLAYER roles (see s4d_layer.v): default = layer 0, a real
    // file, not an invented placeholder -- whoever instantiates it always overrides
    parameter         W_FILE     = "l0_s4d_wout.mem",
    parameter         B_FILE     = "l0_s4d_bout.mem",
    parameter         SIG_FILE   = "sigmoid_lut.mem"
)(
    input  wire                        clk,
    input  wire                        rst_n,

    // ---- weight load: bank ld_sel_w of row ld_addr_w (= input h) ----
    input  wire                        ld_we_w,
    input  wire [LOG2_NPH+LOG2_H-1:0]  ld_addr_w,     // with LOG2_NPH=0: only h (0..H-1)
    input  wire [LOG2_NBANK-1:0]       ld_sel_w,      // 0..NBANK-1 (was 4 bits)
    input  wire [LD_W-1:0]             ld_data_w,
    // ---- load of the bias of output ld_addr_b (0..2H-1) ----
    input  wire                        ld_we_b,
    input  wire [LOG2_H:0]             ld_addr_b,
    input  wire signed [W_ACC-1:0]     ld_data_b,

    // ---- input: frame of H samples, in_last on the last one ----
    input  wire                        in_valid,
    output wire                        in_ready,
    input  wire signed [W_IN-1:0]      in_data,
    input  wire                        in_last,

    // ---- output: H samples per frame, WITHOUT ready (free-running pipeline) ----
    // tail_pop: pulse, a result has left the output queue of the
    // layer (gives back an emission credit)
    input  wire                        tail_pop,
    output reg                         out_valid,
    output reg  signed [W_OUT-1:0]     out_data,
    output reg                         out_last
);

    // ------------------------------------------------------------------
    // CREDIT-BASED BACKPRESSURE
    // The GLU branch (Q1, Q2, sigmoid, G1..G3, output) never stops:
    // every sample emitted from the head of the snapshot leaves after a
    // fixed latency and ends up, through the residual, in the output
    // queue of the layer (TAIL_DEPTH slots). Emission happens only if there is a
    // credit (issue_ok). If the output of the layer stalls, the queue
    // fills up, the credits run out and the snapshot stops; the MAC engine
    // then stays in MAC_DRAIN, nfull does not decrement and in_ready goes low
    // after two frames (same stall chain as the version with oen).
    // ------------------------------------------------------------------

    localparam integer TH    = 2*H;                // outputs of the linear part (256)
    localparam integer NBANK = (NMAC*W_W) / LD_W;  // 64-bit banks per row (48)
    localparam integer FACC  = F_W + F_IN;         // fractional bits of the accumulators
    localparam integer RQ_G  = FACC - F_SIGI;      // acc -> gate g shift (Q.F_SIGI)
    localparam integer RQ_A  = FACC - F_A;         // acc -> value a shift (Q.F_A)
    localparam integer SH_O  = F_A + F_SIGO - F_OUT;  // shift of a*sigmoid(g) -> Q.F_OUT

    // ---- parameter checks (this version is SINGLE-phase ONLY) ----
    initial begin
        if (NMAC != TH) begin
            $display("s4d_mixing_glu: requires NMAC = 2H (%0d), found %0d", TH, NMAC);
            $finish;
        end
        if (LOG2_NPH != 0) begin
            $display("s4d_mixing_glu: with a single phase LOG2_NPH must be 0");
            $finish;
        end
        if (NBANK * LD_W != NMAC * W_W) begin
            $display("s4d_mixing_glu: NMAC*W_W=%0d not a multiple of LD_W=%0d", NMAC*W_W, LD_W);
            $finish;
        end
        if ((1 << LOG2_NBANK) < NBANK) begin
            $display("s4d_mixing_glu: LOG2_NBANK=%0d too narrow for NBANK=%0d", LOG2_NBANK, NBANK);
            $finish;
        end
        if (TAIL_DEPTH < 4) begin
            $display("s4d_mixing_glu: TAIL_DEPTH=%0d too small (minimum 4)", TAIL_DEPTH);
            $finish;
        end
        if (SNAP_GROUP < 1 || (H % SNAP_GROUP) != 0) begin
            $display("s4d_mixing_glu: SNAP_GROUP=%0d must divide H=%0d", SNAP_GROUP, H);
            $finish;
        end
    end

    //-----------------------------------------------------------------
    // Bias: one per MAC, in REGISTERS. MAC c always reads bias_reg[c]
    // (constant index) -> no multiplexer on the read.
    //-----------------------------------------------------------------
    (* ram_style = "registers" *)
    reg signed [W_ACC-1:0] bias_reg [0:TH-1];

    generate
    if (LOADABLE == 0) begin : g_bias_init
        initial $readmemh(B_FILE, bias_reg);
    end else begin : g_bias_load
        always @(posedge clk)
            if (ld_we_b) bias_reg[ld_addr_b] <= ld_data_b;
    end
    endgenerate

    //-----------------------------------------------------------------
    // Weights: row h = the 256 weights W[c][h], c = 0 in the LSBs (3072 bits),
    // split into NBANK = 48 banks of 64 bits, each H = 128 deep.
    //
    // TIMING (150 -> 200+ MHz): before, the read address mac_h went
    // directly to all the 48 banks (fan-out ~6150, measured slack 0.62 ns)
    // and the write data/addresses ld_* to all the banks. Now:
    //   - every bank has ITS OWN registered copy of the read address and enable
    //     (bank_rd_*_q) and of the write ones (bank_we_q, bank_wr_addr_q);
    //   - the write data are copied once every WR_GROUP banks.
    // The copies have dont_touch, otherwise synthesis would merge them into a
    // single register (removal of equivalent registers).
    // 2026-10-05: 2-level TREE distribution of the addresses (before: 1 level).
    // Measured at 266.7 MHz: mac_h -> 48 copies bank_rd_addr_q slack +0.005 ns,
    // ld_wo_addr -> bank_wr_addr_q +0.008 ns (3.5 ns of wire only).
    //   level 1: NWR_GROUPS = 8 copies (one every WR_GROUP = 6 banks)
    //   level 2: one copy per bank
    // Read: address to the group at cycle t, to the bank at cycle t+1, row
    // read at cycle t+2 and registered again at cycle t+3 (wbank_q),
    // i.e. valid 4 cycles after w_rd_en. The second register is the OREG
    // of the URAM (or the output register of the BRAM): same latency for
    // all the styles, so the MAC pipeline does not depend on WMEM_STYLE.
    // Write: 2 cycles of delay with respect to the ld_* ports (irrelevant:
    // the weights are used only hundreds of cycles after the load).
    //-----------------------------------------------------------------
    wire                  w_rd_en;      // the MAC engine reads a row in this cycle
    wire [LOG2_H-1:0]     w_rd_addr;    // row = input h
    wire [NMAC*W_W-1:0]   w_row;        // row read, valid 4 cycles after w_rd_en

    // URAM only if the weights are written from the ports: the URAMs cannot be
    // initialized from the bitstream, so with LOADABLE=0 (weights from W_FILE)
    // the "ultra" style uses the BRAM initialized from file.
    localparam integer USE_URAM = (WMEM_STYLE == "ultra") && (LOADABLE != 0);
    localparam integer USE_BRAM = (WMEM_STYLE == "block") ||
                                  ((WMEM_STYLE == "ultra") && (LOADABLE == 0));

    localparam integer WR_GROUP    = 6;                             // banks served by one copy of the write data
    localparam integer NWR_GROUPS  = (NBANK + WR_GROUP - 1) / WR_GROUP;   // 8 copies

    // ---- level 1 of the tree: one copy per group of WR_GROUP banks ----
    // read: address and enable
    (* dont_touch = "true" *) reg              grp_rd_en_q   [0:NWR_GROUPS-1];
    (* dont_touch = "true" *) reg [LOG2_H-1:0] grp_rd_addr_q [0:NWR_GROUPS-1];
    // write: enable, selected bank, address
    (* dont_touch = "true" *) reg                  grp_we_q      [0:NWR_GROUPS-1];
    (* dont_touch = "true" *) reg [LOG2_NBANK-1:0] grp_sel_q     [0:NWR_GROUPS-1];
    (* dont_touch = "true" *) reg [LOG2_H-1:0]     grp_wr_addr_q [0:NWR_GROUPS-1];

    // write data: a single register (to align them with the addresses,
    // which go through 2 levels) and then one copy per group
    reg [LD_W-1:0] ld_data_pre_q;
    (* dont_touch = "true" *)
    reg [LD_W-1:0] ld_data_group_q [0:NWR_GROUPS-1];

    integer wr_group_idx;
    always @(posedge clk) begin
        ld_data_pre_q <= ld_data_w;
        for (wr_group_idx = 0; wr_group_idx < NWR_GROUPS; wr_group_idx = wr_group_idx + 1) begin
            grp_rd_en_q[wr_group_idx]     <= w_rd_en;
            grp_rd_addr_q[wr_group_idx]   <= w_rd_addr;
            grp_we_q[wr_group_idx]        <= ld_we_w;
            grp_sel_q[wr_group_idx]       <= ld_sel_w;
            grp_wr_addr_q[wr_group_idx]   <= ld_addr_w[LOG2_H-1:0];
            ld_data_group_q[wr_group_idx] <= ld_data_pre_q;
        end
    end

    genvar bank_idx;
    generate
    for (bank_idx = 0; bank_idx < NBANK; bank_idx = bank_idx + 1) begin : g_wbank

        wire [LD_W-1:0] wbank_q;        // output of the bank (2 registers after the address)

        // ---- local copies (one per bank) of addresses and enables ----
        (* dont_touch = "true" *) reg              bank_rd_en_q;     // read: enable
        (* dont_touch = "true" *) reg [LOG2_H-1:0] bank_rd_addr_q;   // read: row h
        (* dont_touch = "true" *) reg              bank_we_q;        // write: this bank
        (* dont_touch = "true" *) reg [LOG2_H-1:0] bank_wr_addr_q;   // write: row

        localparam integer MY_WR_GROUP = bank_idx / WR_GROUP;   // tree group of this bank

        // level 2 of the tree: copy of the group -> copy of the bank
        always @(posedge clk) begin
            bank_rd_en_q   <= grp_rd_en_q[MY_WR_GROUP];
            bank_rd_addr_q <= grp_rd_addr_q[MY_WR_GROUP];
            bank_we_q      <= grp_we_q[MY_WR_GROUP] && (grp_sel_q[MY_WR_GROUP] == bank_idx);
            bank_wr_addr_q <= grp_wr_addr_q[MY_WR_GROUP];
        end

        // write data of the group copy of this bank
        wire [LD_W-1:0] bank_wr_data = ld_data_group_q[MY_WR_GROUP];

        if (USE_URAM) begin : g_uram
`ifdef S4D_BEHAV_MEM
            // behavioral model (simulation without Xilinx libraries):
            // same behavior as the xpm below, latency 2
            reg [LD_W-1:0] wmem [0:H-1];
            reg [LD_W-1:0] uram_rd_q, uram_oreg_q;
            always @(posedge clk)
                if (bank_we_q) wmem[bank_wr_addr_q] <= bank_wr_data;
            always @(posedge clk) begin
                if (bank_rd_en_q) uram_rd_q <= wmem[bank_rd_addr_q];   // read (latch of the URAM)
                uram_oreg_q <= uram_rd_q;                              // OREG
            end
            assign wbank_q = uram_oreg_q;
`else
            // URAM explicitly instantiated: with ram_style = "ultra"
            // Vivado rejected the banks of 128 rows (Synth 8-6849) and
            // fell back to LUTRAM. One URAM288 (4096 x 72) per bank:
            // 48 URAMs out of 96 of the xczu7ev.
            xpm_memory_sdpram #(
                .ADDR_WIDTH_A       (LOG2_H),
                .ADDR_WIDTH_B       (LOG2_H),
                .AUTO_SLEEP_TIME    (0),
                .BYTE_WRITE_WIDTH_A (LD_W),           // write of the whole word
                .CLOCKING_MODE      ("common_clock"),
                .ECC_MODE           ("no_ecc"),
                .MEMORY_INIT_FILE   ("none"),
                .MEMORY_INIT_PARAM  ("0"),
                .MEMORY_OPTIMIZATION("true"),
                .MEMORY_PRIMITIVE   ("ultra"),
                .MEMORY_SIZE        (H*LD_W),         // total bits of the bank
                .MESSAGE_CONTROL    (0),
                .READ_DATA_WIDTH_B  (LD_W),
                .READ_LATENCY_B     (2),              // latch + OREG of the URAM
                .READ_RESET_VALUE_B ("0"),
                .RST_MODE_A         ("SYNC"),
                .RST_MODE_B         ("SYNC"),
                .SIM_ASSERT_CHK     (0),
                .USE_EMBEDDED_CONSTRAINT(0),
                .USE_MEM_INIT       (0),
                .WAKEUP_TIME        ("disable_sleep"),
                .WRITE_DATA_WIDTH_A (LD_W),
                .WRITE_MODE_B       ("read_first")     // the only one allowed with the URAMs
            ) u_wbank_uram (
                .clka           (clk),
                .ena            (1'b1),
                .wea            (bank_we_q),
                .addra          (bank_wr_addr_q),
                .dina           (bank_wr_data),
                .injectsbiterra (1'b0),
                .injectdbiterra (1'b0),
                .clkb           (clk),
                .rstb           (1'b0),
                .enb            (bank_rd_en_q),
                .regceb         (1'b1),
                .addrb          (bank_rd_addr_q),
                .doutb          (wbank_q),
                .sbiterrb       (),
                .dbiterrb       (),
                .sleep          (1'b0)
            );
`endif
        end else if (USE_BRAM) begin : g_bram
            (* ram_style = "block" *)       reg [LD_W-1:0] wmem [0:H-1];
            reg [LD_W-1:0] bram_rd_q, bram_oreg_q;   // read + output register of the BRAM
            if (LOADABLE != 0) begin : g_wr
                always @(posedge clk)
                    if (bank_we_q) wmem[bank_wr_addr_q] <= bank_wr_data;
            end else begin : g_init
                // weights from file: every row of the file has NMAC*W_W bits (single-row
                // format), bank bank_idx takes its 64-bit slice of it
                reg [NMAC*W_W-1:0] w_file [0:H-1];
                integer row_idx;
                initial begin
                    $readmemh(W_FILE, w_file);
                    for (row_idx = 0; row_idx < H; row_idx = row_idx + 1)
                        wmem[row_idx] = w_file[row_idx][bank_idx*LD_W +: LD_W];
                end
            end
            always @(posedge clk) begin
                if (bank_rd_en_q) bram_rd_q <= wmem[bank_rd_addr_q];
                bram_oreg_q <= bram_rd_q;                 // absorbed into the DOB_REG of the BRAM
            end
            assign wbank_q = bram_oreg_q;
        end else begin : g_lutram
            (* ram_style = "distributed" *) reg [LD_W-1:0] wmem [0:H-1];
            reg [LD_W-1:0] lut_rd_q, lut_oreg_q;      // read + second register
            if (LOADABLE != 0) begin : g_wr
                always @(posedge clk)
                    if (bank_we_q) wmem[bank_wr_addr_q] <= bank_wr_data;
            end
            always @(posedge clk) begin
                if (bank_rd_en_q) lut_rd_q <= wmem[bank_rd_addr_q];
                lut_oreg_q <= lut_rd_q;
            end
            assign wbank_q = lut_oreg_q;
        end

        assign w_row[bank_idx*LD_W +: LD_W] = wbank_q;
    end
    endgenerate

    //-----------------------------------------------------------------
    // Input double buffer: g_buf[{bank, h}] = x[h] (GELU output)
    //-----------------------------------------------------------------
    reg signed [W_IN-1:0] g_buf [0:2*H-1];
    reg                   wbank, rbank;   // bank being written / being read
    reg [LOG2_H-1:0]      icnt;           // sample being written
    reg [1:0]             nfull;          // complete frames not yet read by the MACs (0..2)

    assign in_ready = (nfull < 2'd2);     // from a register: no combinational path

    wire cap_fire = in_valid && in_ready;          // sample captured
    wire cap_done = cap_fire && (icnt == H-1);     // frame complete

    always @(posedge clk) if (cap_fire) g_buf[{wbank, icnt}] <= in_data;

    //=================================================================
    // MAC ENGINE: one pass of H cycles per frame, all the 256 MACs
    // in parallel, each on its own output c.
    //
    //   cycle t   : MAC_RUN, mac_h = h -> addresses copied into the groups of
    //               banks, x[h] read from g_buf (x_a)
    //   cycle t+1 : stage A2 -> addresses copied into the banks, x_a2 = x_a
    //   cycle t+2 : stage A3 -> weight row read (latch of the memory),
    //               x_a3 = x_a2
    //   cycle t+3 : stage B -> weight row in the output register of the
    //               memory (wbank_q), x_a3 copied into NMAC_GROUPS registers
    //               (x_b_copy), one per group of MACs
    //   cycle t+4 : stage B2 -> input registers of the MACs
    //   cycle t+5 : stage C -> registered product (prod_c)
    //   cycle t+6 : stage D -> acc[c] updated
    //   (stage A2 added on 2026-10-05 together with the tree of the bank
    //   addresses; stage A3 on 2026-10-15 for the OREG of the URAM: weights and x
    //   stay aligned)
    //
    // TIMING: before, x[h] came out of the BRAM of g_buf and went directly to
    // all the 256 DSPs (fan-out 3840, measured slack 0.138 ns at 150 MHz, 72%
    // routing). Now every copy x_b_copy[g] serves only MAC_GROUP = 16
    // MACs; the valid/first-input flags are also copied per group.
    // Cost: 1 more cycle of latency per frame, throughput unchanged.
    //=================================================================
    localparam [1:0] MAC_IDLE  = 2'd0,    // no frame to process
                     MAC_RUN   = 2'd1,    // H cycles: one input h per cycle
                     MAC_DRAIN = 2'd2;    // empties the pipeline and hands acc to the snapshot

    localparam integer MAC_GROUP   = 16;                 // MACs served by one copy of x
    localparam integer NMAC_GROUPS = NMAC / MAC_GROUP;   // 16 copies

    reg [1:0]        mac_state;
    reg [LOG2_H-1:0] mac_h;               // current input h (0..H-1)

    wire mac_last_h = (mac_state == MAC_RUN) && (mac_h == H-1);   // last input of the frame

    // control flags NOT replicated (needed only by acc_complete)
    reg mac_v_a, mac_first_a;             // stage A: x_a and addresses of the groups valid
    reg mac_v_a2, mac_first_a2;           // stage A2: x_a2 and addresses of the banks valid
    reg mac_v_a3, mac_first_a3;           // stage A3: x_a3 valid, weight row in the latch of the memory
    reg mac_v_b;                        // stage B: x_b_copy and w_row valid
    reg mac_v_b2;                         // stage B2: input registers of the MACs valid
    reg mac_v_c;                          // stage C: products valid (accumulation in progress)

    // data and flags replicated per group of MACs (dont_touch: do not merge the copies)
    reg signed [W_IN-1:0] x_a;                                        // x[h] from g_buf
    reg signed [W_IN-1:0] x_a2;                                       // x[h] aligned with the address tree
    reg signed [W_IN-1:0] x_a3;                                       // x[h] aligned with the memory read
    (* dont_touch = "true" *) reg signed [W_IN-1:0] x_b_copy        [0:NMAC_GROUPS-1];
    (* dont_touch = "true" *) reg                   mac_v_b_copy    [0:NMAC_GROUPS-1];
    (* dont_touch = "true" *) reg                   mac_first_b_copy[0:NMAC_GROUPS-1];
    (* dont_touch = "true" *) reg                   mac_v_b2_copy   [0:NMAC_GROUPS-1];
    (* dont_touch = "true" *) reg                   mac_first_b2_copy[0:NMAC_GROUPS-1];
    (* dont_touch = "true" *) reg                   mac_v_c_copy    [0:NMAC_GROUPS-1];
    (* dont_touch = "true" *) reg                   mac_first_c_copy[0:NMAC_GROUPS-1];

    // acc complete: we are in DRAIN and no update is in flight any more
    wire acc_complete = (mac_state == MAC_DRAIN) && !mac_v_a && !mac_v_a2 && !mac_v_a3
                      && !mac_v_b && !mac_v_b2 && !mac_v_c;

    // the snapshot can receive a new frame (assigned further below)
    wire snap_free;
    // hand-over: acc -> snapshot, in this cycle
    wire snap_take = acc_complete && snap_free;

    // read of the weight row: one cycle per input, only in MAC_RUN
    assign w_rd_en   = (mac_state == MAC_RUN);
    assign w_rd_addr = mac_h;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wbank     <= 1'b0;
            rbank     <= 1'b0;
            icnt      <= {LOG2_H{1'b0}};
            nfull     <= 2'd0;
            mac_state <= MAC_IDLE;
            mac_h     <= {LOG2_H{1'b0}};
        end else begin
            // ---- capture of the input ----
            if (cap_fire) begin
                if (icnt == H-1) begin
                    icnt  <= {LOG2_H{1'b0}};
                    wbank <= ~wbank;               // frame complete: switch bank
                end else begin
                    icnt <= icnt + 1'b1;
                end
            end

            // ---- state machine of the MAC engine ----
            case (mac_state)
                MAC_IDLE: if (nfull != 2'd0) begin
                    mac_state <= MAC_RUN;
                    mac_h     <= {LOG2_H{1'b0}};
                end

                MAC_RUN: begin
                    if (mac_h == H-1) begin
                        mac_state <= MAC_DRAIN;
                        rbank     <= ~rbank;       // next frame: the other bank
                    end else begin
                        mac_h <= mac_h + 1'b1;
                    end
                end

                MAC_DRAIN: if (snap_take) begin
                    // acc handed over: if there is already another frame we restart
                    // immediately (its first update of acc arrives 7 cycles
                    // later, so it does not touch the values just copied)
                    mac_h     <= {LOG2_H{1'b0}};
                    mac_state <= (nfull != 2'd0) ? MAC_RUN : MAC_IDLE;
                end

                default: mac_state <= MAC_IDLE;
            endcase

            // frames ready: +1 at the end of the capture, -1 when the MACs have
            // read the last input (the input bank becomes free again)
            case ({cap_done, mac_last_h})
                2'b10: nfull <= nfull + 2'd1;
                2'b01: nfull <= nfull - 2'd1;
                default: ;
            endcase
        end
    end

    // ---- control flags (not replicated) ----
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mac_v_a      <= 1'b0;
            mac_first_a  <= 1'b0;
            mac_v_a2     <= 1'b0;
            mac_first_a2 <= 1'b0;
            mac_v_a3     <= 1'b0;
            mac_first_a3 <= 1'b0;
            mac_v_b      <= 1'b0;
            mac_v_b2     <= 1'b0;
            mac_v_c      <= 1'b0;
        end else begin
            mac_v_a      <= (mac_state == MAC_RUN);
            mac_first_a  <= (mac_state == MAC_RUN) && (mac_h == 0);
            mac_v_a2     <= mac_v_a;
            mac_first_a2 <= mac_first_a;
            mac_v_a3     <= mac_v_a2;
            mac_first_a3 <= mac_first_a2;
            mac_v_b      <= mac_v_a3;
            mac_v_b2     <= mac_v_b;
            mac_v_c      <= mac_v_b2;
        end
    end

    // ---- flags replicated per group of MACs (they drive only the DSPs of the group) ----
    integer group_idx;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (group_idx = 0; group_idx < NMAC_GROUPS; group_idx = group_idx + 1) begin
                mac_v_b_copy[group_idx]     <= 1'b0;
                mac_first_b_copy[group_idx] <= 1'b0;
                mac_v_b2_copy[group_idx]    <= 1'b0;
                mac_first_b2_copy[group_idx] <= 1'b0;
                mac_v_c_copy[group_idx]     <= 1'b0;
                mac_first_c_copy[group_idx] <= 1'b0;
            end
        end else begin
            for (group_idx = 0; group_idx < NMAC_GROUPS; group_idx = group_idx + 1) begin
                mac_v_b_copy[group_idx]     <= mac_v_a3;
                mac_first_b_copy[group_idx] <= mac_first_a3;
                mac_v_b2_copy[group_idx]     <= mac_v_b_copy[group_idx];
                mac_first_b2_copy[group_idx] <= mac_first_b_copy[group_idx];
                mac_v_c_copy[group_idx]     <= mac_v_b2_copy[group_idx];
                mac_first_c_copy[group_idx] <= mac_first_b2_copy[group_idx];
            end
        end
    end

    // ---- data: x[h] from g_buf (stage A), delays (A2, A3), one copy per group (B) ----
    always @(posedge clk) begin                            // data: no reset
        x_a  <= g_buf[{rbank, mac_h}];
        x_a2 <= x_a;
        x_a3 <= x_a2;
    end

    always @(posedge clk)
        for (group_idx = 0; group_idx < NMAC_GROUPS; group_idx = group_idx + 1)
            x_b_copy[group_idx] <= x_a3;

    //-----------------------------------------------------------------
    // 256 MACs: registered product (stage C) and accumulation (stage D).
    // Structure designed for the DSP48E2: A=x, B=weight, M register = product,
    // P register = accumulator; the first input adds the bias (port C)
    // instead of P. MAC c uses the copy of its group: c / MAC_GROUP.
    //-----------------------------------------------------------------
    wire [TH*W_ACC-1:0] acc_flat;         // all the accumulators, output c in [c*W_ACC +: W_ACC]

    genvar mac_idx;
    generate
    for (mac_idx = 0; mac_idx < NMAC; mac_idx = mac_idx + 1) begin : g_mac

        localparam integer MY_GROUP = mac_idx / MAC_GROUP;                          // group of this MAC

        wire signed [W_W-1:0]      weight_c = $signed(w_row[mac_idx*W_W +: W_W]);   // W[c][h]
        reg  signed [W_W-1:0]      weight_in;                                       // B register of the DSP
        reg  signed [W_IN-1:0]     x_in;                                            // A register of the DSP
        reg  signed [W_W+W_IN-1:0] prod_c;                                          // W[c][h]*x[h]
        reg  signed [W_ACC-1:0]    acc_c;                                           // accumulator of output c

        // stage B2: input registers of the multiplier
        always @(posedge clk)
            if (mac_v_b_copy[MY_GROUP]) begin
                weight_in <= weight_c;
                x_in      <= x_b_copy[MY_GROUP];
            end

        // stage C: registered product
        always @(posedge clk)
            if (mac_v_b2_copy[MY_GROUP]) prod_c <= weight_in * x_in;

        // stage D: accumulation; first input of the frame -> restart from the bias
        always @(posedge clk)
            if (mac_v_c_copy[MY_GROUP])
                acc_c <= (mac_first_c_copy[MY_GROUP] ? bias_reg[mac_idx] : acc_c)
                       + {{(W_ACC-W_W-W_IN){prod_c[W_W+W_IN-1]}}, prod_c};

        assign acc_flat[mac_idx*W_ACC +: W_ACC] = acc_c;
    end
    endgenerate

    //=================================================================
    // SNAPSHOT: copy of the accumulators of a complete frame, read
    // by the GLU at the head (index 0) and shifted by 1 at every output.
    // This way the GLU uses no 128:1 multiplexers and the MAC engine can
    // already compute the next frame.
    //   a_snap[k] = acc[k]      (value a, first half)
    //   g_snap[k] = acc[H+k]    (gate g, second half)
    //
    // Credit-based emission: the head enters the GLU (glu_fire) only if
    // issue_ok, i.e. if the output queue of the layer has a guaranteed slot.
    //=================================================================
    // content of the snapshot, position k in [k*W_ACC +: W_ACC]; the real
    // registers are inside the groups (g_snap[*].a_q / g_q)
    wire [H*W_ACC-1:0] a_snap_flat;
    wire [H*W_ACC-1:0] g_snap_flat;

    // ---- "main" state of the control (used by MAC, Q1 and credits) ----
    reg              glu_active;          // 1 = the snapshot contains outputs not yet emitted
    reg [LOG2_H-1:0] glu_cnt;             // index of the output at the head (0..H-1)
    reg              glu_last;            // register: glu_cnt == H-1 (the head is the last of the frame)
    reg              issue_ok;            // register: there is an emission credit

    // ---- credits: results emitted and not yet out of the queue of the layer ----
    localparam integer CRED_W = $clog2(TAIL_DEPTH + 1);
    reg [CRED_W-1:0] inflight;

    // the head enters the GLU in this cycle
    wire glu_fire = glu_active && issue_ok;

    // the snapshot is free if empty, or if it is handing over its last output NOW
    assign snap_free = !glu_active || (glu_fire && glu_last);

    // ---- state of the next cycle: the same value goes to the main register
    //      and to all its per-group copies ----
    wire glu_active_next = snap_take              ? 1'b1        // new frame
                         : (glu_fire && glu_last) ? 1'b0        // last output emitted
                         :                          glu_active;
    wire glu_last_next   = (H == 1)   ? 1'b1
                         : snap_take  ? 1'b0                    // restart from glu_cnt = 0
                         : glu_fire   ? (glu_cnt == H-2)        // the next head is the last one
                         :              glu_last;
    // credit one cycle ahead of the count: issue_ok at cycle t
    // is inflight(t-1) < TAIL_DEPTH-2, so inflight(t) <= TAIL_DEPTH-2
    // and, after a possible emission, inflight <= TAIL_DEPTH-1: the queue
    // of the layer (TAIL_DEPTH slots) cannot overflow
    wire issue_ok_next   = (inflight < TAIL_DEPTH - 2);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            glu_active <= 1'b0;
            glu_cnt    <= {LOG2_H{1'b0}};
            glu_last   <= (H == 1);
            issue_ok   <= 1'b0;
            inflight   <= {CRED_W{1'b0}};
        end else begin
            glu_active <= glu_active_next;
            glu_last   <= glu_last_next;
            issue_ok   <= issue_ok_next;
            if (snap_take)
                glu_cnt <= {LOG2_H{1'b0}};                       // new frame
            else if (glu_fire)
                glu_cnt <= glu_last ? {LOG2_H{1'b0}} : (glu_cnt + 1'b1);
            // +1 for every emission, -1 for every result that left the layer
            inflight <= inflight + (glu_fire ? 1'b1 : 1'b0) - (tail_pop ? 1'b1 : 1'b0);
        end
    end

    //-----------------------------------------------------------------
    // Snapshot data, in groups of SNAP_GROUP positions.
    // Each group has ITS OWN registered copies of glu_active, glu_last and
    // issue_ok (same D input as the main registers, hence always
    // equal to them) and derives load and shift locally: every
    // enable drives 2*SNAP_GROUP*W_ACC flip-flops instead of 2*H*W_ACC.
    // dont_touch: without it, synthesis would merge the equivalent copies.
    //-----------------------------------------------------------------
    localparam integer NSNAP_GROUPS = H / SNAP_GROUP;

    genvar snap_grp;
    generate
    for (snap_grp = 0; snap_grp < NSNAP_GROUPS; snap_grp = snap_grp + 1) begin : g_snap

        (* dont_touch = "true" *) reg grp_active_q;   // copy of glu_active
        (* dont_touch = "true" *) reg grp_last_q;     // copy of glu_last
        (* dont_touch = "true" *) reg grp_issue_q;    // copy of issue_ok

        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                grp_active_q <= 1'b0;
                grp_last_q   <= (H == 1);
                grp_issue_q  <= 1'b0;
            end else begin
                grp_active_q <= glu_active_next;
                grp_last_q   <= glu_last_next;
                grp_issue_q  <= issue_ok_next;
            end
        end

        // same equations as glu_fire / snap_free / snap_take, on the copies
        wire grp_fire = grp_active_q && grp_issue_q;
        wire grp_take = acc_complete && (!grp_active_q || (grp_fire && grp_last_q));

        // the SNAP_GROUP positions of this group
        genvar pos_in_grp;
        for (pos_in_grp = 0; pos_in_grp < SNAP_GROUP; pos_in_grp = pos_in_grp + 1) begin : g_pos
            localparam integer SNAP_IDX = snap_grp*SNAP_GROUP + pos_in_grp;   // position 0..H-1

            reg [W_ACC-1:0] a_q;          // a_snap[SNAP_IDX]
            reg [W_ACC-1:0] g_q;          // g_snap[SNAP_IDX]

            // value entering with the shift: the next position
            // (the last position stays still, as in the previous version)
            wire [W_ACC-1:0] a_next_pos = (SNAP_IDX < H-1) ? a_snap_flat[((SNAP_IDX+1) % H)*W_ACC +: W_ACC] : a_q;
            wire [W_ACC-1:0] g_next_pos = (SNAP_IDX < H-1) ? g_snap_flat[((SNAP_IDX+1) % H)*W_ACC +: W_ACC] : g_q;

            always @(posedge clk) begin                    // data: no reset
                if (grp_take) begin
                    // parallel load of the frame just completed
                    a_q <= acc_flat[SNAP_IDX*W_ACC     +: W_ACC];
                    g_q <= acc_flat[(H+SNAP_IDX)*W_ACC +: W_ACC];
                end else if (grp_fire) begin
                    // shift towards the head
                    a_q <= a_next_pos;
                    g_q <= g_next_pos;
                end
            end

            assign a_snap_flat[SNAP_IDX*W_ACC +: W_ACC] = a_q;
            assign g_snap_flat[SNAP_IDX*W_ACC +: W_ACC] = g_q;
        end
    end
    endgenerate

    // head of the snapshot (position 0), read by Q1
    wire signed [W_ACC-1:0] a_head = $signed(a_snap_flat[0 +: W_ACC]);
    wire signed [W_ACC-1:0] g_head = $signed(g_snap_flat[0 +: W_ACC]);

    //-----------------------------------------------------------------
    // Requantization of the head of the snapshot (2 stages, free running):
    //   Q1  addition of the half LSB (if the shift is to the right)
    //   Q2  fixed shift (wires) and saturation to 16 signed bits
    // Same arithmetic as the old rq function: sum on W_ACC bits,
    // then (v + 2^(s-1)) >>> s for s > 0, or v <<< -s for s <= 0.
    //-----------------------------------------------------------------
    localparam signed [W_ACC-1:0] RND_G = (RQ_G > 0) ? (1 <<< (RQ_G - 1)) : 0;
    localparam signed [W_ACC-1:0] RND_A = (RQ_A > 0) ? (1 <<< (RQ_A - 1)) : 0;
    localparam signed [W_ACC-1:0] SIG_HI = (1 <<< (W_SIG-1)) - 1;     // limits at W_SIG bits
    localparam signed [W_ACC-1:0] SIG_LO = -(1 <<< (W_SIG-1));
    localparam signed [W_ACC-1:0] A_HI   = (1 <<< (W_A-1)) - 1;       // limits at W_A bits
    localparam signed [W_ACC-1:0] A_LO   = -(1 <<< (W_A-1));

    // shift right (s > 0) or left (s <= 0) by a fixed amount
    function signed [W_ACC-1:0] fixed_shift;
        input signed [W_ACC-1:0] v;
        input integer            s;
        begin
            fixed_shift = (s > 0) ? (v >>> s) : (v <<< (-s));
        end
    endfunction

    // ---- Q1: addition of the half LSB ----
    reg                    q1_valid, q1_last;
    reg signed [W_ACC-1:0] q1_g, q1_a;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            q1_valid <= 1'b0;
            q1_last  <= 1'b0;
        end else begin
            q1_valid <= glu_fire;                // the head has been emitted
            q1_last  <= glu_fire && glu_last;
        end
    end

    always @(posedge clk) begin
        q1_g <= g_head + RND_G;
        q1_a <= a_head + RND_A;
    end

    // ---- Q2: fixed shift and saturation ----
    wire signed [W_ACC-1:0] q1_g_sh = fixed_shift(q1_g, RQ_G);
    wire signed [W_ACC-1:0] q1_a_sh = fixed_shift(q1_a, RQ_A);

    reg                    q2_valid, q2_last;
    reg signed [W_SIG-1:0] g_q;                     // gate in Q.F_SIGI
    reg signed [W_A-1:0]   a_q;                     // value in Q.F_A

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            q2_valid <= 1'b0;
            q2_last  <= 1'b0;
        end else begin
            q2_valid <= q1_valid;
            q2_last  <= q1_last;
        end
    end

    always @(posedge clk) begin
        if      (q1_g_sh > SIG_HI) g_q <= SIG_HI[W_SIG-1:0];
        else if (q1_g_sh < SIG_LO) g_q <= SIG_LO[W_SIG-1:0];
        else                       g_q <= q1_g_sh[W_SIG-1:0];
        if      (q1_a_sh > A_HI)   a_q <= A_HI[W_A-1:0];
        else if (q1_a_sh < A_LO)   a_q <= A_LO[W_A-1:0];
        else                       a_q <= q1_a_sh[W_A-1:0];
    end

    wire                    sig_valid;
    wire                    sig_in_ready;    // always 1 (out_ready = 1), no need to consult it
    wire signed [W_SIG-1:0] sig_out;         // sigmoid(g) in Q.F_SIGO
    wire signed [W_A-1:0]   sig_a;           // a, arrived together with sigmoid(g)
    wire                    sig_last;        // last sample of the frame

    //=================================================================
    // Piecewise-linear approximated sigmoid (256 segments in [-8, 8)),
    // outside the interval: 0 on the left, 1.0 on the right. The tag carries a and the
    // end-of-frame flag through its pipeline (before: chains a_d[] and
    // last_d[] to be kept aligned by hand with the latency of the PWL).
    //=================================================================
    s4d_pwl #(
        .W_IN(W_SIG), .F_IN(F_SIGI), .W_OUT(W_SIG), .F_OUT(F_SIGO),
        .NSEG(256), .LOG2_NSEG(8), .XH(3),
        .MODE_HI(0), .Y_LO_Q(0), .Y_HI_Q(1 << F_SIGO),
        .TAG_W(W_A + 1),
        .LUT_FILE(SIG_FILE)
    ) u_sig (
        .clk       (clk),
        .rst_n     (rst_n),
        .in_valid  (q2_valid),
        .in_ready  (sig_in_ready),
        .in_data   (g_q),
        .in_tag    ({q2_last, a_q}),
        .out_ready (1'b1),                    // free pipeline: no stall
        .out_valid (sig_valid),
        .out_data  (sig_out),
        .out_tag   ({sig_last, sig_a})
    );

    //-----------------------------------------------------------------
    // GLU product a * sigmoid(g), rounding to Q.F_OUT, saturation
    // (3 stages, free running):
    //   G1  product a * sigmoid(g)             (M register of the DSP)
    //   G2  + half LSB                         (P register of the DSP)
    //   G3  fixed shift (wires), saturation -> output register
    // No overflow possible: |a*sig| <= 2^30 on PWO = 32 bits.
    //-----------------------------------------------------------------
    localparam integer PWO = W_A + W_SIG;
    localparam signed [PWO-1:0] OMAX = {{(PWO-W_OUT+1){1'b0}}, {(W_OUT-1){1'b1}}};
    localparam signed [PWO-1:0] OMIN = {{(PWO-W_OUT+1){1'b1}}, {(W_OUT-1){1'b0}}};
    localparam signed [PWO-1:0] PRND = {{(PWO-1){1'b0}}, 1'b1} <<< (SH_O-1);   // output half LSB

    reg                  g1_valid, g1_last, g2_valid, g2_last;
    reg signed [PWO-1:0] g1_prod, g2_sum;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            g1_valid <= 1'b0; g1_last <= 1'b0;
            g2_valid <= 1'b0; g2_last <= 1'b0;
        end else begin
            g1_valid <= sig_valid;  g1_last <= sig_valid && sig_last;
            g2_valid <= g1_valid;   g2_last <= g1_last;
        end
    end

    always @(posedge clk) begin
        g1_prod <= sig_a * sig_out;                 // Q.F_A+F_SIGO
        g2_sum  <= g1_prod + PRND;
    end

    wire signed [PWO-1:0] pout = g2_sum >>> SH_O;

    // output register (free running: the layer always accepts the data)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            out_data  <= {W_OUT{1'b0}};
            out_last  <= 1'b0;
        end else begin
            out_valid <= g2_valid;
            out_last  <= g2_last;

            if      (pout > OMAX) out_data <= OMAX[W_OUT-1:0];
            else if (pout < OMIN) out_data <= OMIN[W_OUT-1:0];
            else                  out_data <= pout[W_OUT-1:0];
        end
    end

    // simulation-only check: in_last must fall on the H-th sample
// synthesis translate_off
    always @(posedge clk)
        if (cap_fire && (in_last != (icnt == H-1))) begin
            $display("s4d_mixing_glu: in_last at icnt=%0d, expected at %0d -- the", icnt, H-1);
            $display("  input frame is not H long: synchronization broken");
            $finish;
        end
// synthesis translate_on

endmodule

`default_nettype wire
