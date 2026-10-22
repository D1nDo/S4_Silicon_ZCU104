`timescale 1ns / 1ps
`default_nettype none

// ============================================================================
// s4d_weightrom_axis -- weights of ALL the layers fixed in an on-chip ROM (BRAM),
// sent as a 64-bit AXI-Stream master, one beat at a time, to the s_axis_w_*
// input of s4d_core_axis. It replaces the weight DMA.
//
// ROM CONTENT: w_stream_nmac256_<DS>.mem (stream with the mixing weights in
// the single-phase format, NMAC=256), NLAYER*NBEAT rows of 64 bits:
//     rows [0       .. NBEAT-1]      -> layer 0
//     rows [NBEAT   .. 2*NBEAT-1]    -> layer 1
//     ...
//     rows [5*NBEAT .. 6*NBEAT-1]    -> layer 5
// Row i is exactly the tdata that the DMA would have put on beat i.
//
// OPERATION (version 2026-10-02, pipelined for timing):
//
//     pointer --> ROM (BRAM) --> rom_q1 --> rom_q2 --> queue (8 slots) --> m_axis
//     rd_rom_addr  read           BRAM reg   BRAM output  fifo_*            tdata = head
//                  ALWAYS active             reg (DOB)
//
//   - The ROM is read every cycle, without clock enable: before, the enable
//     was (!tvalid || tready), i.e. the tready of the core reached all the
//     176 BRAMs (measured slack 0.57 ns at 177.8 MHz). Now tready only
//     drives the extraction from the queue (a few registers).
//   - A read is "issued" (the pointer advances and the data will be put in
//     the queue 2 cycles later) only if there is a guaranteed slot in the
//     queue: counter read_credits = reads in flight + words in queue < 8.
//     This way the queue never overflows, even if the core stops tready.
//   - Steady state: one beat per cycle (issue -> queue -> extraction loop of
//     4 cycles, 8 credits).
//   - rd_rom_addr has max_fanout: synthesis makes copies of it for groups
//     of BRAMs (before: address -> 176 BRAMs, slack 0.64 ns).
//
// REALIGNMENT PER LAYER (wload_start / wload_layer):
//   The core signals with a wload_start pulse (same cycle as
//   u_wload.start) the start of the loading of a layer, and with wload_layer
//   WHICH layer. On that pulse the streamer:
//     - moves the pointer to the start of that layer (wload_layer*NBEAT);
//     - empties the queue and discards the reads in flight;
//     - issues no read in that cycle.
//   Timing: in the cycle of the pulse s4d_wload still has tready=0 (busy goes
//   high the next cycle), so no beat is taken by the core during the
//   flush; from the next cycle the core has tready=1 with tvalid=0 (legal,
//   it waits) and the first row of the layer is at the head of the queue 4
//   cycles after the pulse.
//   Fallback: without pulses, after the last beat of layer 5 the pointer
//   goes back to 0 by itself (same order in which the core iterates).
//
// RESET: local reset bridge (asynchronous assertion, synchronous release
// after 2 cycles), identical to the bridges of s4d_core_axis: core and
// streamer leave reset in the same cycle.
//
// Resource estimate: 89862 x 64 bits ~ 5.75 Mbit ~ 176 BRAM36 (ZCU104: 312).
// URAM cannot be used here: it cannot be initialized from the bitstream.
// ============================================================================
module s4d_weightrom_axis #(
    parameter integer NBEAT        = 14977,   // beats per layer (= NBEAT of s4d_wload)
    parameter integer NLAYER       = 6,       // fixed by the architecture
    parameter integer AXI_DW       = 64,      // fixed by the architecture
    parameter         WEIGHTS_FILE = "w_stream_nmac256_ECG.mem"   // NMAC=256 format
)(
    input  wire                  clk,          // same clock as the core
    input  wire                  rst_n,        // same reset as the core

    // ---- realignment from the core (s4d_core_axis) ----
    input  wire                        wload_start,  // pulse: the core starts loading a layer
    input  wire [$clog2(NLAYER)-1:0]   wload_layer,  // layer the core is about to load (0..NLAYER-1)

    // ---- AXI-Stream master towards s4d_core_axis.s_axis_w_* ----
    output wire                  m_axis_tvalid,
    input  wire                  m_axis_tready,
    output wire [AXI_DW-1:0]     m_axis_tdata,
    output wire                  m_axis_tlast,  // 1 on the last beat of each layer
                                                // (ignored by the core: diagnostics only)

    // ---- diagnostics (can be left unconnected) ----
    output wire                        layer_done,  // pulse: last beat of a layer consumed
    output wire [$clog2(NLAYER)-1:0]   layer_idx    // layer of the beat at the head
);

    // ------------------------------------------------------------------
    // Derived sizes
    // ------------------------------------------------------------------
    localparam integer TOTAL_BEATS   = NLAYER * NBEAT;          // ROM rows (89862)
    localparam integer ROM_ADDR_W    = $clog2(TOTAL_BEATS);     // 17 bits
    localparam integer BEAT_IN_LAY_W = $clog2(NBEAT);           // 14 bits
    localparam integer LAYER_W       = $clog2(NLAYER);          // 3 bits

    localparam integer FIFO_DEPTH    = 8;                       // slots of the output queue
    localparam integer FIFO_IDX_W    = 3;                       // bits of the queue indices
    localparam integer COUNT_W       = 4;                       // bits of the counters (0..8)

    // ------------------------------------------------------------------
    // Local reset bridge (see the header)
    // ------------------------------------------------------------------
    (* keep = "true" *) reg [1:0] rst_bridge;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) rst_bridge <= 2'b00;
        else        rst_bridge <= {rst_bridge[0], 1'b1};
    end
    wire rst_local_n = rst_bridge[1];

    // ------------------------------------------------------------------
    // Weight ROM: initialized from file, forced to BRAM
    // ------------------------------------------------------------------
    (* rom_style = "block" *)
    reg [AXI_DW-1:0] weight_rom [0:TOTAL_BEATS-1];
    initial $readmemh(WEIGHTS_FILE, weight_rom);

    // ------------------------------------------------------------------
    // Read pointer: the next beat to issue. Three aligned counters
    // (absolute address, position in the layer, layer) so that
    // tlast/layer_idx need no divisions.
    // ------------------------------------------------------------------
    //
    // 2026-10-05 (timing): the pointer NO longer has the asynchronous reset.
    // The copies of rd_rom_addr created by max_fanout sit close to the 176
    // BRAMs; the reset had to reach all of them (measured slack +0.007 ns at
    // 266.7 MHz). It is not needed: before the core asks for the weights a
    // wload_start always arrives, which realigns the pointer. The initial
    // value (0) is given by the register initialization at FPGA
    // configuration (and it also holds in simulation).
    (* max_fanout = 32 *)
    reg [ROM_ADDR_W-1:0]    rd_rom_addr      = {ROM_ADDR_W{1'b0}};      // 0 .. TOTAL_BEATS-1
    reg [BEAT_IN_LAY_W-1:0] rd_beat_in_layer = {BEAT_IN_LAY_W{1'b0}};   // 0 .. NBEAT-1
    reg [LAYER_W-1:0]       rd_layer         = {LAYER_W{1'b0}};         // 0 .. NLAYER-1

    wire rd_is_last_of_layer = (rd_beat_in_layer == NBEAT - 1);
    wire rd_is_last_of_rom   = (rd_rom_addr == TOTAL_BEATS - 1);

    // Address of the first row of the layer requested by the core:
    // wload_layer * NBEAT (multiplication by a constant -> few LUTs).
    wire [ROM_ADDR_W-1:0] requested_layer_base = wload_layer * NBEAT;

    // An out-of-range layer index (6 or 7 on 3 bits) must not move the
    // pointer outside the ROM: in that case the pulse is ignored.
    wire resync_to_layer = wload_start && (wload_layer < NLAYER);

    // ------------------------------------------------------------------
    // Output queue
    // ------------------------------------------------------------------
    reg [AXI_DW-1:0]     fifo_data  [0:FIFO_DEPTH-1];   // beat
    reg                  fifo_last  [0:FIFO_DEPTH-1];   // last beat of its layer
    reg [LAYER_W-1:0]    fifo_layer [0:FIFO_DEPTH-1];   // layer of the beat
    reg [FIFO_IDX_W-1:0] fifo_wr_idx, fifo_rd_idx;      // queue pointers
    reg [COUNT_W-1:0]    fifo_count;                    // words in the queue (0..8)
    reg [COUNT_W-1:0]    read_credits;                  // reads in flight + words in the queue (0..8)

    wire out_valid     = (fifo_count != {COUNT_W{1'b0}});   // there is a beat at the head
    wire out_handshake = out_valid && m_axis_tready;        // the core takes it now

    // a read is issued only with a guaranteed slot in the queue and never in
    // the cycle of the realignment (the pointer is changing)
    wire issue_read = (read_credits < FIFO_DEPTH) && !resync_to_layer;

    // ------------------------------------------------------------------
    // ROM read: 2 registered stages, ALWAYS enabled, no reset
    //   rom_q1: BRAM read register (address = rd_rom_addr)
    //   rom_q2: BRAM output register (DOB_REG)
    // ------------------------------------------------------------------
    reg [AXI_DW-1:0] rom_q1, rom_q2;
    always @(posedge clk) begin
        rom_q1 <= weight_rom[rd_rom_addr];
        rom_q2 <= rom_q1;
    end

    // labels that travel with the data through the 2 stages
    reg               read_valid_1, read_valid_2;     // the data in the stage is an issued read
    reg               read_last_1,  read_last_2;      // last beat of the layer
    reg [LAYER_W-1:0] read_layer_1, read_layer_2;     // layer of the beat

    wire fifo_push = read_valid_2 && !resync_to_layer;   // the data of rom_q2 enters the queue

    // ------------------------------------------------------------------
    // Control: pointer, read stages, queue, credits
    // ------------------------------------------------------------------
    always @(posedge clk or negedge rst_local_n) begin
        if (!rst_local_n) begin
            read_valid_1     <= 1'b0;
            read_valid_2     <= 1'b0;
            read_last_1      <= 1'b0;
            read_last_2      <= 1'b0;
            read_layer_1     <= {LAYER_W{1'b0}};
            read_layer_2     <= {LAYER_W{1'b0}};
            fifo_wr_idx      <= {FIFO_IDX_W{1'b0}};
            fifo_rd_idx      <= {FIFO_IDX_W{1'b0}};
            fifo_count       <= {COUNT_W{1'b0}};
            read_credits     <= {COUNT_W{1'b0}};
        end else if (resync_to_layer) begin
            // Realignment: pointer to the first row of the requested layer,
            // reads in flight discarded, queue emptied (the core in this
            // cycle has tready=0, so it is not taking any beat).
            // The pointer is realigned in the block without reset, below.
            read_valid_1     <= 1'b0;
            read_valid_2     <= 1'b0;
            fifo_wr_idx      <= {FIFO_IDX_W{1'b0}};
            fifo_rd_idx      <= {FIFO_IDX_W{1'b0}};
            fifo_count       <= {COUNT_W{1'b0}};
            read_credits     <= {COUNT_W{1'b0}};
        end else begin

            // ---- labels along the 2 read stages ----
            read_valid_1 <= issue_read;
            read_last_1  <= rd_is_last_of_layer;
            read_layer_1 <= rd_layer;
            read_valid_2 <= read_valid_1;
            read_last_2  <= read_last_1;
            read_layer_2 <= read_layer_1;

            // ---- queue: push (from the second stage) and pop (by the core) ----
            if (fifo_push)     fifo_wr_idx <= fifo_wr_idx + 1'b1;
            if (out_handshake) fifo_rd_idx <= fifo_rd_idx + 1'b1;

            fifo_count   <= fifo_count   + (fifo_push  ? 1'b1 : 1'b0) - (out_handshake ? 1'b1 : 1'b0);
            read_credits <= read_credits + (issue_read ? 1'b1 : 1'b0) - (out_handshake ? 1'b1 : 1'b0);
        end
    end

    // ------------------------------------------------------------------
    // Read pointer (WITHOUT reset, see the declaration):
    //   realignment -> first row of the requested layer;
    //   issue       -> next beat, back to 0 after the last one.
    // ------------------------------------------------------------------
    always @(posedge clk) begin
        if (resync_to_layer) begin
            rd_rom_addr      <= requested_layer_base;
            rd_beat_in_layer <= {BEAT_IN_LAY_W{1'b0}};
            rd_layer         <= wload_layer;
        end else if (issue_read) begin
            rd_rom_addr <= rd_is_last_of_rom ? {ROM_ADDR_W{1'b0}}
                                             : rd_rom_addr + 1'b1;
            if (rd_is_last_of_layer) begin
                // end of layer: position to 0, next layer (5 -> 0)
                rd_beat_in_layer <= {BEAT_IN_LAY_W{1'b0}};
                rd_layer         <= (rd_layer == NLAYER - 1) ? {LAYER_W{1'b0}}
                                                             : rd_layer + 1'b1;
            end else begin
                rd_beat_in_layer <= rd_beat_in_layer + 1'b1;
            end
        end
    end

    // queue data (no reset: they matter only through fifo_count)
    always @(posedge clk)
        if (fifo_push) begin
            fifo_data[fifo_wr_idx]  <= rom_q2;
            fifo_last[fifo_wr_idx]  <= read_last_2;
            fifo_layer[fifo_wr_idx] <= read_layer_2;
        end

    // ------------------------------------------------------------------
    // Outputs: head of the queue
    // ------------------------------------------------------------------
    assign m_axis_tvalid = out_valid;
    assign m_axis_tdata  = fifo_data[fifo_rd_idx];
    assign m_axis_tlast  = out_valid && fifo_last[fifo_rd_idx];

    // pulse when the core takes the last beat of a layer
    assign layer_done = out_handshake && fifo_last[fifo_rd_idx];
    assign layer_idx  = fifo_layer[fifo_rd_idx];

endmodule

`default_nettype wire
