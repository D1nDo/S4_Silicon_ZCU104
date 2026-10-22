`timescale 1ns / 1ps
`default_nettype none
//=====================================================================
// s4d_biquad_bank  --  S4D convolution of one frame (H channels)
//
//   For each channel ch:
//     y[ch] = D[ch]*u[ch] + sum_{k<MODES} biquad_k(u[ch])
//   where each biquad_k is a pair of complex-conjugate poles of the
//   S4D kernel, implemented as a 2nd-order IIR filter with state.
//
//   Organization:
//     - NUNITS units (s4d_biquad_unit) work in parallel and in
//       lock-step; unit j serves the channels j*NCH .. j*NCH+NCH-1.
//     - Each unit scans NB = NCH*MODES biquads, one per cycle:
//       with NUNITS=32, NCH=4 -> NB = 128, one pass (one frame) takes
//       ~128 cycles + latency (with NUNITS=8, NCH=16 it was 512).
//     - Double-buffered input (ubuf): while the units process one
//       frame, the next one is collected in the other half.
//     - The results (one per channel) are collected in res; at the end
//       of the pass res is copied into the shift register out_shift
//       and emitted one channel per cycle, with rounding (F_Y -> F_OUT)
//       and saturation to W_OUT bits. The emission overlaps the next
//       pass (res_pending prevents res from being overwritten before
//       the copy).
//
//   State reset: the first frame of each new sequence starts with
//   the biquad state at zero (clr), see clr_next / buf_lastseq.
//=====================================================================

module s4d_biquad_bank #(
    parameter integer H        = 128,   // channels
    parameter integer LOG2_H   = 7,
    parameter integer NUNITS   = 8,     // parallel units
    parameter integer LOG2_NU  = 3,
    parameter integer MODES    = 32,    // biquads (modes) per channel
    parameter integer LOG2_MOD = 5,
    parameter integer NCH      = 16,    // channels per unit (NUNITS*NCH = H)
    parameter integer LOG2_NCH = 4,

    parameter integer LOG2_NB  = LOG2_MOD + LOG2_NCH,   // address bits in the unit
    parameter integer W_U      = 16,    // input u
    parameter integer F_U      = 11,
    parameter integer W_A      = 24,    // coefficients a1,a2
    parameter integer F_A      = 22,
    parameter integer W_B      = 16,    // coefficients b0,b1
    parameter integer F_B      = 15,
    parameter integer W_S      = 32,    // state / output of a biquad
    parameter integer F_Y      = 19,
    parameter integer W_D      = 16,    // coefficient D
    parameter integer F_D      = 14,

    parameter integer W_BSH    = 5,     // shift of the b coefficients
    parameter integer W_ACC    = 40,    // per-channel accumulator (Q.F_Y)
    parameter integer W_OUT    = 16,    // bank output
    parameter integer F_OUT    = 11
)(
    input  wire                    clk,
    input  wire                    rst_n,

    // ---- write of the coefficients {b1,b0,a2,a1} into unit cfg_unit ----
    input  wire                    cfg_we,
    input  wire [LOG2_NU-1:0]      cfg_unit,
    input  wire [LOG2_NB-1:0]      cfg_addr,
    input  wire [2*W_A+2*W_B-1:0]  cfg_data,

    // ---- write of D and of the shift of the b of channel cfgd_ch ----
    input  wire                    cfgd_we,
    input  wire [LOG2_H-1:0]       cfgd_ch,
    input  wire signed [W_D-1:0]   cfgd_data,
    input  wire [W_BSH-1:0]        cfgd_bsh,

    // ---- input: H samples per frame ----
    input  wire                    in_valid,
    output wire                    in_ready,
    input  wire signed [W_U-1:0]   in_data,
    input  wire                    in_last_seq,  // the frame is the last of the sequence

    // ---- output: H samples per frame ----
    output reg                     out_valid,
    input  wire                    out_ready,
    output reg  signed [W_OUT-1:0] out_data,
    output reg  [LOG2_H-1:0]       out_ch,       // channel of the output sample
    output reg                     out_last,     // last channel of the frame

    output wire                    ovf           // overflow in at least one unit (sticky)
);

    localparam integer SH_O = F_Y - F_OUT;   // output shift: Q.F_Y -> Q.F_OUT

    // saturation limits at W_OUT bits, extended to W_ACC
    localparam signed [W_ACC-1:0] OMAX = {{(W_ACC-W_OUT+1){1'b0}}, {(W_OUT-1){1'b1}}};
    localparam signed [W_ACC-1:0] OMIN = {{(W_ACC-W_OUT+1){1'b1}}, {(W_OUT-1){1'b0}}};

    //-----------------------------------------------------------------
    // Double input buffer: 2 frames of H samples (in registers)
    //   half wsel : being written
    //   half rsel : being read by the units
    //-----------------------------------------------------------------
    reg [2*H*W_U-1:0] ubuf;
    reg               wsel, rsel;    // half being written / being read
    reg [LOG2_H-1:0]  wcnt;          // channel being written
    reg               buf_full;      // 1 = a complete frame is waiting to be processed

    // input blocked while a complete frame is waiting
    assign in_ready = !buf_full;

    reg                     start;        // pulse: the units start a pass

    reg                     clr;          // pulse: the pass starts with a cleared state
    reg                     clr_next;     // the NEXT pass will have to clear the state
    reg                     buf_lastseq;  // the collected frame is the last of the sequence
    wire [NUNITS-1:0]       u_busy;       // units still busy
    wire [NUNITS-1:0]       u_ovalid;     // result of a channel ready (all together)
    wire [NUNITS*W_ACC-1:0] u_oacc;       // results of the units
    wire [NUNITS-1:0]       u_ovf;

    assign ovf = |u_ovf;

    //=================================================================
    // TREE distribution of the signals common to all the units
    // (timing 2026-10-02, second iteration).
    //   level 1: NGROUPS = NUNITS/UNITS_PER_GROUP copies (one every 8 units)
    //   level 2: one copy per unit (inside g_unit, further below)
    // Measured at 177.8 MHz with the per-unit copy only: the path
    // s4d_wload -> 32 units was still ~5 ns of wire only (slack 0.128 ns),
    // and so was clr -> clr_r of the units (0.41 ns). With the intermediate
    // level each segment covers about half of the distance.
    // Delay: +1 cycle on the coefficient writes (irrelevant) and
    // +2 cycles on start/clr, equal for all the units (lock-step
    // preserved; the control uses pass_active, so it does not restart earlier).
    // dont_touch: prevents synthesis from merging the copies.
    //=================================================================
    localparam integer UNITS_PER_GROUP = 8;
    localparam integer NGROUPS         = (NUNITS + UNITS_PER_GROUP - 1) / UNITS_PER_GROUP;

    (* dont_touch = "true" *) reg                     grp_cfg_we   [0:NGROUPS-1];
    (* dont_touch = "true" *) reg [LOG2_NU-1:0]       grp_cfg_unit [0:NGROUPS-1];
    (* dont_touch = "true" *) reg [LOG2_NB-1:0]       grp_cfg_addr [0:NGROUPS-1];
    (* dont_touch = "true" *) reg [2*W_A+2*W_B-1:0]   grp_cfg_data [0:NGROUPS-1];
    (* dont_touch = "true" *) reg                     grp_cfgd_we  [0:NGROUPS-1];
    (* dont_touch = "true" *) reg [LOG2_H-1:0]        grp_cfgd_ch  [0:NGROUPS-1];
    (* dont_touch = "true" *) reg signed [W_D-1:0]    grp_cfgd_data[0:NGROUPS-1];
    (* dont_touch = "true" *) reg [W_BSH-1:0]         grp_cfgd_bsh [0:NGROUPS-1];
    (* dont_touch = "true" *) reg                     grp_start    [0:NGROUPS-1];
    (* dont_touch = "true" *) reg                     grp_clr      [0:NGROUPS-1];

    integer grp_idx;
    // enables and control pulses: with reset (no spurious events)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (grp_idx = 0; grp_idx < NGROUPS; grp_idx = grp_idx + 1) begin
                grp_cfg_we[grp_idx]  <= 1'b0;
                grp_cfgd_we[grp_idx] <= 1'b0;
                grp_start[grp_idx]   <= 1'b0;
                grp_clr[grp_idx]     <= 1'b0;
            end
        end else begin
            for (grp_idx = 0; grp_idx < NGROUPS; grp_idx = grp_idx + 1) begin
                grp_cfg_we[grp_idx]  <= cfg_we;
                grp_cfgd_we[grp_idx] <= cfgd_we;
                grp_start[grp_idx]   <= start;
                grp_clr[grp_idx]     <= clr;
            end
        end
    end
    // addresses and data: no reset (they matter only with the enable)
    always @(posedge clk) begin
        for (grp_idx = 0; grp_idx < NGROUPS; grp_idx = grp_idx + 1) begin
            grp_cfg_unit[grp_idx]  <= cfg_unit;
            grp_cfg_addr[grp_idx]  <= cfg_addr;
            grp_cfg_data[grp_idx]  <= cfg_data;
            grp_cfgd_ch[grp_idx]   <= cfgd_ch;
            grp_cfgd_data[grp_idx] <= cfgd_data;
            grp_cfgd_bsh[grp_idx]  <= cfgd_bsh;
        end
    end

    //=================================================================
    // NUNITS biquad units
    //=================================================================
    genvar j;
    generate
        for (j = 0; j < NUNITS; j = j + 1) begin : g_unit

            localparam integer MY_GROUP = j / UNITS_PER_GROUP;   // tree group of this unit

            // the NCH samples of the frame being read that belong to unit j
            wire [NCH*W_U-1:0] u_slice =
                ubuf[(rsel*H + j*NCH)*W_U +: NCH*W_U];

            //---------------------------------------------------------
            // LOCAL copy (level 2 of the tree) of the configuration
            // bus and of start/clr, taken from the copy of the group.
            // Before, cfg_* / cfgd_* went directly to the LUTRAMs of
            // all the 32 units (fan-out up to 4032, slack 0.20-0.49 ns
            // at 150 MHz). Here the enable is already decoded.
            // Total cost: the coefficient writes arrive 2 cycles
            // later (irrelevant: the bank uses them only once a pass is started).
            //---------------------------------------------------------
            (* dont_touch = "true" *) reg                      unit_cfg_we_q;     // write coefficients in this unit
            (* dont_touch = "true" *) reg [LOG2_NB-1:0]        unit_cfg_addr_q;   // local channel*MODES + mode
            (* dont_touch = "true" *) reg [2*W_A+2*W_B-1:0]    unit_cfg_data_q;   // {b1, b0, a2, a1}
            (* dont_touch = "true" *) reg                      unit_cfgd_we_q;    // write D/shift in this unit
            (* dont_touch = "true" *) reg [LOG2_NCH-1:0]       unit_cfgd_addr_q;  // local channel
            (* dont_touch = "true" *) reg signed [W_D-1:0]     unit_cfgd_data_q;  // D
            (* dont_touch = "true" *) reg [W_BSH-1:0]          unit_cfgd_bsh_q;   // shift of the b
            (* dont_touch = "true" *) reg                      unit_start_q;      // pass start pulse
            (* dont_touch = "true" *) reg                      unit_clr_q;        // pass with cleared state

            // enables with reset: after reset no spurious write
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    unit_cfg_we_q  <= 1'b0;
                    unit_cfgd_we_q <= 1'b0;
                    unit_start_q   <= 1'b0;
                    unit_clr_q     <= 1'b0;
                end else begin
                    // coefficients: only the selected unit writes
                    unit_cfg_we_q  <= grp_cfg_we[MY_GROUP]
                                   && (grp_cfg_unit[MY_GROUP] == j[LOG2_NU-1:0]);
                    // D and shift: high bits of the channel = unit, low bits = local channel
                    unit_cfgd_we_q <= grp_cfgd_we[MY_GROUP]
                                   && (grp_cfgd_ch[MY_GROUP][LOG2_H-1:LOG2_NCH] == j[LOG2_NU-1:0]);
                    unit_start_q   <= grp_start[MY_GROUP];
                    unit_clr_q     <= grp_clr[MY_GROUP];
                end
            end

            // addresses and data: no reset (they matter only with the enable)
            always @(posedge clk) begin
                unit_cfg_addr_q  <= grp_cfg_addr[MY_GROUP];
                unit_cfg_data_q  <= grp_cfg_data[MY_GROUP];
                unit_cfgd_addr_q <= grp_cfgd_ch[MY_GROUP][LOG2_NCH-1:0];
                unit_cfgd_data_q <= grp_cfgd_data[MY_GROUP];
                unit_cfgd_bsh_q  <= grp_cfgd_bsh[MY_GROUP];
            end

            s4d_biquad_unit #(
                .NCH(NCH), .MODES(MODES),
                .LOG2_NB(LOG2_NB), .LOG2_MOD(LOG2_MOD), .LOG2_CH(LOG2_NCH),
                .W_U(W_U), .F_U(F_U), .W_A(W_A), .F_A(F_A),
                .W_B(W_B), .F_B(F_B), .W_S(W_S), .F_Y(F_Y),
                .W_D(W_D), .F_D(F_D), .W_BSH(W_BSH), .W_ACC(W_ACC)
            ) u_bq (
                .clk       (clk),
                .rst_n     (rst_n),
                // configuration from the local copy (see above)
                .cfg_we    (unit_cfg_we_q),
                .cfg_addr  (unit_cfg_addr_q),
                .cfg_data  (unit_cfg_data_q),
                .cfgd_we   (unit_cfgd_we_q),
                .cfgd_addr (unit_cfgd_addr_q),
                .cfgd_data (unit_cfgd_data_q),
                .cfgd_bsh  (unit_cfgd_bsh_q),
                .start     (unit_start_q),       // local copy (+2 cycles, equal for all)
                .clr       (unit_clr_q),
                .u_vec     (u_slice),
                .busy      (u_busy[j]),
                .out_valid (u_ovalid[j]),
                .out_acc   (u_oacc[j*W_ACC +: W_ACC]),
                .ovf       (u_ovf[j])
            );
        end
    endgenerate

    //-----------------------------------------------------------------
    // Collection of the results: at every res_ev each unit
    // delivers the result of its local channel slot_cnt
    // -> channel jj*NCH + slot_cnt
    //-----------------------------------------------------------------
    reg [H*W_ACC-1:0]  res;            // one accumulator per channel
    reg [LOG2_NCH-1:0] slot_cnt;       // local channel of the next result

    wire res_ev    = u_ovalid[0];                           // units in lock-step: unit 0 is enough
    wire pass_done = res_ev && (slot_cnt == NCH-1);         // the last channel has arrived

    integer jj;
    always @(posedge clk) begin
        if (res_ev)
            for (jj = 0; jj < NUNITS; jj = jj + 1)
                res[(jj*NCH + slot_cnt)*W_ACC +: W_ACC] <=
                    u_oacc[jj*W_ACC +: W_ACC];
    end

    reg pass_active;                   // 1 = the units are processing a frame
    reg res_pending;                   // 1 = res holds a complete frame not yet copied

    //-----------------------------------------------------------------
    // EMISSION shift register: copy of res, emitted one channel per
    // cycle, always reading the head (index 0).
    // It lets the next pass start while emitting, and removes the
    // 40-bit 128:1 multiplexer (was res[ocnt]).
    //-----------------------------------------------------------------
    reg signed [W_ACC-1:0] out_shift [0:H-1];
    reg                    stream_active;   // 1 = out_shift holds channels to emit
    reg [LOG2_H-1:0]       stream_cnt;      // channel at the head (0..H-1)

    // handshake of the output register
    wire out_free  = !out_valid || out_ready;                  // the output register can load
    wire beat      = stream_active && out_free;                // a channel is emitted
    wire last_beat = beat && (stream_cnt == H-1);              // the last channel is emitted

    // copy res -> out_shift: frame ready and emission register free
    // (empty, or it is emitting its last channel NOW)
    wire res_copy  = res_pending && (!stream_active || last_beat);

    // out_shift data: parallel load or shift (no reset)
    integer shift_idx;
    always @(posedge clk) begin
        if (res_copy) begin
            for (shift_idx = 0; shift_idx < H; shift_idx = shift_idx + 1)
                out_shift[shift_idx] <= $signed(res[shift_idx*W_ACC +: W_ACC]);
        end else if (beat) begin
            for (shift_idx = 0; shift_idx < H-1; shift_idx = shift_idx + 1)
                out_shift[shift_idx] <= out_shift[shift_idx+1];
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wsel          <= 1'b0;
            rsel          <= 1'b0;
            wcnt          <= {LOG2_H{1'b0}};
            buf_full      <= 1'b0;
            start         <= 1'b0;
            pass_active   <= 1'b0;
            res_pending   <= 1'b0;
            stream_active <= 1'b0;
            stream_cnt    <= {LOG2_H{1'b0}};
            slot_cnt      <= {LOG2_NCH{1'b0}};
            clr           <= 1'b0;
            clr_next      <= 1'b1;        // the first sequence starts from a zero state
            buf_lastseq   <= 1'b0;
        end else begin
            start <= 1'b0;
            clr   <= 1'b0;

            // ---- collection of a sample into the half being written ----
            if (in_valid && in_ready) begin
                ubuf[(wsel*H + wcnt)*W_U +: W_U] <= in_data;
                if (wcnt == H-1) begin
                    // complete frame: switch to the other half
                    wcnt        <= {LOG2_H{1'b0}};
                    buf_full    <= 1'b1;
                    wsel        <= ~wsel;
                    buf_lastseq <= in_last_seq;
                end else begin
                    wcnt <= wcnt + 1'b1;
                end
            end

            // ---- start of a pass ----
            // frame ready, previous pass finished (last result
            // arrived, pass_active = 0) and res already copied (so the
            // new pass does not overwrite results not yet emitted).
            // It NO longer waits for the end of the emission (!stream).
            // 2026-10-15: it no longer waits for !(|u_busy) either, i.e. the
            // draining of the tail of the units' pipeline (stages
            // S4, 8 cycles after the last result). That tail only writes
            // the biquad state: biquad k is written at the latest
            // 15 cycles after its read, while the next pass reads it
            // again at least ~135 cycles later, so there is no conflict.
            // This way the longer pipeline does not lengthen the frame. u_busy
            // stays connected for diagnostics only.
            if (buf_full && !pass_active && !res_pending) begin
                start       <= 1'b1;
                clr         <= clr_next;       // clear the state if a new sequence starts
                clr_next    <= buf_lastseq;    // after the last frame, the next one will clear
                rsel        <= ~wsel;          // read the half just filled
                buf_full    <= 1'b0;
                pass_active <= 1'b1;
                slot_cnt    <= {LOG2_NCH{1'b0}};
            end

            if (res_ev) slot_cnt <= slot_cnt + 1'b1;

            // ---- end of pass: res complete, waiting to be copied ----
            // (pass_done and res_copy cannot coincide: the pass
            // started with res_pending = 0)
            if (pass_done) begin
                pass_active <= 1'b0;
                res_pending <= 1'b1;
            end else if (res_copy) begin
                res_pending <= 1'b0;
            end

            // ---- emission ----
            if (res_copy) begin
                // new frame (possibly in the same cycle as the last channel of the previous one)
                stream_active <= 1'b1;
                stream_cnt    <= {LOG2_H{1'b0}};
            end else if (last_beat) begin
                stream_active <= 1'b0;
                stream_cnt    <= {LOG2_H{1'b0}};
            end else if (beat) begin
                stream_cnt <= stream_cnt + 1'b1;
            end
        end
    end

    //-----------------------------------------------------------------
    // Rounding (round-half-up), shift and saturation of the channel at
    // the head: out = sat( (acc + rnd) >>> SH_O ).
    // Timing (2026-10-15): before, addition and comparisons were in series
    // in the same cycle. Now the comparisons are done on acc, IN PARALLEL with
    // the addition, against equivalent constant thresholds:
    //   (acc + rnd) >>> SH_O >  OMAX  <=>  acc >= (OMAX+1)*2^SH_O - rnd
    //   (acc + rnd) >>> SH_O <  OMIN  <=>  acc <  OMIN*2^SH_O     - rnd
    // Exact as long as acc + rnd does not overflow W_ACC bits: |acc| <= (MODES+1)
    // * 2^(W_S-1) (D*u and MODES y values saturated to W_S bits), checked
    // below at elaboration. Same results as the previous version.
    //-----------------------------------------------------------------
    wire signed [W_ACC-1:0] acc_now = out_shift[0];

    localparam signed [W_ACC-1:0] RND_O  = {{(W_ACC-1){1'b0}}, 1'b1} <<< (SH_O-1);   // output half LSB
    localparam signed [W_ACC-1:0] THR_HI = ((OMAX + 1) <<< SH_O) - RND_O;   // from here up: OMAX
    localparam signed [W_ACC-1:0] THR_LO = (OMIN <<< SH_O) - RND_O;         // below: OMIN

    initial if (W_ACC < W_S + $clog2(MODES + 1) + 1) begin
        $display("s4d_biquad_bank: W_ACC=%0d too narrow for the threshold saturation", W_ACC);
        $finish;
    end

    wire signed [W_ACC-1:0] yrnd   = (acc_now + RND_O) >>> SH_O;   // addition (the shift is wires)
    wire                    sat_hi = (acc_now >= THR_HI);          // comparisons in parallel
    wire                    sat_lo = (acc_now <  THR_LO);

    // output register with saturation to W_OUT bits
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            out_data  <= {W_OUT{1'b0}};
            out_ch    <= {LOG2_H{1'b0}};
            out_last  <= 1'b0;
        end else if (out_free) begin
            out_valid <= beat;
            out_ch    <= stream_cnt;
            out_last  <= last_beat;

            if      (sat_hi) out_data <= OMAX[W_OUT-1:0];
            else if (sat_lo) out_data <= OMIN[W_OUT-1:0];
            else             out_data <= yrnd[W_OUT-1:0];
        end
    end

endmodule

`default_nettype wire
