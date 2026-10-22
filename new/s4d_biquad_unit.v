`timescale 1ns / 1ps
`default_nettype none
//=====================================================================
// s4d_biquad_unit  --  biquad unit (NCH channels x MODES modes)
//
//   Each S4D "mode" (pair of complex-conjugate poles) is implemented
//   as a 2nd-order IIR filter in transposed direct form II:
//
//       H(z) = (b0 + b1 z^-1) / (1 + a1 z^-1 + a2 z^-2)
//
//       y[n]  = b0*u[n]           + s1[n-1]
//       s1[n] = b1*u[n] - a1*y[n] + s2[n-1]
//       s2[n] =         - a2*y[n]
//
//   The unit holds NB = NCH*MODES biquads and processes them ONE PER CYCLE
//   (address cnt = channel*MODES + mode). For each channel the output is
//
//       out[ch] = D[ch]*u[ch] + sum_{modes} y_mode
//
//   and it is emitted (out_valid) when the last mode of the channel is processed.
//   The state {s2, s1} of each biquad is in memory: it is read when
//   that biquad's turn comes and written back at the end of the pipeline;
//   the same biquad is read again only at the next frame (NB cycles later),
//   so there is no conflict between read and write.
//
//   Shift of the b (bshift, per channel): it acts as a common exponent of
//   the coefficients b0,b1. The contribution b*u is divided by 2^bsh
//   (with rounding) before being added to the state in Q.F_Y.
//   With bsh = F_B+F_U-F_Y (= 7 with the default formats) the b are in
//   "normal" Q.F_B; larger values give more precision to small b.
//
//   Pipeline (vpipe[k] = valid data in the registers of the given stage).
//   At most ONE complex operation per cycle (timing, 2026-10-15):
//     S1  (vpipe[0])  read of b0, b1, s1, u, D, bsh
//     S2  (vpipe[1])  products b0*u, b1*u, D*u;  half LSB 2^(bsh-1)
//     S3a (vpipe[2])  b0*u, b1*u: + half LSB 2^(bsh-1);  D*u: + fixed half LSB
//     S3b (vpipe[3])  b0*u, b1*u: variable shift by bsh bits;  D*u: fixed shift
//     S3c (vpipe[4])  y = s1 + round(b0*u / 2^bsh)
//     S3d (vpipe[5])  saturation to W_S bits of y, b1*u, D*u
//                     -> accumulation of y on the channel
//                     read of a1, a2 and s2 (with the travelling address)
//     S4a (vpipe[6])  products a1*y, a2*y (24 x 32 bits: two cascaded DSPs);
//                     in parallel c = b1*u + s2
//     S4b (vpipe[7])  ... pipeline registers of the product (M/P of the DSPs)
//     S4c (vpipe[8])  ...
//     S4d (vpipe[9])  a*y + half LSB (back to Q.F_Y)
//     S4e (vpipe[10]) fixed shift (wires) and saturation of a1*y, a2*y
//     S4f (vpipe[11]) new s1 = c - a1*y;  new s2 = -a2*y
//     S4g (vpipe[12]) saturation of the new s1 -> write of the state
//
//   2026-10-20 (optimization, arithmetic identical bit for bit):
//     - y = (s1*2^bsh + b0*u + rnd) >>> bsh  becomes  s1 + ((b0*u + rnd) >>> bsh):
//       s1*2^bsh is a multiple of 2^bsh and comes out of the shift unchanged. No
//       more left shift of s1 (PW-bit barrel shifter) and no
//       limit bsh <= PW-W_S (= 24), beyond which s1 << bsh overflowed
//       (bsh used: ECG 9-17, sCIFAR 8-15, Speech Commands 9-22).
//     - b1*u + s2 computed in S4a, in parallel with the products a*y, instead
//       of carrying b1*u and s2 separately up to S4f/S4g (integer
//       additions on PW bits, without overflow: same result).
//     - a1, a2 and s2 read from the memories in S3d, where they are needed,
//       instead of in S1 and then carried for 6-12 stages. The state is split
//       into two memories (state1 = s1, state2 = s2), each with a single read
//       port; clr_r travels with the biquad (clr_s*) up to the read
//       of s2.
//     Pipeline from 15 to 13 stages.
//   State conflicts: biquad k is read (s1 in S1, s2 in S3d)
//   and written back at the end of the pipeline (S4g), i.e. AFTER both
//   reads; it is read again only in the next pass, which starts
//   at least NB cycles later (start is accepted only with run = 0).
//=====================================================================

module s4d_biquad_unit #(
    parameter integer NCH       = 2,    // channels served by the unit
    parameter integer MODES     = 32,   // biquads per channel
    parameter integer LOG2_NB   = 6,    // address bits (NCH*MODES)
    parameter integer LOG2_MOD  = 5,

    parameter integer W_U   = 16,       // input u (Q.F_U)
    parameter integer F_U   = 11,
    parameter integer W_A   = 24,       // a1, a2 (Q.F_A)
    parameter integer F_A   = 22,
    parameter integer W_B   = 16,       // b0, b1 (Q.F_B, scaled by bsh)
    parameter integer F_B   = 15,
    parameter integer W_S   = 32,       // state and output y (Q.F_Y)
    parameter integer F_Y   = 19,
    parameter integer W_D   = 16,       // D (Q.F_D)
    parameter integer F_D   = 14,

    parameter integer W_BSH = 5,        // shift of the b
    parameter integer LOG2_CH = 1,      // bits to index the channels of the unit
    parameter integer W_ACC = 40        // channel accumulator (Q.F_Y)
)(
    input  wire                       clk,
    input  wire                       rst_n,

    // ---- write of the coefficients: cfg_data = {b1, b0, a2, a1} ----
    input  wire                       cfg_we,
    input  wire [LOG2_NB-1:0]         cfg_addr,
    input  wire [2*W_A+2*W_B-1:0]     cfg_data,

    // ---- write of D and of the shift of the b of local channel cfgd_addr ----
    input  wire                       cfgd_we,
    input  wire [LOG2_CH-1:0]         cfgd_addr,
    input  wire signed [W_D-1:0]      cfgd_data,
    input  wire [W_BSH-1:0]           cfgd_bsh,

    input  wire                       start,   // pulse: process a frame
    input  wire                       clr,     // with start: state cleared for this frame
    input  wire [NCH*W_U-1:0]         u_vec,   // u samples of the NCH channels (stable for the whole frame)
    output reg                        busy,    // 1 = processing or pipeline not empty

    output reg                        out_valid, // pulse: result of a channel
    output reg  signed [W_ACC-1:0]    out_acc,   // D*u + sum of the modes, Q.F_Y

    output reg                        ovf        // saturation occurred (sticky)
);

    localparam integer NB   = NCH*MODES;          // biquads in the unit
    localparam integer PW   = W_A + W_S;          // width of the internal products / sums

    localparam integer SH_B0 = F_B + F_U - F_Y;   // "natural" shift of b*u (unused)
    localparam integer SH_A = F_A;                // a*y has F_A+F_Y fractional bits -> Q.F_Y
    localparam integer SH_D = F_D + F_U - F_Y;    // D*u has F_D+F_U fractional bits -> Q.F_Y

    // saturation limits at W_S bits, extended to PW
    localparam signed [PW-1:0] SMAX = {{(PW-W_S+1){1'b0}}, {(W_S-1){1'b1}}};
    localparam signed [PW-1:0] SMIN = {{(PW-W_S+1){1'b1}}, {(W_S-1){1'b0}}};

    //-----------------------------------------------------------------
    // Per-biquad memories (address = channel*MODES + mode), NB = 128
    // deep with NUNITS=32: forced to LUTRAM (RAM128X1D, read and
    // write at different addresses). In BRAM they would be ~48 BRAM36 over 32
    // units, with the ZCU104 budget already taken by the weight ROM (176)
    // and the seqbuf (64). Estimate: ~576 LUTs per unit, ~18k LUTs in total.
    // WARNING: valid for NB = 128. With NUNITS=8 (NB = 512) the LUTRAM
    // would cost 4 times as much: go back to "block".
    // Each memory has ONE read port: coef_b and state1 read in S1
    // (address cnt), coef_a and state2 read in S3d (address a_s3c).
    // (before: only coef_a with ram_style "block", the rest chosen by Vivado)
    //-----------------------------------------------------------------
    (* ram_style = "distributed" *)
    reg [2*W_A-1:0]       coef_a [0:NB-1];   // {a2, a1}
    (* ram_style = "distributed" *)
    reg [2*W_B-1:0]       coef_b [0:NB-1];   // {b1, b0}
    (* ram_style = "distributed" *)
    reg [W_S-1:0]         state1 [0:NB-1];   // s1
    (* ram_style = "distributed" *)
    reg [W_S-1:0]         state2 [0:NB-1];   // s2

    always @(posedge clk)
        if (cfg_we) begin
            coef_a[cfg_addr] <= cfg_data[2*W_A-1:0];
            coef_b[cfg_addr] <= cfg_data[2*W_A+2*W_B-1:2*W_A];
        end

    // per-channel memories: coefficient D and shift of the b
    reg signed [W_D-1:0] dcoef [0:NCH-1];
    reg [W_BSH-1:0]      bshift [0:NCH-1];
    always @(posedge clk)
        if (cfgd_we) begin
            dcoef[cfgd_addr]  <= cfgd_data;
            bshift[cfgd_addr] <= cfgd_bsh;
        end

    //-----------------------------------------------------------------
    // Biquad counter: at every start it scans cnt = 0..NB-1
    //-----------------------------------------------------------------
    reg [LOG2_NB-1:0] cnt;      // biquad being read (S1)
    reg               run;      // 1 = scan in progress
    localparam integer NSTAGE = 13;            // pipeline stages after the counter
    reg [NSTAGE-1:0]  vpipe;    // valid of the pipeline stages (see the header)

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cnt <= {LOG2_NB{1'b0}};
            run <= 1'b0;
        end else if (start && !run) begin
            run <= 1'b1;
            cnt <= {LOG2_NB{1'b0}};
        end else if (run) begin
            if (cnt == NB-1) begin
                run <= 1'b0;
                cnt <= {LOG2_NB{1'b0}};
            end else begin
                cnt <= cnt + 1'b1;
            end
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) vpipe <= {NSTAGE{1'b0}};
        else        vpipe <= {vpipe[NSTAGE-2:0], run};
    end

    // clr_r = 1: for the whole pass the state read is zero
    // (first frame of a new sequence). After reset it is 1.
    reg clr_r;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)                   clr_r <= 1'b1;
        else if (start && !run)       clr_r <= clr;
        else if (run && (cnt == NB-1)) clr_r <= 1'b0;
    end

    // busy while the scan runs or the pipeline holds data
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) busy <= 1'b0;
        else        busy <= run || (|vpipe);
    end

    //=================================================================
    // S1: reads (b0, b1, s1, u, D, bsh)
    //=================================================================
    reg [LOG2_NB-1:0]     a_s1;                     // biquad address along the pipeline
    reg [2*W_B-1:0]       cfb_s1;                   // {b1, b0}
    reg signed [W_S-1:0]  s1_s1;                    // s1 (0 if clr_r)
    reg signed [W_U-1:0]  u_s1;                     // channel input
    reg                   clr_s1;                   // zero state: also applies to s2, read in S3d

    // local channel of the current biquad = high bits of the address
    wire [LOG2_NB-LOG2_MOD-1:0] ch_sel = cnt[LOG2_NB-1:LOG2_MOD];
    wire signed [W_U-1:0]       u_now  = u_vec[ch_sel*W_U +: W_U];

    reg signed [W_D-1:0] d_s1;                      // D of the channel
    reg [W_BSH-1:0]      bsh_s1;                    // shift of the b of the channel

    always @(posedge clk) begin
        cfb_s1 <= coef_b[cnt];
        s1_s1  <= clr_r ? {W_S{1'b0}} : state1[cnt];
        u_s1   <= u_now;
        d_s1   <= dcoef[ch_sel];
        bsh_s1 <= bshift[ch_sel];
        clr_s1 <= clr_r;
        a_s1   <= cnt;
    end

    // field split
    wire signed [W_B-1:0] b0_s1 = $signed(cfb_s1[0             +: W_B]);
    wire signed [W_B-1:0] b1_s1 = $signed(cfb_s1[W_B           +: W_B]);

    //=================================================================
    // S2: products with the input u and half LSB 2^(bsh-1)
    //=================================================================
    reg signed [W_B+W_U-1:0] pb0_s2, pb1_s2;   // b0*u, b1*u (Q.F_B+F_U)
    reg signed [W_D+W_U-1:0] pd_s2;            // D*u (Q.F_D+F_U)
    reg signed [PW-1:0]      rnd_s2;           // 2^(bsh-1), 0 if bsh = 0
    reg signed [W_S-1:0]     s1_s2;
    reg [W_BSH-1:0]          bsh_s2;
    reg                      clr_s2;
    reg [LOG2_NB-1:0]        a_s2;

    always @(posedge clk) begin
        pb0_s2 <= b0_s1 * u_s1;
        pb1_s2 <= b1_s1 * u_s1;
        pd_s2  <= d_s1 * u_s1;
        rnd_s2 <= (bsh_s1 > 0) ? (1 <<< (bsh_s1 - 1)) : {PW{1'b0}};
        s1_s2  <= s1_s1;
        bsh_s2 <= bsh_s1;
        clr_s2 <= clr_s1;
        a_s2   <= a_s1;
    end

    // sign extension to PW bits
    wire signed [PW-1:0] pb0_ext = {{(PW-W_B-W_U){pb0_s2[W_B+W_U-1]}}, pb0_s2};
    wire signed [PW-1:0] pb1_ext = {{(PW-W_B-W_U){pb1_s2[W_B+W_U-1]}}, pb1_s2};
    wire signed [PW-1:0] pd_ext  = {{(PW-W_D-W_U){pd_s2[W_D+W_U-1]}}, pd_s2};

    //=================================================================
    // S3a: + half LSB (rounding of the following divisions)
    //   b0*u, b1*u: 2^(bsh-1)      D*u: 2^(SH_D-1)
    //=================================================================
    localparam signed [PW-1:0] RND_D = (SH_D > 0) ? (1 <<< (SH_D - 1)) : 0;

    reg signed [PW-1:0]      pb0_s3a, pb1_s3a, pd_s3a;
    reg signed [W_S-1:0]     s1_s3a;
    reg [W_BSH-1:0]          bsh_s3a;
    reg                      clr_s3a;
    reg [LOG2_NB-1:0]        a_s3a;

    always @(posedge clk) begin
        pb0_s3a <= pb0_ext + rnd_s2;
        pb1_s3a <= pb1_ext + rnd_s2;
        pd_s3a  <= pd_ext + RND_D;
        s1_s3a  <= s1_s2;
        bsh_s3a <= bsh_s2;
        clr_s3a <= clr_s2;
        a_s3a   <= a_s2;
    end

    //=================================================================
    // S3b: arithmetic shifts (round-half-up together with the half LSB
    //      already added, like the old round_shift function)
    //   b0*u, b1*u: variable by bsh bits      D*u: fixed by SH_D bits
    //=================================================================
    reg signed [PW-1:0]      pb0_t_s3b, pb1_t_s3b, du_t_s3b;
    reg signed [W_S-1:0]     s1_s3b;
    reg                      clr_s3b;
    reg [LOG2_NB-1:0]        a_s3b;

    always @(posedge clk) begin
        pb0_t_s3b <= pb0_s3a >>> bsh_s3a;
        pb1_t_s3b <= pb1_s3a >>> bsh_s3a;
        du_t_s3b  <= pd_s3a >>> SH_D;
        s1_s3b    <= s1_s3a;
        clr_s3b   <= clr_s3a;
        a_s3b     <= a_s3a;
    end

    //=================================================================
    // S3c: y = s1 + round(b0*u / 2^bsh)
    //   (= round((s1*2^bsh + b0*u) / 2^bsh) of the previous version:
    //   s1*2^bsh is a multiple of 2^bsh and comes out of the shift unchanged)
    //=================================================================
    wire signed [PW-1:0] s1_ext = {{(PW-W_S){s1_s3b[W_S-1]}}, s1_s3b};

    reg signed [PW-1:0]      y_t_s3c, pb1_t_s3c, du_t_s3c;
    reg                      clr_s3c;
    reg [LOG2_NB-1:0]        a_s3c;

    always @(posedge clk) begin
        y_t_s3c   <= pb0_t_s3b + s1_ext;
        pb1_t_s3c <= pb1_t_s3b;
        du_t_s3c  <= du_t_s3b;
        clr_s3c   <= clr_s3b;
        a_s3c     <= a_s3b;
    end

    //-----------------------------------------------------------------
    // sat_only: saturation to W_S bits, returns {overflow, value}
    //-----------------------------------------------------------------
    function [W_S:0] sat_only;
        input signed [PW-1:0] t;
        begin
            if      (t > SMAX) sat_only = {1'b1, SMAX[W_S-1:0]};
            else if (t < SMIN) sat_only = {1'b1, SMIN[W_S-1:0]};
            else               sat_only = {1'b0, t[W_S-1:0]};
        end
    endfunction

    //=================================================================
    // S3d: saturation to W_S bits of y, b1*u, D*u; read of a1, a2 and
    //      s2 of the biquad (needed from S4a onwards)
    //=================================================================
    wire [W_S:0] y_r   = sat_only(y_t_s3c);
    wire [W_S:0] pb1_r = sat_only(pb1_t_s3c);
    wire [W_S:0] du_r  = sat_only(du_t_s3c);

    reg signed [W_S-1:0] y_s3, pb1_s3, du_s3, s2_s3;
    reg [2*W_A-1:0]      cfa_s3;                // {a2, a1}
    reg                  ov_s3;                 // saturation in this stage
    reg [LOG2_NB-1:0]    a_s3d;

    always @(posedge clk) begin
        y_s3   <= $signed(y_r[W_S-1:0]);
        pb1_s3 <= $signed(pb1_r[W_S-1:0]);
        du_s3  <= $signed(du_r[W_S-1:0]);
        ov_s3  <= y_r[W_S] | pb1_r[W_S] | du_r[W_S];
        cfa_s3 <= coef_a[a_s3c];
        s2_s3  <= clr_s3c ? {W_S{1'b0}} : state2[a_s3c];
        a_s3d  <= a_s3c;
    end

    wire signed [W_A-1:0] a1_s3 = $signed(cfa_s3[0             +: W_A]);
    wire signed [W_A-1:0] a2_s3 = $signed(cfa_s3[W_A           +: W_A]);

    //=================================================================
    // S4a-S4c: products a1*y, a2*y (24 x 32 bits) with three registers:
    // synthesis moves them inside the cascaded DSPs (M and P registers).
    // In S4a, in parallel: c = b1*u + s2 (W_S+1 bits, without saturation)
    //=================================================================
    reg signed [PW-1:0]  pa1_s4a, pa2_s4a, pa1_s4b, pa2_s4b, pa1_s4c, pa2_s4c;
    reg signed [W_S:0]   c_s4a, c_s4b, c_s4c;
    reg                  ov_s4a, ov_s4b, ov_s4c;
    reg [LOG2_NB-1:0]    a_s4a, a_s4b, a_s4c;

    always @(posedge clk) begin
        pa1_s4a <= a1_s3 * y_s3;
        pa2_s4a <= a2_s3 * y_s3;
        pa1_s4b <= pa1_s4a;
        pa2_s4b <= pa2_s4a;
        pa1_s4c <= pa1_s4b;
        pa2_s4c <= pa2_s4b;
        c_s4a   <= {pb1_s3[W_S-1], pb1_s3} + {s2_s3[W_S-1], s2_s3};
        c_s4b   <= c_s4a;   c_s4c   <= c_s4b;
        ov_s4a  <= ov_s3;   ov_s4b  <= ov_s4a;   ov_s4c  <= ov_s4b;
        a_s4a   <= a_s3d;   a_s4b   <= a_s4a;    a_s4c   <= a_s4b;
    end

    //=================================================================
    // S4d: a*y + half LSB (a*y has F_A+F_Y fractional bits -> Q.F_Y)
    //=================================================================
    localparam signed [PW-1:0] RND_A = (SH_A > 0) ? (1 <<< (SH_A - 1)) : 0;

    reg signed [PW-1:0]  pa1_s4d, pa2_s4d;
    reg signed [W_S:0]   c_s4d;
    reg                  ov_s4d;
    reg [LOG2_NB-1:0]    a_s4d;

    always @(posedge clk) begin
        pa1_s4d <= pa1_s4c + RND_A;
        pa2_s4d <= pa2_s4c + RND_A;
        c_s4d   <= c_s4c;
        ov_s4d  <= ov_s4c;
        a_s4d   <= a_s4c;
    end

    //=================================================================
    // S4e: fixed shift by SH_A bits (wires) and saturation to W_S bits
    //=================================================================
    wire signed [PW-1:0] pa1_sh = pa1_s4d >>> SH_A;
    wire signed [PW-1:0] pa2_sh = pa2_s4d >>> SH_A;
    wire [W_S:0]         a1y_r  = sat_only(pa1_sh);
    wire [W_S:0]         a2y_r  = sat_only(pa2_sh);

    reg signed [W_S-1:0] a1y_s4e, a2y_s4e;
    reg signed [W_S:0]   c_s4e;
    reg                  ov_s4e;
    reg [LOG2_NB-1:0]    a_s4e;

    always @(posedge clk) begin
        a1y_s4e <= $signed(a1y_r[W_S-1:0]);
        a2y_s4e <= $signed(a2y_r[W_S-1:0]);
        c_s4e   <= c_s4d;
        ov_s4e  <= ov_s4d | a1y_r[W_S] | a2y_r[W_S];
        a_s4e   <= a_s4d;
    end

    //=================================================================
    // S4f: new unsaturated s1 = (b1*u + s2) - a1*y (on PW bits)
    //      new s2 = -a2*y (on W_S bits: -SMIN gives back SMIN as in the
    //      previous version)
    //=================================================================
    reg signed [PW-1:0]  s1_t_s4f;
    reg signed [W_S-1:0] s2_new_s4f;
    reg                  ov_s4f;
    reg [LOG2_NB-1:0]    a_s4f;

    always @(posedge clk) begin
        s1_t_s4f   <= {{(PW-W_S-1){c_s4e[W_S]}}, c_s4e}
                    - {{(PW-W_S){a1y_s4e[W_S-1]}}, a1y_s4e};
        s2_new_s4f <= -a2y_s4e;
        ov_s4f     <= ov_s4e;
        a_s4f      <= a_s4e;
    end

    //=================================================================
    // S4g: saturation of the new s1, then write of the state {s2, s1}
    //      (the memory is written from registers, in the next cycle)
    //=================================================================
    wire [W_S:0] s1_r = sat_only(s1_t_s4f);

    reg signed [W_S-1:0] s1_new_s4g, s2_new_s4g;
    reg                  ov_s4g;
    reg [LOG2_NB-1:0]    a_s4g;

    always @(posedge clk) begin
        s1_new_s4g <= $signed(s1_r[W_S-1:0]);
        s2_new_s4g <= s2_new_s4f;
        ov_s4g     <= ov_s4f | s1_r[W_S];
        a_s4g      <= a_s4f;
    end

    always @(posedge clk)
        if (vpipe[12]) begin
            state1[a_s4g] <= s1_new_s4g;
            state2[a_s4g] <= s2_new_s4g;
        end

    // saturation at any point of the computation of this biquad
    wire stage4_ovf = vpipe[12] & ov_s4g;

    //=================================================================
    // Accumulation on the channel (uses the saturated y of S3d)
    //   first mode : acc = D*u + y
    //   other modes: acc = acc + y
    //   last mode  : the result goes out on out_acc
    //=================================================================
    wire [LOG2_MOD-1:0] mode_s3  = a_s3d[LOG2_MOD-1:0];  // mode = low bits of the address
    wire                first_s3 = (mode_s3 == 0);
    wire                last_s3  = (mode_s3 == MODES-1);

    reg  signed [W_ACC-1:0] acc;
    wire signed [W_ACC-1:0] y_ext   = {{(W_ACC-W_S){y_s3[W_S-1]}}, y_s3};
    wire signed [W_ACC-1:0] du_ext  = {{(W_ACC-W_S){du_s3[W_S-1]}}, du_s3};

    wire signed [W_ACC-1:0] acc_nxt = (first_s3 ? du_ext : acc) + y_ext;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc       <= {W_ACC{1'b0}};
            out_valid <= 1'b0;
            out_acc   <= {W_ACC{1'b0}};
        end else begin
            out_valid <= 1'b0;
            if (vpipe[5]) begin
                acc <= acc_nxt;
                if (last_s3) begin
                    out_valid <= 1'b1;
                    out_acc   <= acc_nxt;
                end
            end
        end
    end

    // overflow flag (stays at 1 until reset)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)          ovf <= 1'b0;
        else if (stage4_ovf) ovf <= 1'b1;
    end

endmodule

`default_nettype wire
