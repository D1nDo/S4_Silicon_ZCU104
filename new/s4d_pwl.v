`timescale 1ns / 1ps
`default_nettype none
//=====================================================================
// s4d_pwl  --  piecewise linear function
//
//   Used for the GELU (in s4d_layer) and the sigmoid (in s4d_mixing_glu).
//
//   Covered interval: [-2^XH, +2^XH) = [-8, 8) with XH = 3, split into
//   NSEG = 256 equal segments (width 1/16).
//     xs   = x + 2^XH                  (x moved into [0, 2^(XH+1)))
//     idx  = high bits of xs           -> which segment
//     frac = FRACW low bits of xs      -> position inside the segment
//     y    = base[idx] + slope[idx] * frac / 2^FRACW   (rounded)
//
//   LUT (LUT_FILE): NSEG words of 2*W_OUT bits = {slope, base}, both in
//   the output format Q.F_OUT.
//
//   Outside the interval:
//     x <  -2^XH : y = Y_LO_Q
//     x >= +2^XH : y = x        if MODE_HI = 1 (GELU: gelu(x) ~ x)
//                  y = Y_HI_Q   if MODE_HI = 0 (sigmoid: 1.0)
//
//   Pipeline with LAT = 5 stages, at most ONE complex operation per
//   cycle (timing, 2026-10-15; before: 3 stages with multiplication, two
//   additions, comparisons and mux in the last one):
//     1 x + 2^XH and comparisons with the limits (in parallel) -> idx, frac
//     2 LUT read
//     3 product slope * frac                         (M register of the DSP)
//     4 sum product + (base << FRACW) + half LSB     (P register of the DSP)
//     5 shift (wires only), saturation, out-of-range mux -> output
//   Stage 4 uses the exact identity
//     base + ((prod + rnd) >>> FRACW) = (prod + rnd + base*2^FRACW) >>> FRACW
//   and the term (base << FRACW) + rnd is a concatenation (rnd < 2^FRACW),
//   so there is a single adder. No overflow: |prod + rnd| < 2^(W_OUT-1+FRACW)
//   and the sum with base*2^FRACW fits in PW3 = W_OUT+FRACW+1 bits, as before.
//
//   in_tag / out_tag: TAG_W bits that cross the pipeline together with the
//   data, with the same enable (e.g. "last channel of the frame", or the
//   value a of the GLU). The user of the module does not need to know its latency.
//=====================================================================

module s4d_pwl #(
    parameter integer W_IN      = 16,   // input (Q.F_IN)
    parameter integer F_IN      = 11,
    parameter integer W_OUT     = 16,   // output (Q.F_OUT)
    parameter integer F_OUT     = 11,
    parameter integer NSEG      = 256,  // segments
    parameter integer LOG2_NSEG = 8,
    parameter integer XH        = 3,    // half-width of the interval = 2^XH
    parameter integer MODE_HI   = 1,    // 1: above the interval y = x; 0: y = Y_HI_Q
    parameter integer Y_LO_Q    = 0,    // value below the interval (Q.F_OUT)
    parameter integer Y_HI_Q    = 0,    // value above the interval if MODE_HI = 0
    parameter integer TAG_W     = 1,    // bits of the tag travelling with the data
    parameter         LUT_FILE  = "gelu_lut.mem"
)(
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire                    in_valid,
    output wire                    in_ready,
    input  wire signed [W_IN-1:0]  in_data,
    input  wire [TAG_W-1:0]        in_tag,     // travels with in_data
    input  wire                    out_ready,
    output reg                     out_valid,
    output reg  signed [W_OUT-1:0] out_data,
    output reg  [TAG_W-1:0]        out_tag     // travels with out_data
);

    // Globally STALLABLE pipeline: out_ready low freezes all the stages
    // together. Same scheme as s4d_affine.
    // NOTE: in_ready is combinational from out_ready -> pass-through path.
    wire en = out_ready;
    assign in_ready = out_ready;

    localparam integer FRACW  = F_IN + XH + 1 - LOG2_NSEG;   // bits of the position in the segment
    localparam integer XW     = W_IN + 1;                    // one extra bit for x + offset
    // offset = 2^XH in Q.F_IN
    localparam signed [XW-1:0] OFFS  = {{(XW-XH-1-F_IN){1'b0}}, 1'b1, {(XH+F_IN){1'b0}}};
    localparam signed [XW-1:0] X_LO  = -OFFS;                // lower limit -2^XH
    localparam signed [XW-1:0] X_HI  =  OFFS;                // upper limit +2^XH

    localparam integer PW3 = W_OUT + FRACW + 1;              // width of product and sum

    initial if (FRACW < 1) begin
        $display("s4d_pwl: FRACW=%0d, at least 1 bit of position in the segment is needed", FRACW);
        $finish;
    end

    // segment LUT: {slope, base}
    reg [2*W_OUT-1:0] lut [0:NSEG-1];
    initial $readmemh(LUT_FILE, lut);

    wire signed [XW-1:0] x   = {in_data[W_IN-1], in_data};   // x extended by one bit
    wire signed [XW-1:0] xs  = x + OFFS;                     // x moved into [0, 2^(XH+1))

    // valid of the 4 internal stages (stage 5 is out_valid)
    reg v1, v2, v3, v4;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v1 <= 1'b0; v2 <= 1'b0; v3 <= 1'b0; v4 <= 1'b0;
        end else if (en) begin
            v1 <= in_valid; v2 <= v1; v3 <= v2; v4 <= v3;
        end
    end

    //=================================================================
    // Stage 1: offset addition and comparisons (in parallel)
    //=================================================================
    reg                      below1, above1;       // below / above the interval
    reg [LOG2_NSEG-1:0]      idx1;                 // segment
    reg [FRACW-1:0]          frac1;                // position in the segment
    reg signed [W_IN-1:0]    x1;                   // original x (needed if MODE_HI = 1)
    reg [TAG_W-1:0]          tag1;

    always @(posedge clk) if (en) begin
        below1 <= (x <  X_LO);
        above1 <= (x >= X_HI);
        idx1   <= xs[FRACW +: LOG2_NSEG];
        frac1  <= xs[FRACW-1:0];
        x1     <= in_data;
        tag1   <= in_tag;
    end

    //=================================================================
    // Stage 2: LUT read
    //=================================================================
    reg [2*W_OUT-1:0]     seg2;                    // {slope, base} of the segment
    reg [FRACW-1:0]       frac2;
    reg                   below2, above2;
    reg signed [W_IN-1:0] x2;
    reg [TAG_W-1:0]       tag2;

    always @(posedge clk) if (en) begin
        seg2   <= lut[idx1];
        frac2  <= frac1;
        below2 <= below1;
        above2 <= above1;
        x2     <= x1;
        tag2   <= tag1;
    end

    wire signed [W_OUT-1:0] base2  = $signed(seg2[0     +: W_OUT]);
    wire signed [W_OUT-1:0] slope2 = $signed(seg2[W_OUT +: W_OUT]);

    //=================================================================
    // Stage 3: product slope * frac (frac is unsigned)
    //   in parallel: constant term (base << FRACW) + half LSB, which is
    //   a concatenation {base, 1, 0...0} (no adder)
    //=================================================================
    reg signed [PW3-1:0]  prod3;                   // slope * frac
    reg signed [PW3-1:0]  addend3;                 // base * 2^FRACW + 2^(FRACW-1)
    reg                   below3, above3;
    reg signed [W_IN-1:0] x3;
    reg [TAG_W-1:0]       tag3;

    always @(posedge clk) if (en) begin
        prod3   <= slope2 * $signed({1'b0, frac2});
        addend3 <= $signed({base2[W_OUT-1], base2, 1'b1, {(FRACW-1){1'b0}}});
        below3  <= below2;
        above3  <= above2;
        x3      <= x2;
        tag3    <= tag2;
    end

    //=================================================================
    // Stage 4: a single adder
    //=================================================================
    reg signed [PW3-1:0]  sum4;                    // prod + rnd + base * 2^FRACW
    reg                   below4, above4;
    reg signed [W_IN-1:0] x4;
    reg [TAG_W-1:0]       tag4;

    always @(posedge clk) if (en) begin
        sum4   <= prod3 + addend3;
        below4 <= below3;
        above4 <= above3;
        x4     <= x3;
        tag4   <= tag3;
    end

    //=================================================================
    // Stage 5: shift (wires), saturation, out-of-range values
    //=================================================================
    wire signed [PW3-1:0] interp = sum4 >>> FRACW;

    localparam signed [W_OUT-1:0] YLO = Y_LO_Q[W_OUT-1:0];
    localparam signed [W_OUT-1:0] YHI = Y_HI_Q[W_OUT-1:0];
    localparam signed [PW3-1:0] OMAX = {{(PW3-W_OUT+1){1'b0}}, {(W_OUT-1){1'b1}}};
    localparam signed [PW3-1:0] OMIN = {{(PW3-W_OUT+1){1'b1}}, {(W_OUT-1){1'b0}}};

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            out_data  <= {W_OUT{1'b0}};
            out_tag   <= {TAG_W{1'b0}};
        end else if (en) begin
            out_valid <= v4;
            out_tag   <= tag4;

            if (below4)
                out_data <= YLO;
            else if (above4)
                // MODE_HI = 1 assumes the same format at input and output
                out_data <= (MODE_HI != 0) ? x4[W_OUT-1:0] : YHI;
            else if (interp > OMAX)
                out_data <= OMAX[W_OUT-1:0];
            else if (interp < OMIN)
                out_data <= OMIN[W_OUT-1:0];
            else
                out_data <= interp[W_OUT-1:0];
        end
    end

endmodule

`default_nettype wire
