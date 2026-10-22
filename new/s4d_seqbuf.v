`timescale 1ns / 1ps
`default_nettype none
//=====================================================================
// s4d_seqbuf  --  sequence buffer
//
//   Dual-port memory of NWORDS words x W bits, in Block RAM.
//   It holds the whole sequence (L time steps x H channels) between one
//   pass of the layer and the next.
//     - write port: 1 word per cycle
//     - read port : synchronous, data available RD_LAT cycles after rd_en
//       RD_LAT = 1: direct BRAM output
//       RD_LAT = 2: + output register (absorbed into the DOB_REG of the BRAM)
//       RD_LAT = 3: + one register AFTER the multiplexer between the BRAM chains
//                   (recommended at 250+ MHz, see CASCADE below)
//
//   CASCADE (timing 2026-10-05): without constraints Vivado built the memory
//   as chains of 8 cascaded BRAMs; the read crossed 7 BRAMs + one LUT6 in a
//   single cycle (2.56 ns of logic, measured slack -0.153 ns at 266.7 MHz,
//   the only failing path). Now:
//     cascade_height = 2 -> chains of at most 2 BRAMs;
//     RD_LAT = 3         -> BRAM output, multiplexer between the chains and
//                           final register in separate cycles.
//   If the same address is read and written in the same cycle, the read
//   returns the OLD value (read-first).
//
//   err_ovw check: during a pass the write happens "in place" and must
//   always stay BEHIND the read (only what has already been read is
//   overwritten). If wr_addr >= rd_addr a word not yet read would be
//   overwritten -> sticky flag.
//=====================================================================

module s4d_seqbuf #(
    parameter integer W      = 16,       // word width
    parameter integer NWORDS = 131072,   // words (L*H)
    parameter integer AW     = 17,       // address bits
    parameter integer RD_LAT = 3         // read latency: 1 = BRAM only, 2 = + output register,
                                         // 3 = + register after the multiplexer between the chains
)(
    input  wire                clk,
    input  wire                rst_n,

    // write port
    input  wire                wr_en,
    input  wire [AW-1:0]       wr_addr,
    input  wire signed [W-1:0] wr_data,

    // read port (latency RD_LAT cycles)
    input  wire                rd_en,
    input  wire [AW-1:0]       rd_addr,
    output wire signed [W-1:0] rd_data,   // was output reg: valid RD_LAT cycles after rd_en

    // overwrite check
    input  wire                chk_en,   // 1 = check active (during the passes)
    output reg                 err_ovw   // sticky: write >= read
);

    (* ram_style = "block", cascade_height = 2 *)   // chains of at most 2 BRAMs
    reg [W-1:0] mem [0:NWORDS-1];
    reg [W-1:0] rd_mem_q;                 // BRAM output (latency 1)

    // synchronous write and read in the same memory
    always @(posedge clk) begin
        if (wr_en) mem[wr_addr] <= wr_data;
        if (rd_en) rd_mem_q <= mem[rd_addr];
    end

    generate
    if (RD_LAT == 3) begin : g_out_reg2
        reg [W-1:0] rd_out_q;             // BRAM output register (DOB_REG)
        reg [W-1:0] rd_out_q2;            // register after the multiplexer between the chains
        always @(posedge clk) begin
            rd_out_q  <= rd_mem_q;
            rd_out_q2 <= rd_out_q;
        end
        assign rd_data = $signed(rd_out_q2);
    end else if (RD_LAT == 2) begin : g_out_reg
        reg [W-1:0] rd_out_q;             // output register, always enabled
        always @(posedge clk) rd_out_q <= rd_mem_q;
        assign rd_data = $signed(rd_out_q);
    end else begin : g_no_out_reg
        assign rd_data = $signed(rd_mem_q);
    end
    endgenerate

    // overwrite flag (stays at 1 until reset)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)                                    err_ovw <= 1'b0;
        else if (chk_en && wr_en && (wr_addr >= rd_addr)) err_ovw <= 1'b1;
    end

endmodule

`default_nettype wire
