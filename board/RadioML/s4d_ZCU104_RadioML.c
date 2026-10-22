/* ============================================================================
 * s4d_ZCU104_RadioML.c -- board test of the S4D RadioML 2018.01A bitstream
 * on ZCU104, from Linux userspace through /dev/mem.
 *
 * For each RadioML frame it:
 *   1. copies the frame (4096 bytes) into the input buffer in DDR;
 *   2. arms the S2MM to receive the 24 logits (96 bytes);
 *   3. starts the MM2S of the frame towards the encoder;
 *   4. waits for the end of the S2MM and reads the 24 int32 logits.
 * The weights do NOT go through the DMA: they are in the on-chip ROM
 * (s4d_weightrom_axis).
 *
 * At the end it prints:
 *   - bit-exact comparison of the logits with the fixed-point golden (all frames);
 *   - accuracy (argmax of the logits) averaged over all SNRs (the standard
 *     RadioML metric), at SNR >= 0 dB, at SNR >= 10 dB and for each SNR;
 *   - inference time per frame (min / mean / max), throughput and cycles,
 *     using the ACTUAL pl_clk0 frequency read from the PS registers.
 *
 * Map from the S4D_RadioML block design (same as S4D_PTB_XL):
 *   axi_dma_0 @ 0xA000_0000 (S_AXI_LITE via HPM0_FPD), simple mode, no SG
 *     MM2S 64 bit -> s4d_encoder_axis   (I/Q frame)
 *     S2MM 32 bit <- s4d_decoder_axis   (24 logits, tlast on the 24th)
 *   MM2S and S2MM towards HPC0_DDR_LOW (0x0000_0000 - 0x7FFF_FFFF).
 *
 * Frame format: L = 1024 time steps x 2 inputs (I, Q), int16 Q5.11
 * little-endian, order [t][I,Q] (I = c0 in the low bits) -> 4096 bytes =
 * 512 beats of 64 bit (2 time steps per beat, no padding).
 * Code = sat16(round(HDF5 sample * 1.3024024307265825 * 2048)).
 *
 * Files expected in the working directory (board/make_board_files.py RadioML;
 * 9984 test frames, 16 per (class, SNR) pair):
 *   samples_RadioML_9984.bin        9984 x 4096 bytes
 *   labels_RadioML_9984.bin         9984 bytes, true class 0..23 (CLASS_NAME order)
 *   snr_RadioML_9984.bin            9984 bytes, SNR in dB (int8, -20..30)
 *   logits_golden_RadioML_9984.bin  9984 x 24 int32 little-endian [frame][class]
 * Writes:
 *   logits_board_ZCU104_RadioML.bin N x 24 int32, the logits read from the board
 *
 * Fixed-point reference on the same 9984 frames:
 *   accuracy over all SNRs 0.5995, SNR >= 0: 0.8654, SNR >= 10: 0.9306.
 *
 * DDR MEMORY FOR THE DMA: as in the ECG test, a region that Linux does NOT
 * use, below 2 GB (HPC0_DDR_LOW); default DDR_RESERVED_BASE_PHYS below,
 * it can be changed from the command line.
 *
 * Build (on the board):  gcc -O2 -o s4d_ZCU104_RadioML s4d_ZCU104_RadioML.c
 * Usage (root):  ./s4d_ZCU104_RadioML [N_FRAMES] [BASE_DDR_HEX]
 *                default N_FRAMES = 9984, BASE_DDR_HEX = 0x70000000
 * ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <errno.h>
#include <time.h>

/* ------------------------------- CONFIG --------------------------------- */

#define BUILD_TAG  "S4D RadioML 2018.01A -- ZCU104, logits + golden + inference time"

/* AXI DMA registers (address from the block design) */
#define DMA_REGS_BASE_PHYS      0xA0000000UL
#define DMA_REGS_SPAN           0x10000UL         /* 64K as in the Address Editor */

/* model */
#define L_SEQ                   1024              /* time steps per frame */
#define N_INPUTS                2                 /* I and Q */
#define INPUT_RECORD_BYTES      (L_SEQ * N_INPUTS * 2)   /* 4096 bytes = 512 beats of 64 bit */
#define K_CLASSES               24                /* RadioML 2018.01A modulations */
#define OUTPUT_LOGITS_BYTES     (K_CLASSES * 4)          /* 24 int32 logits = 96 bytes */

#define N_RECORDS_MAX           9984              /* test frames (24 classes x 26 SNRs x 16) */
#define N_GOLDEN_RECORDS        9984              /* golden for every frame */

#define SNR_MIN_DB              (-20)             /* SNR from -20 to +30 dB, step 2 */
#define SNR_STEP_DB             2
#define N_SNR_LEVELS            26

/* PS clock registers (CRL_APB): the actual pl_clk0 frequency is read here,
 * to convert the measured times into cycles.
 * Same fields as in pl0_clk_set.c (see its header). */
#define CRL_APB_BASE_PHYS       0xFF5E0000UL
#define CRL_APB_SPAN            0x1000UL
#define REG_IOPLL_CTRL          0x020
#define REG_IOPLL_FRAC_CFG      0x028
#define REG_RPLL_CTRL           0x030
#define REG_RPLL_FRAC_CFG       0x038
#define REG_PL0_REF_CTRL        0x0C0
#define PSS_REF_CLK_MHZ         33.333          /* PS reference oscillator on the ZCU104 */

/* DDR region reserved for the DMA buffers */
#define DDR_RESERVED_BASE_PHYS  0x70000000UL
#define DDR_RESERVED_SPAN       0x00100000UL      /* 1 MB: ~33 KB are enough */
/* position of the two buffers inside the reserved region (4 KB aligned) */
#define DDR_INPUT_RECORD_OFFSET 0x00000000UL      /* frame read by the MM2S */
#define DDR_OUTPUT_LOGITS_OFFSET 0x00008000UL     /* 24 logits written by the S2MM (after 32 KB) */

#define SAMPLES_BIN_PATH        "samples_RadioML_9984.bin"
#define LABELS_BIN_PATH         "labels_RadioML_9984.bin"
#define SNR_BIN_PATH            "snr_RadioML_9984.bin"
#define GOLDEN_BIN_PATH         "logits_golden_RadioML_9984.bin"
#define BOARD_LOGITS_BIN_PATH   "logits_board_ZCU104_RadioML.bin"

#define DMA_TIMEOUT_SEC         10.0              /* a frame takes ~ms: 10 s = stuck */
#define N_MISMATCH_PRINT_MAX    10                /* how many frames differing from the golden to print */

static const char *CLASS_NAME[K_CLASSES] = {
    "OOK", "4ASK", "8ASK", "BPSK", "QPSK", "8PSK", "16PSK", "32PSK",
    "16APSK", "32APSK", "64APSK", "128APSK", "16QAM", "32QAM", "64QAM",
    "128QAM", "256QAM", "AM-SSB-WC", "AM-SSB-SC", "AM-DSB-WC",
    "AM-DSB-SC", "FM", "GMSK", "OQPSK" };

/* --------------------------- AXI DMA registers --------------------------- */

#define REG_MM2S_DMACR   0x00
#define REG_MM2S_DMASR   0x04
#define REG_MM2S_SA      0x18
#define REG_MM2S_SA_MSB  0x1C
#define REG_MM2S_LENGTH  0x28

#define REG_S2MM_DMACR   0x30
#define REG_S2MM_DMASR   0x34
#define REG_S2MM_DA      0x48
#define REG_S2MM_DA_MSB  0x4C
#define REG_S2MM_LENGTH  0x58

#define DMACR_RS         (1u << 0)     /* run/stop */
#define DMACR_RESET      (1u << 2)     /* soft reset */
#define DMASR_HALTED     (1u << 0)
#define DMASR_IDLE       (1u << 1)
#define DMASR_ERR_MASK   0x770u        /* DMAIntErr, DMASlvErr, DMADecErr, SG* */
#define DMASR_IOC_IRQ    (1u << 12)
#define DMASR_IRQ_MASK   0x7000u       /* interrupt bits, cleared by writing 1 */

/* ------------------------------- utilities ------------------------------ */

/* monotonic time in seconds (ns resolution) */
static double now_sec(void)
{
    struct timespec time_now;
    clock_gettime(CLOCK_MONOTONIC, &time_now);
    return (double)time_now.tv_sec + (double)time_now.tv_nsec / 1e9;
}

/* map a physical region (registers or reserved DDR) into the process */
static void *map_physical(int devmem_fd, uint64_t region_phys, size_t region_span,
                          const char *region_name)
{
    void *region_virt = mmap(NULL, region_span, PROT_READ | PROT_WRITE, MAP_SHARED,
                             devmem_fd, (off_t)region_phys);
    if (region_virt == MAP_FAILED) {
        fprintf(stderr, "mmap %s @0x%lx (%zu bytes): %s\n", region_name,
                (unsigned long)region_phys, region_span, strerror(errno));
        exit(1);
    }
    return region_virt;
}

static inline void reg_write(volatile uint32_t *dma_regs_virt, unsigned reg_offset, uint32_t value)
{
    dma_regs_virt[reg_offset / 4] = value;
}

static inline uint32_t reg_read(volatile uint32_t *dma_regs_virt, unsigned reg_offset)
{
    return dma_regs_virt[reg_offset / 4];
}

/* soft reset of a channel (the reset acts on both channels of the DMA) */
static void dma_reset(volatile uint32_t *dma_regs_virt, unsigned dmacr_offset)
{
    reg_write(dma_regs_virt, dmacr_offset, DMACR_RESET);
    int poll_guard = 1000000;
    while ((reg_read(dma_regs_virt, dmacr_offset) & DMACR_RESET) && poll_guard--) {}
    if (poll_guard <= 0) {
        fprintf(stderr, "dma_reset: timeout\n");
        exit(1);
    }
}

/* wait for the end of a transfer (IDLE or IOC); exit on error or timeout */
static void dma_wait_done(volatile uint32_t *dma_regs_virt, unsigned dmasr_offset,
                          const char *channel_name)
{
    double wait_start_sec = now_sec();
    for (;;) {
        uint32_t dma_status = reg_read(dma_regs_virt, dmasr_offset);
        if (dma_status & DMASR_ERR_MASK) {
            fprintf(stderr, "%s: DMA error, DMASR=0x%08x\n", channel_name, dma_status);
            exit(1);
        }
        if (dma_status & (DMASR_IDLE | DMASR_IOC_IRQ)) {
            /* clear the interrupt bits for the next transfer */
            reg_write(dma_regs_virt, dmasr_offset, dma_status & DMASR_IRQ_MASK);
            return;
        }
        if (now_sec() - wait_start_sec > DMA_TIMEOUT_SEC) {
            fprintf(stderr, "%s: timeout after %.1f s, DMASR=0x%08x "
                    "(core stuck? bitstream loaded? PL0 running?)\n",
                    channel_name, DMA_TIMEOUT_SEC, dma_status);
            exit(1);
        }
    }
}

/* start the MM2S: DDR -> encoder. Writing LENGTH starts the transfer */
static void dma_mm2s_start(volatile uint32_t *dma_regs_virt, uint64_t source_phys, uint32_t length_bytes)
{
    reg_write(dma_regs_virt, REG_MM2S_DMACR, DMACR_RS);
    reg_write(dma_regs_virt, REG_MM2S_SA, (uint32_t)source_phys);
    reg_write(dma_regs_virt, REG_MM2S_SA_MSB, (uint32_t)(source_phys >> 32));   /* ignored if c_addr_width = 32 */
    reg_write(dma_regs_virt, REG_MM2S_LENGTH, length_bytes);
}

/* arm the S2MM: decoder -> DDR. It must be armed BEFORE sending the input:
 * the decoder has no backpressure towards the core */
static void dma_s2mm_start(volatile uint32_t *dma_regs_virt, uint64_t destination_phys, uint32_t length_bytes)
{
    reg_write(dma_regs_virt, REG_S2MM_DMACR, DMACR_RS);
    reg_write(dma_regs_virt, REG_S2MM_DA, (uint32_t)destination_phys);
    reg_write(dma_regs_virt, REG_S2MM_DA_MSB, (uint32_t)(destination_phys >> 32));
    reg_write(dma_regs_virt, REG_S2MM_LENGTH, length_bytes);
}

/* output frequency of an LPD PLL from its CTRL and FRAC_CFG registers */
static double pll_output_mhz(volatile uint32_t *crl_apb_regs_virt, unsigned ctrl_offset, unsigned frac_offset)
{
    uint32_t pll_ctrl     = crl_apb_regs_virt[ctrl_offset / 4];
    uint32_t pll_frac_cfg = crl_apb_regs_virt[frac_offset / 4];
    unsigned feedback_div = (pll_ctrl >> 8) & 0x7Fu;        /* FBDIV: integer multiplier */
    unsigned output_div2  = (pll_ctrl >> 16) & 0x1u;        /* DIV2: output divided by 2 */
    unsigned pll_bypass   = (pll_ctrl >> 3) & 0x1u;         /* BYPASS: output = reference */
    double   frac_part    = ((pll_frac_cfg >> 31) & 1u) ? (pll_frac_cfg & 0xFFFFu) / 65536.0 : 0.0;
    if (pll_bypass) return PSS_REF_CLK_MHZ;
    return PSS_REF_CLK_MHZ * (feedback_div + frac_part) / (output_div2 ? 2.0 : 1.0);
}

/* actual frequency of pl_clk0 in MHz (source IOPLL or RPLL / DIVISOR0 / DIVISOR1).
 * Returns -1 if the source is the DPLL (not decoded here). */
static double read_pl0_clock_mhz(int devmem_fd, const char **source_name)
{
    volatile uint32_t *crl_apb_regs_virt =
        (volatile uint32_t *)map_physical(devmem_fd, CRL_APB_BASE_PHYS, CRL_APB_SPAN, "CRL_APB");
    uint32_t pl0_ref_ctrl  = crl_apb_regs_virt[REG_PL0_REF_CTRL / 4];
    unsigned source_select = pl0_ref_ctrl & 0x7u;           /* 0 IOPLL, 2 RPLL, 3 DPLL_TO_LPD */
    unsigned divisor0      = (pl0_ref_ctrl >> 8) & 0x3Fu;
    unsigned divisor1      = (pl0_ref_ctrl >> 16) & 0x3Fu;
    if (divisor0 == 0) divisor0 = 1;                        /* 0 behaves as 1 */
    if (divisor1 == 0) divisor1 = 1;

    double source_mhz = -1.0;
    if (source_select == 0) {
        *source_name = "IOPLL";
        source_mhz = pll_output_mhz(crl_apb_regs_virt, REG_IOPLL_CTRL, REG_IOPLL_FRAC_CFG);
    } else if (source_select == 2) {
        *source_name = "RPLL";
        source_mhz = pll_output_mhz(crl_apb_regs_virt, REG_RPLL_CTRL, REG_RPLL_FRAC_CFG);
    } else {
        *source_name = "DPLL_TO_LPD (not decoded)";
    }
    munmap((void *)crl_apb_regs_virt, CRL_APB_SPAN);
    return (source_mhz < 0) ? -1.0 : source_mhz / divisor0 / divisor1;
}

/* read exactly n bytes from a file, otherwise exit */
static void read_exact(const char *file_path, FILE *file_handle, void *destination, size_t n_bytes)
{
    if (fread(destination, 1, n_bytes, file_handle) != n_bytes) {
        fprintf(stderr, "%s too short (another %zu bytes expected)\n", file_path, n_bytes);
        exit(1);
    }
}

/* copy to the reserved DDR with aligned 64-bit words.
 * The "no-map" DDR mapped through /dev/mem is non-cacheable memory (Device
 * type on arm64): memcpy() could issue unaligned accesses and raise SIGBUS. */
static void copy_to_ddr64(volatile uint64_t *ddr_destination_virt, const uint64_t *source_buffer, size_t n_bytes)
{
    for (size_t word_index = 0; word_index < n_bytes / 8; word_index++)
        ddr_destination_virt[word_index] = source_buffer[word_index];
}

/* -------------------------------- metrics ------------------------------- */

static int32_t board_logits[N_RECORDS_MAX][K_CLASSES];     /* logits read from the board */
static int32_t golden_logits[N_GOLDEN_RECORDS][K_CLASSES]; /* fixed-point golden logits */
static uint8_t true_class[N_RECORDS_MAX];                  /* true class 0..23 */
static int8_t  frame_snr_db[N_RECORDS_MAX];                /* SNR of the frame in dB */
static double  inference_time_sec[N_RECORDS_MAX];          /* MM2S started -> S2MM done */
static double  input_transfer_sec[N_RECORDS_MAX];          /* MM2S started -> MM2S done */

/* predicted class = index of the largest logit; on ties the lowest index
 * wins (strictly greater), as argmax of the golden */
static int predicted_class(const int32_t *frame_logits)
{
    int best_class = 0;
    for (int class_index = 1; class_index < K_CLASSES; class_index++)
        if (frame_logits[class_index] > frame_logits[best_class]) best_class = class_index;
    return best_class;
}

/* statistics of a vector of times */
static void time_stats(const double *time_values, int n_values,
                       double *min_out, double *mean_out, double *max_out)
{
    double min_value = time_values[0], max_value = time_values[0], sum_values = 0.0;
    for (int value_index = 0; value_index < n_values; value_index++) {
        if (time_values[value_index] < min_value) min_value = time_values[value_index];
        if (time_values[value_index] > max_value) max_value = time_values[value_index];
        sum_values += time_values[value_index];
    }
    *min_out = min_value; *mean_out = sum_values / n_values; *max_out = max_value;
}

/* print the logits of a frame on one line */
static void print_logits(const char *row_label, int record_index, const int32_t *frame_logits)
{
    printf("  frame %d  %s: [", record_index, row_label);
    for (int class_index = 0; class_index < K_CLASSES; class_index++)
        printf("%s%d", class_index ? ", " : "", frame_logits[class_index]);
    printf("]\n");
}

/* --------------------------------- main --------------------------------- */

int main(int argc, char **argv)
{
    /* ---- arguments ---- */
    int n_records = N_RECORDS_MAX;
    uint64_t ddr_reserved_base_phys = DDR_RESERVED_BASE_PHYS;
    if (argc > 1) {
        n_records = atoi(argv[1]);
        if (n_records < 1 || n_records > N_RECORDS_MAX) {
            fprintf(stderr, "N_FRAMES must be between 1 and %d\n", N_RECORDS_MAX);
            return 1;
        }
    }
    if (argc > 2) ddr_reserved_base_phys = strtoull(argv[2], NULL, 16);
    if (ddr_reserved_base_phys + DDR_RESERVED_SPAN > 0x80000000ULL) {
        fprintf(stderr, "the DDR region must be below 2 GB (HPC0_DDR_LOW in the BD)\n");
        return 1;
    }

    /* PHYSICAL addresses of the buffers, the ones seen by the DMA */
    const uint64_t ddr_input_record_phys  = ddr_reserved_base_phys + DDR_INPUT_RECORD_OFFSET;
    const uint64_t ddr_output_logits_phys = ddr_reserved_base_phys + DDR_OUTPUT_LOGITS_OFFSET;

    printf("=== %s ===\n", BUILD_TAG);
    printf("DMA @0x%08lx | frame in DDR @0x%08lx (%d B) | logits in DDR @0x%08lx (%d B)\n",
           DMA_REGS_BASE_PHYS, (unsigned long)ddr_input_record_phys, INPUT_RECORD_BYTES,
           (unsigned long)ddr_output_logits_phys, OUTPUT_LOGITS_BYTES);

    /* ---- mappings ---- */
    int devmem_fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (devmem_fd < 0) {
        fprintf(stderr, "open /dev/mem (root required): %s\n", strerror(errno));
        return 1;
    }

    /* actual pl_clk0 frequency, read BEFORE the test */
    const char *pl0_source_name = "";
    double pl0_clock_mhz = read_pl0_clock_mhz(devmem_fd, &pl0_source_name);
    if (pl0_clock_mhz > 0)
        printf("pl_clk0: %.4f MHz (source %s, read from PL0_REF_CTRL)\n\n", pl0_clock_mhz, pl0_source_name);
    else
        printf("pl_clk0: source %s, frequency not available (cycles not computed)\n\n", pl0_source_name);
    volatile uint32_t *dma_regs_virt =
        (volatile uint32_t *)map_physical(devmem_fd, DMA_REGS_BASE_PHYS, DMA_REGS_SPAN, "DMA registers");
    uint8_t *ddr_reserved_virt =
        (uint8_t *)map_physical(devmem_fd, ddr_reserved_base_phys, DDR_RESERVED_SPAN, "reserved DDR");

    /* VIRTUAL pointers to the same buffers, for the CPU */
    volatile uint64_t *ddr_input_record_virt  = (volatile uint64_t *)(ddr_reserved_virt + DDR_INPUT_RECORD_OFFSET);
    volatile uint32_t *ddr_output_logits_virt = (volatile uint32_t *)(ddr_reserved_virt + DDR_OUTPUT_LOGITS_OFFSET);

    /* ---- input files ---- */
    FILE *labels_file = fopen(LABELS_BIN_PATH, "rb");
    if (!labels_file) { fprintf(stderr, "open %s: %s\n", LABELS_BIN_PATH, strerror(errno)); return 1; }
    read_exact(LABELS_BIN_PATH, labels_file, true_class, (size_t)n_records);
    fclose(labels_file);

    FILE *snr_file = fopen(SNR_BIN_PATH, "rb");
    if (!snr_file) { fprintf(stderr, "open %s: %s\n", SNR_BIN_PATH, strerror(errno)); return 1; }
    read_exact(SNR_BIN_PATH, snr_file, frame_snr_db, (size_t)n_records);
    fclose(snr_file);

    FILE *golden_file = fopen(GOLDEN_BIN_PATH, "rb");
    if (!golden_file) { fprintf(stderr, "open %s: %s\n", GOLDEN_BIN_PATH, strerror(errno)); return 1; }
    read_exact(GOLDEN_BIN_PATH, golden_file, golden_logits, sizeof(golden_logits));
    fclose(golden_file);

    FILE *samples_file = fopen(SAMPLES_BIN_PATH, "rb");
    if (!samples_file) { fprintf(stderr, "open %s: %s\n", SAMPLES_BIN_PATH, strerror(errno)); return 1; }

    /* frame in normal RAM before the copy into the reserved DDR (8-byte aligned) */
    static uint64_t host_record_buffer[INPUT_RECORD_BYTES / 8];

    /* ---- DMA reset ---- */
    dma_reset(dma_regs_virt, REG_MM2S_DMACR);
    dma_reset(dma_regs_virt, REG_S2MM_DMACR);

    printf("Running %d RadioML frames...\n", n_records);
    double run_start_sec = now_sec();

    for (int record_index = 0; record_index < n_records; record_index++) {
        /* file read and copy into DDR: OUTSIDE the time measurement */
        read_exact(SAMPLES_BIN_PATH, samples_file, host_record_buffer, INPUT_RECORD_BYTES);
        copy_to_ddr64(ddr_input_record_virt, host_record_buffer, INPUT_RECORD_BYTES);
        for (int class_index = 0; class_index < K_CLASSES; class_index++)
            ddr_output_logits_virt[class_index] = 0xDEADBEEFu;   /* sentinel: logit not written */

        /* 1) S2MM armed first (the decoder does not wait) */
        dma_s2mm_start(dma_regs_virt, ddr_output_logits_phys, OUTPUT_LOGITS_BYTES);

        /* 2) start of the measurement: the MM2S of the frame starts */
        double inference_start_sec = now_sec();
        dma_mm2s_start(dma_regs_virt, ddr_input_record_phys, INPUT_RECORD_BYTES);
        dma_wait_done(dma_regs_virt, REG_MM2S_DMASR, "MM2S");
        double input_done_sec = now_sec();

        /* 3) end of the measurement: the 24 logits are in DDR */
        dma_wait_done(dma_regs_virt, REG_S2MM_DMASR, "S2MM");
        double inference_end_sec = now_sec();

        input_transfer_sec[record_index] = input_done_sec - inference_start_sec;
        inference_time_sec[record_index] = inference_end_sec - inference_start_sec;

        /* 4) read the logits */
        for (int class_index = 0; class_index < K_CLASSES; class_index++)
            board_logits[record_index][class_index] = (int32_t)ddr_output_logits_virt[class_index];

        /* print the first frames for a visual check */
        if (record_index < 3) {
            print_logits("board logits ", record_index, board_logits[record_index]);
            print_logits("golden logits", record_index, golden_logits[record_index]);
            printf("  frame %d  (%.3f ms)\n", record_index, 1e3 * inference_time_sec[record_index]);
        }
        if ((record_index + 1) % 1000 == 0 || record_index + 1 == n_records) {
            printf("  %4d/%d\n", record_index + 1, n_records);
            fflush(stdout);
        }
    }
    fclose(samples_file);
    double run_total_sec = now_sec() - run_start_sec;

    /* ---- save the board logits ---- */
    FILE *board_logits_file = fopen(BOARD_LOGITS_BIN_PATH, "wb");
    if (board_logits_file) {
        fwrite(board_logits, sizeof(int32_t) * K_CLASSES, (size_t)n_records, board_logits_file);
        fclose(board_logits_file);
    } else {
        fprintf(stderr, "warning: cannot write %s\n", BOARD_LOGITS_BIN_PATH);
    }

    /* ---- 1. bit-exact comparison with the golden ---- */
    int n_compared = (n_records < N_GOLDEN_RECORDS) ? n_records : N_GOLDEN_RECORDS;
    int n_records_mismatch = 0, n_logits_mismatch = 0, n_sentinel_left = 0;
    printf("\n=== 1. Board logits vs fixed-point golden (%d frames) ===\n", n_compared);
    for (int record_index = 0; record_index < n_compared; record_index++) {
        int record_differs = 0;
        for (int class_index = 0; class_index < K_CLASSES; class_index++) {
            if ((uint32_t)board_logits[record_index][class_index] == 0xDEADBEEFu) n_sentinel_left++;
            if (board_logits[record_index][class_index] != golden_logits[record_index][class_index]) {
                n_logits_mismatch++;
                record_differs = 1;
            }
        }
        if (record_differs) {
            if (n_records_mismatch < N_MISMATCH_PRINT_MAX) {
                printf("  MISMATCH frame %d:\n", record_index);
                print_logits("board ", record_index, board_logits[record_index]);
                print_logits("golden", record_index, golden_logits[record_index]);
            }
            n_records_mismatch++;
        }
    }
    printf("  identical frames: %d/%d   different logits: %d/%d\n",
           n_compared - n_records_mismatch, n_compared, n_logits_mismatch, n_compared * K_CLASSES);
    if (n_sentinel_left)
        printf("  WARNING: %d logits never written by the S2MM (sentinel 0xDEADBEEF)\n", n_sentinel_left);
    printf("  %s\n", n_records_mismatch == 0 ? "OK: BIT-EXACT with the golden"
                                             : "NOT bit-exact: see the frames above");

    /* ---- 2. accuracy over all SNRs, at SNR >= 0 and >= 10, and per SNR ---- */
    long n_correct_snr[N_SNR_LEVELS] = {0}, n_total_snr[N_SNR_LEVELS] = {0};
    long n_correct_all = 0, n_correct_ge0 = 0, n_total_ge0 = 0, n_correct_ge10 = 0, n_total_ge10 = 0;
    for (int record_index = 0; record_index < n_records; record_index++) {
        int is_correct = (predicted_class(board_logits[record_index]) == true_class[record_index]);
        int snr_db = frame_snr_db[record_index];
        int snr_level = (snr_db - SNR_MIN_DB) / SNR_STEP_DB;      /* -20 -> 0, ..., 30 -> 25 */
        n_correct_all += is_correct;
        if (snr_level >= 0 && snr_level < N_SNR_LEVELS) {
            n_correct_snr[snr_level] += is_correct;
            n_total_snr[snr_level]++;
        }
        if (snr_db >= 0)  { n_correct_ge0  += is_correct; n_total_ge0++;  }
        if (snr_db >= 10) { n_correct_ge10 += is_correct; n_total_ge10++; }
    }
    printf("\n=== 2. Accuracy (argmax of the board logits, %d frames) ===\n", n_records);
    printf("  all SNRs   %.4f  (%ld/%d)        expected fixed-point: 0.5995 on 9984\n",
           (double)n_correct_all / n_records, n_correct_all, n_records);
    if (n_total_ge0)
        printf("  SNR >= 0   %.4f  (%ld/%ld)        expected fixed-point: 0.8654\n",
               (double)n_correct_ge0 / n_total_ge0, n_correct_ge0, n_total_ge0);
    if (n_total_ge10)
        printf("  SNR >= 10  %.4f  (%ld/%ld)        expected fixed-point: 0.9306\n",
               (double)n_correct_ge10 / n_total_ge10, n_correct_ge10, n_total_ge10);
    printf("  per SNR (dB: accuracy):\n");
    for (int snr_level = 0; snr_level < N_SNR_LEVELS; snr_level++) {
        if (!n_total_snr[snr_level]) continue;
        printf("    %+3d: %.3f%s", SNR_MIN_DB + SNR_STEP_DB * snr_level,
               (double)n_correct_snr[snr_level] / n_total_snr[snr_level],
               (snr_level % 6 == 5) ? "\n" : "  ");
    }
    printf("\n");

    /* recall per class at SNR >= 10 (where the residual error shows) */
    long n_correct_class[K_CLASSES] = {0}, n_total_class[K_CLASSES] = {0};
    for (int record_index = 0; record_index < n_records; record_index++) {
        if (frame_snr_db[record_index] < 10 || true_class[record_index] >= K_CLASSES) continue;
        n_total_class[true_class[record_index]]++;
        n_correct_class[true_class[record_index]] +=
            (predicted_class(board_logits[record_index]) == true_class[record_index]);
    }
    printf("  recall per class at SNR >= 10:\n");
    for (int class_index = 0; class_index < K_CLASSES; class_index++) {
        if (!n_total_class[class_index]) continue;
        printf("    %-9s %.3f%s", CLASS_NAME[class_index],
               (double)n_correct_class[class_index] / n_total_class[class_index],
               (class_index % 4 == 3) ? "\n" : "  ");
    }
    printf("\n");

    /* ---- 3. times ---- */
    double infer_min, infer_mean, infer_max, input_min, input_mean, input_max;
    time_stats(inference_time_sec, n_records, &infer_min, &infer_mean, &infer_max);
    time_stats(input_transfer_sec, n_records, &input_min, &input_mean, &input_max);
    printf("\n=== 3. Inference time (measured with CLOCK_MONOTONIC) ===\n");
    printf("  pl_clk0: ");
    if (pl0_clock_mhz > 0) printf("%.4f MHz (%s)\n", pl0_clock_mhz, pl0_source_name);
    else                   printf("not available (%s)\n", pl0_source_name);
    printf("  inference per frame (MM2S started -> 24 logits in DDR):\n");
    printf("    min %.3f ms   mean %.3f ms   max %.3f ms\n", 1e3 * infer_min, 1e3 * infer_mean, 1e3 * infer_max);
    if (pl0_clock_mhz > 0) {
        /* cycles = measured time x frequency read: computed, not counted in the PL */
        printf("    = about %.0f cycles (mean time x pl_clk0)\n", infer_mean * pl0_clock_mhz * 1e6);
        printf("    = about %.1f cycles per step per layer (/ (6 x %d))\n",
               infer_mean * pl0_clock_mhz * 1e6 / (6.0 * L_SEQ), L_SEQ);
    }
    printf("  of which input transfer (MM2S done):\n");
    printf("    min %.3f ms   mean %.3f ms   max %.3f ms\n", 1e3 * input_min, 1e3 * input_mean, 1e3 * input_max);
    if (pl0_clock_mhz > 0)
        printf("    = about %.0f cycles (mean time x pl_clk0)\n", input_mean * pl0_clock_mhz * 1e6);
    printf("  inference-only throughput: %.1f frames/s\n", 1.0 / infer_mean);
    printf("  total test time: %.2f s (file reads and DDR copies included)\n", run_total_sec);

    munmap(ddr_reserved_virt, DDR_RESERVED_SPAN);
    munmap((void *)dma_regs_virt, DMA_REGS_SPAN);
    close(devmem_fd);
    return 0;
}
