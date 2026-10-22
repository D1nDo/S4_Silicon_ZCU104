/* ============================================================================
 * s4d_ZCU104_sCIFAR.c -- board test of the S4D sCIFAR bitstream (sequential
 * CIFAR-10, grayscale) on ZCU104, from Linux userspace through /dev/mem.
 *
 * For each test image it:
 *   1. copies the image (1024 bytes) into the input buffer in DDR;
 *   2. arms the S2MM to receive the 10 logits (40 bytes);
 *   3. starts the MM2S of the image towards the encoder;
 *   4. waits for the end of the S2MM and reads the 10 int32 logits.
 * The weights do NOT go through the DMA: they are in the on-chip ROM
 * (s4d_weightrom_axis).
 *
 * At the end it prints:
 *   - bit-exact comparison of the logits with the fixed-point golden (first
 *     1000 images);
 *   - top-1 accuracy on the 10000 test images, the standard sCIFAR metric.
 *     The bit-accurate fixed-point model gives 7640/10000 = 0.7640: if the
 *     board is bit-exact it must give EXACTLY the same number of correct
 *     predictions;
 *   - recall per class;
 *   - inference time per image (min / mean / max), throughput and cycles,
 *     using the ACTUAL pl_clk0 frequency read from the PS registers.
 * It saves the logits of ALL the images: they are used for the confusion matrix.
 *
 * Map from the S4D_sCIFAR block design (same as S4D_PTB_XL, S4D_RadioML, S4D_S_C):
 *   axi_dma_0 @ 0xA000_0000 (S_AXI_LITE via HPM0_FPD), simple mode, no SG
 *     MM2S 64 bit -> s4d_encoder_axis   (image, NIN 1, uint8 pixels)
 *     S2MM 32 bit <- s4d_decoder_axis   (10 logits, tlast on the 10th)
 *   MM2S and S2MM towards HPC0_DDR_LOW (0x0000_0000 - 0x7FFF_FFFF).
 *
 * Image format: 32 x 32 grayscale pixels, row-major, one unsigned byte per
 * pixel (raw value 0..255) -> L = 1024 steps = 1024 bytes = 128 beats of
 * 64 bit (8 pixels per beat, first pixel in the low bits).
 *
 * Files expected in the working directory (board/make_board_files.py sCIFAR):
 *   samples_sCIFAR_10000.bin       10000 x 1024 bytes, CIFAR-10 test set in dataset order
 *   labels_sCIFAR_10000.bin        10000 bytes, true class 0..9 (CLASS_NAME order)
 *   logits_golden_sCIFAR_1000.bin  1000 x 10 int32 little-endian [image][class]
 * Writes:
 *   logits_board_ZCU104_sCIFAR.bin N x 10 int32, the logits read from the board
 *
 * DDR MEMORY FOR THE DMA: as in the other tests, a region that Linux does NOT
 * use, below 2 GB (HPC0_DDR_LOW); default DDR_RESERVED_BASE_PHYS below,
 * it can be changed from the command line.
 *
 * Build (on the board):  gcc -O2 -o s4d_ZCU104_sCIFAR s4d_ZCU104_sCIFAR.c
 * Usage (root):  ./s4d_ZCU104_sCIFAR [N_IMAGES] [BASE_DDR_HEX]
 *                default N_IMAGES = 10000, BASE_DDR_HEX = 0x70000000
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

#define BUILD_TAG  "S4D sCIFAR -- ZCU104, logits + golden + inference time"

/* AXI DMA registers (address from the block design) */
#define DMA_REGS_BASE_PHYS      0xA0000000UL
#define DMA_REGS_SPAN           0x10000UL         /* 64K as in the Address Editor */

/* model */
#define L_SEQ                   1024              /* pixels per image = steps */
#define N_INPUTS                1                 /* one pixel per step */
#define INPUT_RECORD_BYTES      (L_SEQ * N_INPUTS * 1)   /* 1024 bytes = 128 beats of 64 bit */
#define K_CLASSES               10                /* CIFAR-10 classes */
#define OUTPUT_LOGITS_BYTES     (K_CLASSES * 4)          /* 10 int32 logits = 40 bytes */

#define N_RECORDS_MAX           10000             /* images in the CIFAR-10 test set */
#define N_GOLDEN_RECORDS        1000              /* images with golden logits (the first ones) */
#define FIXED_POINT_CORRECT     7640              /* correct predictions of the fixed-point model on 10000 (0.7640) */

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
#define DDR_INPUT_RECORD_OFFSET 0x00000000UL      /* image read by the MM2S */
#define DDR_OUTPUT_LOGITS_OFFSET 0x00008000UL     /* 10 logits written by the S2MM (after 32 KB) */

#define SAMPLES_BIN_PATH        "samples_sCIFAR_10000.bin"
#define LABELS_BIN_PATH         "labels_sCIFAR_10000.bin"
#define GOLDEN_BIN_PATH         "logits_golden_sCIFAR_1000.bin"
#define BOARD_LOGITS_BIN_PATH   "logits_board_ZCU104_sCIFAR.bin"

#define DMA_TIMEOUT_SEC         10.0              /* an image takes ~ms: 10 s = stuck */
#define N_MISMATCH_PRINT_MAX    10                /* how many images differing from the golden to print */

static const char *CLASS_NAME[K_CLASSES] = {
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck" };

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

static int32_t board_logits[N_RECORDS_MAX][K_CLASSES];         /* logits read from the board */
static int32_t golden_logits[N_GOLDEN_RECORDS][K_CLASSES];     /* fixed-point golden logits of the first images */
static uint8_t true_class[N_RECORDS_MAX];                      /* true class 0..9 */
static double  inference_time_sec[N_RECORDS_MAX];              /* MM2S started -> S2MM done */
static double  input_transfer_sec[N_RECORDS_MAX];              /* MM2S started -> MM2S done */

/* predicted class = index of the largest logit; on ties the lowest index
 * wins (strictly greater), as argmax of the golden */
static int predicted_class(const int32_t *record_logits)
{
    int best_class = 0;
    for (int class_index = 1; class_index < K_CLASSES; class_index++)
        if (record_logits[class_index] > record_logits[best_class]) best_class = class_index;
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

/* print the logits of an image on one line */
static void print_logits(const char *row_label, int record_index, const int32_t *record_logits)
{
    printf("  img %d  %s: [", record_index, row_label);
    for (int class_index = 0; class_index < K_CLASSES; class_index++)
        printf("%s%d", class_index ? ", " : "", record_logits[class_index]);
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
            fprintf(stderr, "N_IMAGES must be between 1 and %d\n", N_RECORDS_MAX);
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
    printf("DMA @0x%08lx | image in DDR @0x%08lx (%d B) | logits in DDR @0x%08lx (%d B)\n",
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

    FILE *golden_file = fopen(GOLDEN_BIN_PATH, "rb");
    if (!golden_file) { fprintf(stderr, "open %s: %s\n", GOLDEN_BIN_PATH, strerror(errno)); return 1; }
    read_exact(GOLDEN_BIN_PATH, golden_file, golden_logits, sizeof(golden_logits));
    fclose(golden_file);

    FILE *samples_file = fopen(SAMPLES_BIN_PATH, "rb");
    if (!samples_file) { fprintf(stderr, "open %s: %s\n", SAMPLES_BIN_PATH, strerror(errno)); return 1; }

    /* image in normal RAM before the copy into the reserved DDR (8-byte aligned) */
    static uint64_t host_record_buffer[INPUT_RECORD_BYTES / 8];

    /* ---- DMA reset ---- */
    dma_reset(dma_regs_virt, REG_MM2S_DMACR);
    dma_reset(dma_regs_virt, REG_S2MM_DMACR);

    printf("Running %d sCIFAR images...\n", n_records);
    double run_start_sec = now_sec();

    for (int record_index = 0; record_index < n_records; record_index++) {
        /* file read and copy into DDR: OUTSIDE the time measurement */
        read_exact(SAMPLES_BIN_PATH, samples_file, host_record_buffer, INPUT_RECORD_BYTES);
        copy_to_ddr64(ddr_input_record_virt, host_record_buffer, INPUT_RECORD_BYTES);
        for (int class_index = 0; class_index < K_CLASSES; class_index++)
            ddr_output_logits_virt[class_index] = 0xDEADBEEFu;   /* sentinel: logit not written */

        /* 1) S2MM armed first (the decoder does not wait) */
        dma_s2mm_start(dma_regs_virt, ddr_output_logits_phys, OUTPUT_LOGITS_BYTES);

        /* 2) start of the measurement: the MM2S of the image starts */
        double inference_start_sec = now_sec();
        dma_mm2s_start(dma_regs_virt, ddr_input_record_phys, INPUT_RECORD_BYTES);
        dma_wait_done(dma_regs_virt, REG_MM2S_DMASR, "MM2S");
        double input_done_sec = now_sec();

        /* 3) end of the measurement: the 10 logits are in DDR */
        dma_wait_done(dma_regs_virt, REG_S2MM_DMASR, "S2MM");
        double inference_end_sec = now_sec();

        input_transfer_sec[record_index] = input_done_sec - inference_start_sec;
        inference_time_sec[record_index] = inference_end_sec - inference_start_sec;

        /* 4) read the logits */
        for (int class_index = 0; class_index < K_CLASSES; class_index++)
            board_logits[record_index][class_index] = (int32_t)ddr_output_logits_virt[class_index];

        /* print the first images for a visual check */
        if (record_index < 3) {
            print_logits("board logits ", record_index, board_logits[record_index]);
            print_logits("golden logits", record_index, golden_logits[record_index]);
            printf("  img %d  (%.3f ms)\n", record_index, 1e3 * inference_time_sec[record_index]);
        }
        if ((record_index + 1) % 1000 == 0 || record_index + 1 == n_records) {
            printf("  %5d/%d\n", record_index + 1, n_records);
            fflush(stdout);
        }
    }
    fclose(samples_file);
    double run_total_sec = now_sec() - run_start_sec;

    /* ---- save the board logits (for the confusion matrix) ---- */
    FILE *board_logits_file = fopen(BOARD_LOGITS_BIN_PATH, "wb");
    if (board_logits_file) {
        fwrite(board_logits, sizeof(int32_t) * K_CLASSES, (size_t)n_records, board_logits_file);
        fclose(board_logits_file);
    } else {
        fprintf(stderr, "warning: cannot write %s\n", BOARD_LOGITS_BIN_PATH);
    }

    /* ---- 1. bit-exact comparison with the golden (first N_GOLDEN_RECORDS images) ---- */
    int n_compared = (n_records < N_GOLDEN_RECORDS) ? n_records : N_GOLDEN_RECORDS;
    int n_records_mismatch = 0, n_logits_mismatch = 0, n_sentinel_left = 0;
    printf("\n=== 1. Board logits vs fixed-point golden (first %d images) ===\n", n_compared);
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
                printf("  MISMATCH img %d:\n", record_index);
                print_logits("board ", record_index, board_logits[record_index]);
                print_logits("golden", record_index, golden_logits[record_index]);
            }
            n_records_mismatch++;
        }
    }
    printf("  identical images: %d/%d   different logits: %d/%d\n",
           n_compared - n_records_mismatch, n_compared, n_logits_mismatch, n_compared * K_CLASSES);
    if (n_sentinel_left)
        printf("  WARNING: %d logits never written by the S2MM (sentinel 0xDEADBEEF)\n", n_sentinel_left);
    printf("  %s\n", n_records_mismatch == 0 ? "OK: BIT-EXACT with the golden"
                                             : "NOT bit-exact: see the images above");

    /* ---- 2. top-1 accuracy (standard metric of sCIFAR) ---- */
    long n_correct_class[K_CLASSES] = {0}, n_total_class[K_CLASSES] = {0};
    long n_correct_all = 0;
    for (int record_index = 0; record_index < n_records; record_index++) {
        int label = true_class[record_index];
        int is_correct = (predicted_class(board_logits[record_index]) == label);
        n_correct_all += is_correct;
        if (label < K_CLASSES) { n_correct_class[label] += is_correct; n_total_class[label]++; }
    }
    printf("\n=== 2. Top-1 accuracy (argmax of the board logits, %d images) ===\n", n_records);
    printf("  TOP-1 ACCURACY: %.4f  (%ld/%d)\n", (double)n_correct_all / n_records, n_correct_all, n_records);
    if (n_records == N_RECORDS_MAX)
        printf("  expected fixed-point: 0.7640 (%d/%d) -> %s\n", FIXED_POINT_CORRECT, N_RECORDS_MAX,
               n_correct_all == FIXED_POINT_CORRECT ? "EQUAL" : "DIFFERENT");
    printf("  recall per class:\n");
    for (int class_index = 0; class_index < K_CLASSES; class_index++) {
        if (!n_total_class[class_index]) continue;
        printf("    %-10s %.3f%s", CLASS_NAME[class_index],
               (double)n_correct_class[class_index] / n_total_class[class_index],
               (class_index % 5 == 4) ? "\n" : "  ");
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
    printf("  inference per image (MM2S started -> 10 logits in DDR):\n");
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
    printf("  inference-only throughput: %.1f images/s\n", 1.0 / infer_mean);
    printf("  total test time: %.2f s (file reads and DDR copies included)\n", run_total_sec);

    munmap(ddr_reserved_virt, DDR_RESERVED_SPAN);
    munmap((void *)dma_regs_virt, DMA_REGS_SPAN);
    close(devmem_fd);
    return 0;
}
