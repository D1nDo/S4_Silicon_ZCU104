/* ============================================================================
 * pl0_clk_set.c -- read and set the frequency of a PS->PL clock (pl_clk0..3)
 * of the Zynq UltraScale+ (ZCU104) from Linux userspace, through /dev/mem.
 *
 * Works with ANY bitstream: it only touches the PLx_REF_CTRL register of the
 * PS (source and two dividers). The PLLs are NEVER reprogrammed:
 *   IOPLL feeds UART, Ethernet, USB, I2C, QSPI, LPD bus;
 *   RPLL  feeds the SD card (root filesystem), LPD/IOU switch, R5, SPI;
 *   DPLL  feeds the DDR controller.
 * Changing them would hang Linux.
 *
 * Achievable frequency:
 *   f_PL = f_PLL / (DIVISOR0 * DIVISOR1),  PLL = IOPLL or RPLL, DIVISOR0/1 = 1..63
 *   f_PLL = f_REF * (FBDIV + FRAC/65536) / (DIV2 ? 2 : 1),  f_REF = 33.333 MHz (ZCU104)
 * The program tries every combination and picks the best one.
 *
 * WARNING: the program sets the frequency you ask for, but the design works
 * correctly only up to the frequency at which the bitstream met timing
 * (WNS >= 0 in the Vivado report). Above that: results are not guaranteed.
 * Change the clock while the PL is idle (no DMA transfer in progress).
 *
 * Registers (CRL_APB @ 0xFF5E0000):
 *   IOPLL_CTRL     0x020   FBDIV [14:8], DIV2 [16], BYPASS [3]
 *   IOPLL_FRAC_CFG 0x028   ENABLED [31], DATA [15:0]
 *   RPLL_CTRL      0x030   same as IOPLL_CTRL
 *   RPLL_FRAC_CFG  0x038   same as IOPLL_FRAC_CFG
 *   PL0_REF_CTRL   0x0C0   SRCSEL [2:0], DIVISOR0 [13:8], DIVISOR1 [21:16], CLKACT [24]
 *   PL1/2/3_REF_CTRL 0x0C4 / 0x0C8 / 0x0CC
 *   SRCSEL: 0 = IOPLL, 2 = RPLL, 3 = DPLL_CLK_TO_LPD (read only, never selected)
 *
 * Build (on the board):  gcc -O2 -o pl0_clk_set pl0_clk_set.c -lm
 * Usage (root):
 *   ./pl0_clk_set                    print the PLLs and the current PL clock
 *   ./pl0_clk_set 200                set the highest frequency <= 200 MHz
 *   ./pl0_clk_set 200 --nearest      set the frequency closest to 200 MHz (also above)
 *   ./pl0_clk_set 200 --dry-run      show what would be done, without writing
 *   ./pl0_clk_set --list 150 280     list the achievable frequencies between 150 and 280 MHz
 *   option --pl N (N = 0..3) for pl_clk1..3 instead of pl_clk0
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
#include <math.h>

/* ------------------------------- CONFIG --------------------------------- */

#define CRL_APB_BASE_PHYS      0xFF5E0000UL   /* LPD clock registers */
#define CRL_APB_SPAN           0x1000UL

#define REG_IOPLL_CTRL         0x020
#define REG_IOPLL_FRAC_CFG     0x028
#define REG_RPLL_CTRL          0x030
#define REG_RPLL_FRAC_CFG      0x038
#define REG_PL0_REF_CTRL       0x0C0          /* PLn_REF_CTRL = 0x0C0 + 4*n */

#define PSS_REF_CLK_MHZ        33.333         /* PS reference oscillator on the ZCU104 */

#define SRCSEL_IOPLL           0
#define SRCSEL_RPLL            2
#define SRCSEL_DPLL_TO_LPD     3

#define DIVISOR_MIN            1
#define DIVISOR_MAX            63             /* 6-bit fields */

/* fields of PLx_REF_CTRL */
#define REF_CTRL_SRCSEL_MASK   0x00000007u
#define REF_CTRL_DIV0_SHIFT    8
#define REF_CTRL_DIV1_SHIFT    16
#define REF_CTRL_DIV_MASK      0x3Fu
#define REF_CTRL_CLKACT        (1u << 24)

/* ------------------------------- registers ------------------------------ */

static volatile uint32_t *crl_apb_regs_virt;   /* CRL_APB mapped into the process */

static inline uint32_t crl_read(unsigned reg_offset)
{
    return crl_apb_regs_virt[reg_offset / 4];
}

static inline void crl_write(unsigned reg_offset, uint32_t value)
{
    crl_apb_regs_virt[reg_offset / 4] = value;
}

/* output frequency of an LPD PLL from its CTRL and FRAC_CFG registers */
static double pll_output_mhz(unsigned ctrl_offset, unsigned frac_offset, const char *pll_name, int verbose)
{
    uint32_t pll_ctrl     = crl_read(ctrl_offset);
    uint32_t pll_frac_cfg = crl_read(frac_offset);
    unsigned feedback_div = (pll_ctrl >> 8) & 0x7Fu;        /* FBDIV: integer multiplier */
    unsigned output_div2  = (pll_ctrl >> 16) & 0x1u;        /* DIV2: output divided by 2 */
    unsigned pll_bypass   = (pll_ctrl >> 3) & 0x1u;         /* BYPASS: output = reference */
    double   frac_part    = ((pll_frac_cfg >> 31) & 1u) ? (pll_frac_cfg & 0xFFFFu) / 65536.0 : 0.0;

    double vco_mhz    = PSS_REF_CLK_MHZ * (feedback_div + frac_part);
    double output_mhz = pll_bypass ? PSS_REF_CLK_MHZ : vco_mhz / (output_div2 ? 2.0 : 1.0);

    if (verbose)
        printf("  %-5s FBDIV=%u%s DIV2=%u%s -> %.3f MHz\n", pll_name, feedback_div,
               frac_part > 0.0 ? " (fractional)" : "", output_div2,
               pll_bypass ? " BYPASS" : "", output_mhz);
    return output_mhz;
}

/* current frequency of a PL clock from the fields of its PLx_REF_CTRL */
static double pl_clock_mhz(uint32_t ref_ctrl, double iopll_mhz, double rpll_mhz, const char **source_name)
{
    unsigned source_select = ref_ctrl & REF_CTRL_SRCSEL_MASK;
    unsigned divisor0 = (ref_ctrl >> REF_CTRL_DIV0_SHIFT) & REF_CTRL_DIV_MASK;
    unsigned divisor1 = (ref_ctrl >> REF_CTRL_DIV1_SHIFT) & REF_CTRL_DIV_MASK;
    if (divisor0 == 0) divisor0 = 1;               /* 0 behaves as 1 */
    if (divisor1 == 0) divisor1 = 1;

    double source_mhz;
    switch (source_select) {
    case SRCSEL_IOPLL: *source_name = "IOPLL"; source_mhz = iopll_mhz; break;
    case SRCSEL_RPLL:  *source_name = "RPLL";  source_mhz = rpll_mhz;  break;
    case SRCSEL_DPLL_TO_LPD: *source_name = "DPLL_TO_LPD (not decoded)"; return -1.0;
    default:           *source_name = "unknown"; return -1.0;
    }
    return source_mhz / divisor0 / divisor1;
}

/* --------------------------- divider search ----------------------------- */

typedef struct {
    unsigned source_select;   /* SRCSEL_IOPLL or SRCSEL_RPLL */
    unsigned divisor0;
    unsigned divisor1;
    double   frequency_mhz;
} clock_setting_t;

/* choose source and dividers.
 *   below_only = 1: the highest frequency that does NOT exceed the request
 *   below_only = 0: the closest one overall
 * For equal frequencies, DIVISOR1 = 1 (a single divider) is preferred. */
static int find_best_setting(double requested_mhz, double iopll_mhz, double rpll_mhz,
                             int below_only, clock_setting_t *best_setting)
{
    const unsigned source_list[2] = { SRCSEL_IOPLL, SRCSEL_RPLL };
    const double   source_mhz[2]  = { iopll_mhz, rpll_mhz };
    int found = 0;
    double best_error = 1e30;

    for (int source_index = 0; source_index < 2; source_index++) {
        for (unsigned divisor1 = DIVISOR_MIN; divisor1 <= DIVISOR_MAX; divisor1++) {
            for (unsigned divisor0 = DIVISOR_MIN; divisor0 <= DIVISOR_MAX; divisor0++) {
                double candidate_mhz = source_mhz[source_index] / divisor0 / divisor1;
                /* 0.01% tolerance: whoever types "213.33" wants 213.3312, not 187.5 */
                if (below_only && candidate_mhz > requested_mhz * (1.0 + 1e-4)) continue;
                double candidate_error = fabs(candidate_mhz - requested_mhz);
                /* strictly better: combinations with a small divisor1 come first */
                if (candidate_error < best_error - 1e-9) {
                    best_error = candidate_error;
                    best_setting->source_select = source_list[source_index];
                    best_setting->divisor0      = divisor0;
                    best_setting->divisor1      = divisor1;
                    best_setting->frequency_mhz = candidate_mhz;
                    found = 1;
                }
            }
        }
    }
    return found;
}

/* comparison used to sort the listed frequencies */
static int compare_setting_frequency(const void *setting_a, const void *setting_b)
{
    double freq_a = ((const clock_setting_t *)setting_a)->frequency_mhz;
    double freq_b = ((const clock_setting_t *)setting_b)->frequency_mhz;
    return (freq_a > freq_b) - (freq_a < freq_b);
}

/* list the distinct achievable frequencies in [low_mhz, high_mhz] */
static void list_settings(double low_mhz, double high_mhz, double iopll_mhz, double rpll_mhz)
{
    static clock_setting_t all_settings[2 * DIVISOR_MAX * DIVISOR_MAX];
    int n_settings = 0;
    const unsigned source_list[2] = { SRCSEL_IOPLL, SRCSEL_RPLL };
    const double   source_mhz[2]  = { iopll_mhz, rpll_mhz };

    for (int source_index = 0; source_index < 2; source_index++)
        for (unsigned divisor1 = DIVISOR_MIN; divisor1 <= DIVISOR_MAX; divisor1++)
            for (unsigned divisor0 = DIVISOR_MIN; divisor0 <= DIVISOR_MAX; divisor0++) {
                double candidate_mhz = source_mhz[source_index] / divisor0 / divisor1;
                if (candidate_mhz < low_mhz || candidate_mhz > high_mhz) continue;
                all_settings[n_settings].source_select = source_list[source_index];
                all_settings[n_settings].divisor0      = divisor0;
                all_settings[n_settings].divisor1      = divisor1;
                all_settings[n_settings].frequency_mhz = candidate_mhz;
                n_settings++;
            }
    qsort(all_settings, (size_t)n_settings, sizeof(clock_setting_t), compare_setting_frequency);

    printf("Achievable frequencies between %.3f and %.3f MHz:\n", low_mhz, high_mhz);
    double last_printed_mhz = -1.0;
    for (int setting_index = 0; setting_index < n_settings; setting_index++) {
        /* one line per distinct frequency (equivalent combinations skipped) */
        if (fabs(all_settings[setting_index].frequency_mhz - last_printed_mhz) < 1e-6) continue;
        printf("  %10.4f MHz   %-5s / %2u / %2u\n", all_settings[setting_index].frequency_mhz,
               all_settings[setting_index].source_select == SRCSEL_IOPLL ? "IOPLL" : "RPLL",
               all_settings[setting_index].divisor0, all_settings[setting_index].divisor1);
        last_printed_mhz = all_settings[setting_index].frequency_mhz;
    }
}

/* --------------------------------- main --------------------------------- */

static void print_usage(const char *program_name)
{
    fprintf(stderr,
            "Usage: %s [MHz] [--nearest] [--dry-run] [--pl N]\n"
            "       %s --list MHz_MIN MHz_MAX\n", program_name, program_name);
}

int main(int argc, char **argv)
{
    double requested_mhz = -1.0;      /* < 0: read only */
    int nearest_mode = 0, dry_run = 0, pl_clock_index = 0;
    int list_mode = 0;
    double list_low_mhz = 0.0, list_high_mhz = 0.0;

    /* ---- arguments ---- */
    for (int arg_index = 1; arg_index < argc; arg_index++) {
        if (!strcmp(argv[arg_index], "--nearest")) nearest_mode = 1;
        else if (!strcmp(argv[arg_index], "--dry-run")) dry_run = 1;
        else if (!strcmp(argv[arg_index], "--pl") && arg_index + 1 < argc)
            pl_clock_index = atoi(argv[++arg_index]);
        else if (!strcmp(argv[arg_index], "--list") && arg_index + 2 < argc) {
            list_mode = 1;
            list_low_mhz  = atof(argv[++arg_index]);
            list_high_mhz = atof(argv[++arg_index]);
        } else if (argv[arg_index][0] != '-' && requested_mhz < 0)
            requested_mhz = atof(argv[arg_index]);
        else { print_usage(argv[0]); return 1; }
    }
    if (pl_clock_index < 0 || pl_clock_index > 3) { fprintf(stderr, "--pl must be 0..3\n"); return 1; }
    if (requested_mhz == 0.0) { fprintf(stderr, "invalid frequency\n"); return 1; }
    const unsigned ref_ctrl_offset = REG_PL0_REF_CTRL + 4u * (unsigned)pl_clock_index;

    /* ---- map CRL_APB ---- */
    int devmem_fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (devmem_fd < 0) { fprintf(stderr, "open /dev/mem (root required): %s\n", strerror(errno)); return 1; }
    void *mapped_virt = mmap(NULL, CRL_APB_SPAN, PROT_READ | PROT_WRITE, MAP_SHARED,
                             devmem_fd, (off_t)CRL_APB_BASE_PHYS);
    if (mapped_virt == MAP_FAILED) { fprintf(stderr, "mmap CRL_APB: %s\n", strerror(errno)); return 1; }
    crl_apb_regs_virt = (volatile uint32_t *)mapped_virt;

    /* ---- current state ---- */
    printf("LPD PLLs (reference %.3f MHz):\n", PSS_REF_CLK_MHZ);
    double iopll_mhz = pll_output_mhz(REG_IOPLL_CTRL, REG_IOPLL_FRAC_CFG, "IOPLL", 1);
    double rpll_mhz  = pll_output_mhz(REG_RPLL_CTRL,  REG_RPLL_FRAC_CFG,  "RPLL",  1);

    uint32_t ref_ctrl_before = crl_read(ref_ctrl_offset);
    const char *source_before = "";
    double pl_mhz_before = pl_clock_mhz(ref_ctrl_before, iopll_mhz, rpll_mhz, &source_before);
    printf("\ncurrent pl_clk%d: PL%d_REF_CTRL=0x%08x  source %s  DIV0=%u  DIV1=%u  CLKACT=%u\n",
           pl_clock_index, pl_clock_index, ref_ctrl_before, source_before,
           (ref_ctrl_before >> REF_CTRL_DIV0_SHIFT) & REF_CTRL_DIV_MASK,
           (ref_ctrl_before >> REF_CTRL_DIV1_SHIFT) & REF_CTRL_DIV_MASK,
           (ref_ctrl_before & REF_CTRL_CLKACT) ? 1u : 0u);
    if (pl_mhz_before > 0) printf("              = %.4f MHz\n", pl_mhz_before);

    if (list_mode) {
        printf("\n");
        list_settings(list_low_mhz, list_high_mhz, iopll_mhz, rpll_mhz);
        goto cleanup;
    }
    if (requested_mhz < 0) goto cleanup;           /* read only */

    /* ---- choose the dividers ---- */
    clock_setting_t chosen_setting;
    if (!find_best_setting(requested_mhz, iopll_mhz, rpll_mhz, !nearest_mode, &chosen_setting)) {
        fprintf(stderr, "no combination %s %.3f MHz\n",
                nearest_mode ? "close to" : "below", requested_mhz);
        munmap(mapped_virt, CRL_APB_SPAN);
        close(devmem_fd);
        return 1;
    }
    printf("\nRequested %.4f MHz (%s) -> %s / %u / %u = %.4f MHz (error %+.4f MHz)\n",
           requested_mhz, nearest_mode ? "closest" : "highest not above",
           chosen_setting.source_select == SRCSEL_IOPLL ? "IOPLL" : "RPLL",
           chosen_setting.divisor0, chosen_setting.divisor1, chosen_setting.frequency_mhz,
           chosen_setting.frequency_mhz - requested_mhz);

    /* new register value: only SRCSEL, DIVISOR0 and DIVISOR1 change,
     * the other bits (CLKACT included) are kept */
    uint32_t ref_ctrl_after = ref_ctrl_before;
    ref_ctrl_after &= ~(REF_CTRL_SRCSEL_MASK
                        | (REF_CTRL_DIV_MASK << REF_CTRL_DIV0_SHIFT)
                        | (REF_CTRL_DIV_MASK << REF_CTRL_DIV1_SHIFT));
    ref_ctrl_after |= chosen_setting.source_select
                      | (chosen_setting.divisor0 << REF_CTRL_DIV0_SHIFT)
                      | (chosen_setting.divisor1 << REF_CTRL_DIV1_SHIFT)
                      | REF_CTRL_CLKACT;           /* clock active */

    if (dry_run) {
        printf("--dry-run: would write PL%d_REF_CTRL = 0x%08x (now 0x%08x), nothing changed\n",
               pl_clock_index, ref_ctrl_after, ref_ctrl_before);
        goto cleanup;
    }

    /* ---- write and verify ---- */
    crl_write(ref_ctrl_offset, ref_ctrl_after);
    usleep(1000);                                  /* 1 ms to settle */
    uint32_t ref_ctrl_readback = crl_read(ref_ctrl_offset);
    const char *source_after = "";
    double pl_mhz_after = pl_clock_mhz(ref_ctrl_readback, iopll_mhz, rpll_mhz, &source_after);
    printf("Wrote PL%d_REF_CTRL = 0x%08x, read back 0x%08x -> %s = %.4f MHz\n",
           pl_clock_index, ref_ctrl_after, ref_ctrl_readback, source_after, pl_mhz_after);
    if (ref_ctrl_readback != ref_ctrl_after)
        fprintf(stderr, "WARNING: the register read back differs (write blocked?)\n");
    printf("Remember: correct only if the bitstream meets timing at this frequency.\n");

cleanup:
    munmap(mapped_virt, CRL_APB_SPAN);
    close(devmem_fd);
    return 0;
}
