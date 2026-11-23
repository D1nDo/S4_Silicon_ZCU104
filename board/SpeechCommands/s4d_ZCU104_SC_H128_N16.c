/* ============================================================================
 * s4d_ZCU104_SC_H128_N16.c -- test in scheda del bitstream S4D_S_C (Speech Commands v0.02, 35 classi, H = 128, N = 16)
 * su ZCU104, da userspace Linux tramite /dev/mem.
 *
 * Basato su s4d_ZCU104_*.c del paper (stessa mappa del block design, stesso
 * protocollo DMA). Per ogni esempio del test set:
 *   1. copia l'esempio (24576 byte) nel buffer d'ingresso in DDR;
 *   2. arma la S2MM per ricevere i 35 logit;
 *   3. lancia la MM2S verso l'encoder e aspetta la fine della S2MM;
 *   4. legge i 35 logit int32.
 * I pesi NON passano dal DMA: sono nella ROM on-chip (s4d_weightrom_axis).
 *
 * Alla fine stampa:
 *   1. confronto BIT-EXACT dei logit con il golden fixed-point su TUTTI gli esempi;
 *   2. metrica del dataset calcolata sui logit della scheda: accuracy top-1 sul test split ufficiale;
 *   3. tempo totale e tempo medio per inferenza (CLOCK_MONOTONIC), cicli con
 *      la frequenza VERA di pl_clk0 letta dai registri del PS;
 *   4. potenza ed energia per inferenza dai sensori della ZCU104 in /sys/class/hwmon
 *      (ina226 sull'ingresso 12 V, regolatori irps5401): potenza a riposo e durante una
 *      sequenza di inferenze consecutive, energia per inferenza = potenza x tempo.
 *
 * Mappa dal block design S4D_S_C:
 *   axi_dma_0 @ 0xA000_0000 (S_AXI_LITE via HPM0_FPD), simple mode, no SG
 *     MM2S 64 bit -> s4d_encoder_axis   (esempio)
 *     S2MM 32 bit <- s4d_decoder_axis   (35 logit, tlast sull'ultimo)
 *
 * File attesi nella cartella di lavoro:
 *   samples_SC_11005.bin   11005 x 24576 byte   (stessi del test del paper)
 *   labels_SC_11005.bin   11005 byte   classe vera 0..34 (ordine CLASS_NAME)
 *   logits_golden_SC_11005.bin   11005 x 35 int32 little-endian [esempio][classe]
 * Scrive:
 *   logits_board_ZCU104_SC_H128_N16.bin   N x 35 int32, i logit letti dalla scheda
 *
 * Valori attesi (golden fixed-point bit-accurate, stessi esempi):
 *   accuracy top-1 0.9505 (float dopo fine-tuning con clamp 0.9506)
 *
 * Compilazione (sulla scheda):  gcc -O2 -o s4d_ZCU104_SC_H128_N16 s4d_ZCU104_SC_H128_N16.c -lpthread
 * Uso (root):  ./s4d_ZCU104_SC_H128_N16 [N_ESEMPI] [BASE_DDR_HEX] [SECONDI_ENERGIA]
 *              default N_ESEMPI = 11005, BASE_DDR_HEX = 0x70000000, SECONDI_ENERGIA = 10
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
#include <dirent.h>
#include <pthread.h>

/* ------------------------------- CONFIG --------------------------------- */

#define BUILD_TAG  "S4D Speech Commands v0.02, 35 classi H=128 N=16 -- ZCU104: bit-exact, metrica, tempo, energia"

#define DMA_REGS_BASE_PHYS      0xA0000000UL
#define DMA_REGS_SPAN           0x10000UL

#define L_SEQ                   1024
#define INPUT_RECORD_BYTES      24576
#define K_CLASSES               35
#define OUTPUT_LOGITS_BYTES     (K_CLASSES * 4)
#define N_RECORDS_MAX           11005

#define CRL_APB_BASE_PHYS       0xFF5E0000UL
#define CRL_APB_SPAN            0x1000UL
#define REG_IOPLL_CTRL          0x020
#define REG_IOPLL_FRAC_CFG      0x028
#define REG_RPLL_CTRL           0x030
#define REG_RPLL_FRAC_CFG       0x038
#define REG_PL0_REF_CTRL        0x0C0
#define PSS_REF_CLK_MHZ         33.333

#define DDR_RESERVED_BASE_PHYS  0x70000000UL
#define DDR_RESERVED_SPAN       0x00100000UL
#define DDR_INPUT_RECORD_OFFSET 0x00000000UL
#define DDR_OUTPUT_LOGITS_OFFSET 0x00008000UL     /* dopo 32 KB: l'esempio piu' grande e' 24 KB */

#define SAMPLES_BIN_PATH        "samples_SC_11005.bin"
#define LABELS_BIN_PATH         "labels_SC_11005.bin"
#define GOLDEN_BIN_PATH         "logits_golden_SC_11005.bin"
#define BOARD_LOGITS_BIN_PATH   "logits_board_ZCU104_SC_H128_N16.bin"

#define DMA_TIMEOUT_SEC         10.0
#define N_MISMATCH_PRINT_MAX    10

/* misura di potenza */
#define HWMON_DIR               "/sys/class/hwmon"
#define MAX_RAILS               32
#define POWER_SAMPLE_US         2000              /* periodo di campionamento dei sensori */
#define IDLE_SECONDS            3.0               /* potenza a riposo (PL caricata, nessuna inferenza) */

__attribute__((unused)) static const char *CLASS_NAME[K_CLASSES] = {
    "backward", "bed", "bird", "cat", "dog", "down", "eight", "five", "follow",
    "forward", "four", "go", "happy", "house", "learn", "left", "marvin", "nine",
    "no", "off", "on", "one", "right", "seven", "sheila", "six", "stop", "three",
    "tree", "two", "up", "visual", "wow", "yes", "zero" };

/* --------------------------- Registri AXI DMA ---------------------------- */

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

#define DMACR_RS         (1u << 0)
#define DMACR_RESET      (1u << 2)
#define DMASR_IDLE       (1u << 1)
#define DMASR_ERR_MASK   0x770u
#define DMASR_IOC_IRQ    (1u << 12)
#define DMASR_IRQ_MASK   0x7000u

/* ------------------------------- utilita' ------------------------------- */

static double now_sec(void)
{
    struct timespec time_now;
    clock_gettime(CLOCK_MONOTONIC, &time_now);
    return (double)time_now.tv_sec + (double)time_now.tv_nsec / 1e9;
}

static void *map_physical(int devmem_fd, uint64_t region_phys, size_t region_span, const char *region_name)
{
    void *region_virt = mmap(NULL, region_span, PROT_READ | PROT_WRITE, MAP_SHARED, devmem_fd, (off_t)region_phys);
    if (region_virt == MAP_FAILED) {
        fprintf(stderr, "mmap %s @0x%lx (%zu byte): %s\n", region_name,
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

static void dma_reset(volatile uint32_t *dma_regs_virt, unsigned dmacr_offset)
{
    reg_write(dma_regs_virt, dmacr_offset, DMACR_RESET);
    int poll_guard = 1000000;
    while ((reg_read(dma_regs_virt, dmacr_offset) & DMACR_RESET) && poll_guard--) {}
    if (poll_guard <= 0) { fprintf(stderr, "dma_reset: timeout\n"); exit(1); }
}

static void dma_wait_done(volatile uint32_t *dma_regs_virt, unsigned dmasr_offset, const char *channel_name)
{
    double wait_start_sec = now_sec();
    for (;;) {
        uint32_t dma_status = reg_read(dma_regs_virt, dmasr_offset);
        if (dma_status & DMASR_ERR_MASK) {
            fprintf(stderr, "%s: errore DMA, DMASR=0x%08x\n", channel_name, dma_status);
            exit(1);
        }
        if (dma_status & (DMASR_IDLE | DMASR_IOC_IRQ)) {
            reg_write(dma_regs_virt, dmasr_offset, dma_status & DMASR_IRQ_MASK);
            return;
        }
        if (now_sec() - wait_start_sec > DMA_TIMEOUT_SEC) {
            fprintf(stderr, "%s: timeout dopo %.1f s, DMASR=0x%08x "
                    "(core bloccato? bitstream caricato? PL0 attivo?)\n",
                    channel_name, DMA_TIMEOUT_SEC, dma_status);
            exit(1);
        }
    }
}

static void dma_mm2s_start(volatile uint32_t *dma_regs_virt, uint64_t source_phys, uint32_t length_bytes)
{
    reg_write(dma_regs_virt, REG_MM2S_DMACR, DMACR_RS);
    reg_write(dma_regs_virt, REG_MM2S_SA, (uint32_t)source_phys);
    reg_write(dma_regs_virt, REG_MM2S_SA_MSB, (uint32_t)(source_phys >> 32));
    reg_write(dma_regs_virt, REG_MM2S_LENGTH, length_bytes);
}

/* la S2MM va armata PRIMA di mandare l'esempio: il decoder non ha backpressure */
static void dma_s2mm_start(volatile uint32_t *dma_regs_virt, uint64_t destination_phys, uint32_t length_bytes)
{
    reg_write(dma_regs_virt, REG_S2MM_DMACR, DMACR_RS);
    reg_write(dma_regs_virt, REG_S2MM_DA, (uint32_t)destination_phys);
    reg_write(dma_regs_virt, REG_S2MM_DA_MSB, (uint32_t)(destination_phys >> 32));
    reg_write(dma_regs_virt, REG_S2MM_LENGTH, length_bytes);
}

static double pll_output_mhz(volatile uint32_t *crl_apb_regs_virt, unsigned ctrl_offset, unsigned frac_offset)
{
    uint32_t pll_ctrl     = crl_apb_regs_virt[ctrl_offset / 4];
    uint32_t pll_frac_cfg = crl_apb_regs_virt[frac_offset / 4];
    unsigned feedback_div = (pll_ctrl >> 8) & 0x7Fu;
    unsigned output_div2  = (pll_ctrl >> 16) & 0x1u;
    unsigned pll_bypass   = (pll_ctrl >> 3) & 0x1u;
    double   frac_part    = ((pll_frac_cfg >> 31) & 1u) ? (pll_frac_cfg & 0xFFFFu) / 65536.0 : 0.0;
    if (pll_bypass) return PSS_REF_CLK_MHZ;
    return PSS_REF_CLK_MHZ * (feedback_div + frac_part) / (output_div2 ? 2.0 : 1.0);
}

static double read_pl0_clock_mhz(int devmem_fd, const char **source_name)
{
    volatile uint32_t *crl_apb_regs_virt =
        (volatile uint32_t *)map_physical(devmem_fd, CRL_APB_BASE_PHYS, CRL_APB_SPAN, "CRL_APB");
    uint32_t pl0_ref_ctrl  = crl_apb_regs_virt[REG_PL0_REF_CTRL / 4];
    unsigned source_select = pl0_ref_ctrl & 0x7u;
    unsigned divisor0      = (pl0_ref_ctrl >> 8) & 0x3Fu;
    unsigned divisor1      = (pl0_ref_ctrl >> 16) & 0x3Fu;
    if (divisor0 == 0) divisor0 = 1;
    if (divisor1 == 0) divisor1 = 1;
    double source_mhz = -1.0;
    if (source_select == 0) {
        *source_name = "IOPLL";
        source_mhz = pll_output_mhz(crl_apb_regs_virt, REG_IOPLL_CTRL, REG_IOPLL_FRAC_CFG);
    } else if (source_select == 2) {
        *source_name = "RPLL";
        source_mhz = pll_output_mhz(crl_apb_regs_virt, REG_RPLL_CTRL, REG_RPLL_FRAC_CFG);
    } else {
        *source_name = "DPLL_TO_LPD (non decodificata)";
    }
    munmap((void *)crl_apb_regs_virt, CRL_APB_SPAN);
    return (source_mhz < 0) ? -1.0 : source_mhz / divisor0 / divisor1;
}

static void read_exact(const char *file_path, FILE *file_handle, void *destination, size_t n_bytes)
{
    if (fread(destination, 1, n_bytes, file_handle) != n_bytes) {
        fprintf(stderr, "%s troppo corto (attesi altri %zu byte)\n", file_path, n_bytes);
        exit(1);
    }
}

static void read_whole_file(const char *file_path, void *destination, size_t n_bytes)
{
    FILE *file_handle = fopen(file_path, "rb");
    if (!file_handle) { fprintf(stderr, "apri %s: %s\n", file_path, strerror(errno)); exit(1); }
    read_exact(file_path, file_handle, destination, n_bytes);
    fclose(file_handle);
}

/* copia verso la DDR riservata a parole da 64 bit allineate (memoria Device su arm64) */
static void copy_to_ddr64(volatile uint64_t *ddr_destination_virt, const uint64_t *source_buffer, size_t n_bytes)
{
    for (size_t word_index = 0; word_index < n_bytes / 8; word_index++)
        ddr_destination_virt[word_index] = source_buffer[word_index];
}

static void time_stats(const double *time_values, int n_values, double *min_out, double *mean_out, double *max_out)
{
    double min_value = time_values[0], max_value = time_values[0], sum_values = 0.0;
    for (int value_index = 0; value_index < n_values; value_index++) {
        if (time_values[value_index] < min_value) min_value = time_values[value_index];
        if (time_values[value_index] > max_value) max_value = time_values[value_index];
        sum_values += time_values[value_index];
    }
    *min_out = min_value; *mean_out = sum_values / n_values; *max_out = max_value;
}

/* --------------------- sensori di potenza (INA226) ---------------------- */
/* Il driver ina2xx espone power1_input in microwatt per ogni rail. Un thread
 * campiona tutti i rail ogni POWER_SAMPLE_US e accumula la media. */

typedef struct {
    char   name[320];
    char   power_path[300];
    double sum_uw;
    long   n_samples;
} power_rail_t;

static power_rail_t power_rails[MAX_RAILS];
static int n_power_rails = 0;
static volatile int sampler_running = 0;
static volatile int sampler_accumulate = 0;

static void find_power_rails(void)
{
    /* tutti i canali power<i>_input dei dispositivi hwmon (ina226, irps5401, ...), esclusi gli ams */
    DIR *hwmon_dir = opendir(HWMON_DIR);
    if (!hwmon_dir) return;
    struct dirent *entry;
    while ((entry = readdir(hwmon_dir))) {
        if (entry->d_name[0] == '.') continue;
        char name_path[300], device_name[64] = "";
        snprintf(name_path, sizeof(name_path), HWMON_DIR "/%s/name", entry->d_name);
        FILE *name_file = fopen(name_path, "r");
        if (!name_file) continue;
        if (!fgets(device_name, sizeof(device_name), name_file)) device_name[0] = 0;
        fclose(name_file);
        device_name[strcspn(device_name, "\n")] = 0;
        if (strcmp(device_name, "ams") == 0) continue;
        for (int channel = 1; channel <= 8 && n_power_rails < MAX_RAILS; channel++) {
            power_rail_t *rail = &power_rails[n_power_rails];
            snprintf(rail->power_path, sizeof(rail->power_path), HWMON_DIR "/%s/power%d_input", entry->d_name, channel);
            if (access(rail->power_path, R_OK) != 0) continue;
            char label_path[300], channel_label[64] = "";
            snprintf(label_path, sizeof(label_path), HWMON_DIR "/%s/power%d_label", entry->d_name, channel);
            FILE *label_file = fopen(label_path, "r");
            if (label_file) {
                if (!fgets(channel_label, sizeof(channel_label), label_file)) channel_label[0] = 0;
                fclose(label_file);
                channel_label[strcspn(channel_label, "\n")] = 0;
            }
            if (!channel_label[0]) snprintf(channel_label, sizeof(channel_label), "power%d", channel);
            snprintf(rail->name, sizeof(rail->name), "%s/%s %s", entry->d_name, device_name, channel_label);
            n_power_rails++;
        }
    }
    closedir(hwmon_dir);
}

static double read_rail_uw(const power_rail_t *rail)
{
    FILE *power_file = fopen(rail->power_path, "r");
    if (!power_file) return -1.0;
    long power_uw = -1;
    if (fscanf(power_file, "%ld", &power_uw) != 1) power_uw = -1;
    fclose(power_file);
    return (double)power_uw;
}

static void *power_sampler(void *unused)
{
    (void)unused;
    while (sampler_running) {
        if (sampler_accumulate) {
            for (int rail_index = 0; rail_index < n_power_rails; rail_index++) {
                double value_uw = read_rail_uw(&power_rails[rail_index]);
                if (value_uw >= 0) { power_rails[rail_index].sum_uw += value_uw; power_rails[rail_index].n_samples++; }
            }
        }
        usleep(POWER_SAMPLE_US);
    }
    return NULL;
}

static void reset_power_sums(void)
{
    for (int rail_index = 0; rail_index < n_power_rails; rail_index++) {
        power_rails[rail_index].sum_uw = 0.0;
        power_rails[rail_index].n_samples = 0;
    }
}

/* copia le medie (W) per rail e ritorna la somma su tutti i rail */
static double take_power_means(double *mean_w)  /* ritorna la somma, non usata nella stampa */
{
    double total_w = 0.0;
    for (int rail_index = 0; rail_index < n_power_rails; rail_index++) {
        mean_w[rail_index] = power_rails[rail_index].n_samples
            ? power_rails[rail_index].sum_uw / power_rails[rail_index].n_samples / 1e6 : 0.0;
        total_w += mean_w[rail_index];
    }
    return total_w;
}

/* ------------------------------ metriche -------------------------------- */

static int32_t board_logits[N_RECORDS_MAX][K_CLASSES];
static int32_t golden_logits[N_RECORDS_MAX][K_CLASSES];
static uint8_t true_label[N_RECORDS_MAX];
static double  inference_time_sec[N_RECORDS_MAX];


__attribute__((unused)) static int predicted_class(const int32_t *record_logits)
{
    int best_class = 0;
    for (int class_index = 1; class_index < K_CLASSES; class_index++)
        if (record_logits[class_index] > record_logits[best_class]) best_class = class_index;
    return best_class;
}

static void print_logits(const char *row_label, int record_index, const int32_t *record_logits)
{
    printf("  esempio %d  %s: [", record_index, row_label);
    for (int class_index = 0; class_index < K_CLASSES; class_index++)
        printf("%s%d", class_index ? ", " : "", record_logits[class_index]);
    printf("]\n");
}


/* --------------------------------- main --------------------------------- */

int main(int argc, char **argv)
{
    int n_records = N_RECORDS_MAX;
    uint64_t ddr_reserved_base_phys = DDR_RESERVED_BASE_PHYS;
    double energy_seconds = 10.0;
    if (argc > 1) {
        n_records = atoi(argv[1]);
        if (n_records < 1 || n_records > N_RECORDS_MAX) {
            fprintf(stderr, "N_ESEMPI deve essere fra 1 e %d\n", N_RECORDS_MAX);
            return 1;
        }
    }
    if (argc > 2) ddr_reserved_base_phys = strtoull(argv[2], NULL, 16);
    if (argc > 3) energy_seconds = atof(argv[3]);
    if (ddr_reserved_base_phys + DDR_RESERVED_SPAN > 0x80000000ULL) {
        fprintf(stderr, "la zona DDR deve stare sotto 2 GB (HPC0_DDR_LOW nel BD)\n");
        return 1;
    }
    const uint64_t ddr_input_record_phys  = ddr_reserved_base_phys + DDR_INPUT_RECORD_OFFSET;
    const uint64_t ddr_output_logits_phys = ddr_reserved_base_phys + DDR_OUTPUT_LOGITS_OFFSET;

    printf("=== %s ===\n", BUILD_TAG);
    printf("DMA @0x%08lx | esempio in DDR @0x%08lx (%d B) | logit in DDR @0x%08lx (%d B)\n",
           DMA_REGS_BASE_PHYS, (unsigned long)ddr_input_record_phys, INPUT_RECORD_BYTES,
           (unsigned long)ddr_output_logits_phys, OUTPUT_LOGITS_BYTES);

    int devmem_fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (devmem_fd < 0) { fprintf(stderr, "apri /dev/mem (serve root): %s\n", strerror(errno)); return 1; }

    const char *pl0_source_name = "";
    double pl0_clock_mhz = read_pl0_clock_mhz(devmem_fd, &pl0_source_name);
    if (pl0_clock_mhz > 0) printf("pl_clk0: %.4f MHz (sorgente %s, letto da PL0_REF_CTRL)\n", pl0_clock_mhz, pl0_source_name);
    else                   printf("pl_clk0: sorgente %s, frequenza non disponibile\n", pl0_source_name);

    volatile uint32_t *dma_regs_virt =
        (volatile uint32_t *)map_physical(devmem_fd, DMA_REGS_BASE_PHYS, DMA_REGS_SPAN, "registri DMA");
    uint8_t *ddr_reserved_virt =
        (uint8_t *)map_physical(devmem_fd, ddr_reserved_base_phys, DDR_RESERVED_SPAN, "DDR riservata");
    volatile uint64_t *ddr_input_record_virt  = (volatile uint64_t *)(ddr_reserved_virt + DDR_INPUT_RECORD_OFFSET);
    volatile uint32_t *ddr_output_logits_virt = (volatile uint32_t *)(ddr_reserved_virt + DDR_OUTPUT_LOGITS_OFFSET);

    /* ---- file d'ingresso ---- */
    read_whole_file(LABELS_BIN_PATH, true_label, (size_t)n_records);

    read_whole_file(GOLDEN_BIN_PATH, golden_logits, sizeof(int32_t) * K_CLASSES * (size_t)n_records);
    FILE *samples_file = fopen(SAMPLES_BIN_PATH, "rb");
    if (!samples_file) { fprintf(stderr, "apri %s: %s\n", SAMPLES_BIN_PATH, strerror(errno)); return 1; }
    static uint64_t host_record_buffer[INPUT_RECORD_BYTES / 8];
    static uint64_t first_record_buffer[INPUT_RECORD_BYTES / 8];

    /* ---- sensori di potenza ---- */
    find_power_rails();
    pthread_t sampler_thread;
    if (n_power_rails > 0) {
        sampler_running = 1;
        pthread_create(&sampler_thread, NULL, power_sampler, NULL);
    }
    printf("canali di potenza in " HWMON_DIR ": %d\n\n", n_power_rails);

    dma_reset(dma_regs_virt, REG_MM2S_DMACR);
    dma_reset(dma_regs_virt, REG_S2MM_DMACR);

    /* ======================= A. inferenza sul test set ======================= */
    printf("A. Inferenza su %d esempi...\n", n_records);
    double run_start_sec = now_sec();
    for (int record_index = 0; record_index < n_records; record_index++) {
        /* lettura dal file e copia in DDR: FUORI dalla misura del tempo */
        read_exact(SAMPLES_BIN_PATH, samples_file, host_record_buffer, INPUT_RECORD_BYTES);
        if (record_index == 0) memcpy(first_record_buffer, host_record_buffer, INPUT_RECORD_BYTES);
        copy_to_ddr64(ddr_input_record_virt, host_record_buffer, INPUT_RECORD_BYTES);
        for (int class_index = 0; class_index < K_CLASSES; class_index++)
            ddr_output_logits_virt[class_index] = 0xDEADBEEFu;          /* sentinella */

        dma_s2mm_start(dma_regs_virt, ddr_output_logits_phys, OUTPUT_LOGITS_BYTES);
        double inference_start_sec = now_sec();
        dma_mm2s_start(dma_regs_virt, ddr_input_record_phys, INPUT_RECORD_BYTES);
        dma_wait_done(dma_regs_virt, REG_MM2S_DMASR, "MM2S");
        dma_wait_done(dma_regs_virt, REG_S2MM_DMASR, "S2MM");
        inference_time_sec[record_index] = now_sec() - inference_start_sec;

        for (int class_index = 0; class_index < K_CLASSES; class_index++)
            board_logits[record_index][class_index] = (int32_t)ddr_output_logits_virt[class_index];

        if (record_index < 2) {
            print_logits("scheda", record_index, board_logits[record_index]);
            print_logits("golden", record_index, golden_logits[record_index]);
        }
        if ((record_index + 1) % 1000 == 0 || record_index + 1 == n_records) {
            printf("  %5d/%d\n", record_index + 1, n_records);
            fflush(stdout);
        }
    }
    fclose(samples_file);
    double run_total_sec = now_sec() - run_start_sec;

    FILE *board_logits_file = fopen(BOARD_LOGITS_BIN_PATH, "wb");
    if (board_logits_file) {
        fwrite(board_logits, sizeof(int32_t) * K_CLASSES, (size_t)n_records, board_logits_file);
        fclose(board_logits_file);
    } else {
        fprintf(stderr, "attenzione: non riesco a scrivere %s\n", BOARD_LOGITS_BIN_PATH);
    }

    /* ============================ 1. bit-exact =============================== */
    int n_records_mismatch = 0, n_logits_mismatch = 0, n_sentinel_left = 0;
    printf("\n=== 1. Logit scheda vs golden fixed-point (%d esempi) ===\n", n_records);
    for (int record_index = 0; record_index < n_records; record_index++) {
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
                printf("  DIVERSO esempio %d:\n", record_index);
                print_logits("scheda", record_index, board_logits[record_index]);
                print_logits("golden", record_index, golden_logits[record_index]);
            }
            n_records_mismatch++;
        }
    }
    printf("  esempi identici: %d/%d   logit diversi: %d/%d\n",
           n_records - n_records_mismatch, n_records, n_logits_mismatch, n_records * K_CLASSES);
    if (n_sentinel_left)
        printf("  ATTENZIONE: %d logit mai scritti dalla S2MM (sentinella 0xDEADBEEF)\n", n_sentinel_left);
    printf("  %s\n", n_records_mismatch == 0 ? "OK: BIT-EXACT con il golden su tutti gli esempi"
                                             : "NON bit-exact: vedi gli esempi sopra");

    /* ============================ 2. metrica ================================ */
    long n_correct = 0;
    for (int record_index = 0; record_index < n_records; record_index++)
        n_correct += (predicted_class(board_logits[record_index]) == true_label[record_index]);
    printf("\n=== 2. Accuracy top-1 (logit della scheda, %d registrazioni) ===\n", n_records);
    printf("  accuracy  %.4f  (%ld/%d)   atteso dal golden fixed-point: 0.9505 su 11005; float 0.9506\n",
           (double)n_correct / n_records, n_correct, n_records);

    /* ============================ 3. tempi =================================== */
    double infer_min, infer_mean, infer_max, infer_total = 0.0;
    time_stats(inference_time_sec, n_records, &infer_min, &infer_mean, &infer_max);
    for (int record_index = 0; record_index < n_records; record_index++) infer_total += inference_time_sec[record_index];
    printf("\n=== 3. Tempo di inferenza (CLOCK_MONOTONIC, MM2S avviata -> %d logit in DDR) ===\n", K_CLASSES);
    printf("  pl_clk0: ");
    if (pl0_clock_mhz > 0) printf("%.4f MHz (%s)\n", pl0_clock_mhz, pl0_source_name);
    else                   printf("non disponibile (%s)\n", pl0_source_name);
    printf("  tempo totale di inferenza (somma su %d esempi): %.3f s\n", n_records, infer_total);
    printf("  tempo medio per inferenza: %.3f ms   (min %.3f ms, max %.3f ms)\n",
           1e3 * infer_mean, 1e3 * infer_min, 1e3 * infer_max);
    if (pl0_clock_mhz > 0)
        printf("    = circa %.0f cicli (tempo medio x pl_clk0)\n", infer_mean * pl0_clock_mhz * 1e6);
    printf("  throughput (solo inferenza): %.1f inferenze/s\n", 1.0 / infer_mean);
    printf("  durata del passo A: %.2f s (lettura file e copie in DDR incluse)\n", run_total_sec);

    /* ============================ 4. energia ================================= */
    printf("\n=== 4. Potenza ed energia per inferenza (sensori hwmon della ZCU104) ===\n");
    if (n_power_rails == 0) {
        printf("  nessun canale power*_input in " HWMON_DIR ": energia non disponibile\n");
    } else {
        double idle_w[MAX_RAILS], active_w[MAX_RAILS];
        /* riposo: PL caricata, nessuna inferenza */
        reset_power_sums(); sampler_accumulate = 1;
        double idle_start_sec = now_sec();
        while (now_sec() - idle_start_sec < IDLE_SECONDS) usleep(10000);
        sampler_accumulate = 0;
        (void)take_power_means(idle_w);

        /* attivo: inferenze consecutive sul primo esempio, gia' in DDR, senza I/O di file */
        copy_to_ddr64(ddr_input_record_virt, first_record_buffer, INPUT_RECORD_BYTES);
        long n_energy_inferences = 0;
        reset_power_sums(); sampler_accumulate = 1;
        double active_start_sec = now_sec();
        while (now_sec() - active_start_sec < energy_seconds) {
            dma_s2mm_start(dma_regs_virt, ddr_output_logits_phys, OUTPUT_LOGITS_BYTES);
            dma_mm2s_start(dma_regs_virt, ddr_input_record_phys, INPUT_RECORD_BYTES);
            dma_wait_done(dma_regs_virt, REG_MM2S_DMASR, "MM2S");
            dma_wait_done(dma_regs_virt, REG_S2MM_DMASR, "S2MM");
            n_energy_inferences++;
        }
        double active_sec = now_sec() - active_start_sec;
        sampler_accumulate = 0;
        (void)take_power_means(active_w);
        double time_per_inference = active_sec / n_energy_inferences;

        printf("  riposo %.1f s, attivo %.1f s (%ld inferenze consecutive, %.3f ms ciascuna)\n",
               IDLE_SECONDS, active_sec, n_energy_inferences, 1e3 * time_per_inference);
        printf("  %-34s %10s %10s %10s %14s %14s\n", "canale", "riposo W", "attivo W", "delta W",
               "E/inf tot mJ", "E/inf delta mJ");
        for (int rail_index = 0; rail_index < n_power_rails; rail_index++)
            printf("  %-34s %10.3f %10.3f %10.3f %14.4f %14.4f\n", power_rails[rail_index].name,
                   idle_w[rail_index], active_w[rail_index], active_w[rail_index] - idle_w[rail_index],
                   1e3 * active_w[rail_index] * time_per_inference,
                   1e3 * (active_w[rail_index] - idle_w[rail_index]) * time_per_inference);
        printf("  E/inf tot = potenza media attiva x tempo per inferenza;"
               " E/inf delta = (attiva - riposo) x tempo per inferenza\n");
        printf("  ina226 power1 = ingresso 12 V dell'intera scheda; irps5401 pin/pout = ingresso/uscite dei regolatori.\n"
               "  I canali si sovrappongono: non vanno sommati.\n");
        sampler_running = 0;
        pthread_join(sampler_thread, NULL);
    }

    munmap(ddr_reserved_virt, DDR_RESERVED_SPAN);
    munmap((void *)dma_regs_virt, DMA_REGS_SPAN);
    close(devmem_fd);
    return 0;
}
