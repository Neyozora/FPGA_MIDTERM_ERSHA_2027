/*
 * Starter HPS ARM Cortex-A9 -> FPGA CNN Accelerator Bridge (`cnn_top.v`)
 *
 * Works in TWO modes:
 *   1. Bare-Metal OpenOCD Mode (NO SD Card): compile with `arm-none-eabi-gcc`
 *      and load into 64 KB On-Chip RAM (0xFFFF0000) via OpenOCD.
 *   2. SD Card Linux Usermode: compile with `-DLINUX_USERMODE` on the board
 *      using `gcc` and run `./hps_cnn_bridge` (maps 0xFF200000 via /dev/mem).
 *
 * Expected Qsys / Platform Designer Register Map on Lightweight AXI (0xFF200000):
 *   Offset 0x00 (W)  : CTRL        bit[0] = start pulse, bit[1] = soft reset
 *   Offset 0x04 (R)  : STATUS      bit[0] = done, bits[4:1] = predicted_class (0..9)
 *   Offset 0x08 (W)  : PIX_WR      bits[9:0] = addr (0..1023), bits[17:10] = int8 pixel, bit[31] = wr_en
 *   Offset 0x10..0x34 (R): FC_SCORE[0..9] (10 x signed INT8 scores from fc_score_ram)
 */

#include <stdint.h>

#define LWHPS2FPGA_BASE  0xFF200000UL
#define CNN_ACCEL_OFFSET 0x00000000UL

#define REG_CTRL         0x00
#define REG_STATUS       0x04
#define REG_PIX_WR       0x08
#define REG_SCORE_BASE   0x10

#ifdef LINUX_USERMODE
#include <fcntl.h>
#include <stdio.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

static inline void mmio_write32(volatile uint8_t *base, uint32_t off, uint32_t val) {
    *(volatile uint32_t *)(base + off) = val;
}

static inline uint32_t mmio_read32(volatile uint8_t *base, uint32_t off) {
    return *(volatile uint32_t *)(base + off);
}

void run_cnn_on_fpga(volatile uint8_t *cnn_base, const int8_t img_32x32[1024],
                     int *out_class, int8_t out_scores[10]) {
    /* 1. Stream 1,024 signed INT8 pixels into `cnn_top` img_ram */
    for (uint32_t addr = 0; addr < 1024; addr++) {
        uint32_t pix_u8 = (uint8_t)img_32x32[addr];
        uint32_t cmd = (1U << 31) | (pix_u8 << 10) | (addr & 0x3FFU);
        mmio_write32(cnn_base, REG_PIX_WR, cmd);
    }

    /* 2. Pulse `start = 1` */
    mmio_write32(cnn_base, REG_CTRL, 0x00000001U);
    mmio_write32(cnn_base, REG_CTRL, 0x00000000U);

    /* 3. Poll `done == 1` (~6.06 ms for Method 1, ~0.28 ms for Method 2) */
    uint32_t status = 0;
    do {
        status = mmio_read32(cnn_base, REG_STATUS);
    } while ((status & 0x1U) == 0);

    *out_class = (int)((status >> 1) & 0xFU);

    /* 4. Read all 10 hardware INT8 class scores */
    for (int i = 0; i < 10; i++) {
        out_scores[i] = (int8_t)(mmio_read32(cnn_base, REG_SCORE_BASE + (i * 4)) & 0xFF);
    }
}

int main(void) {
#ifdef LINUX_USERMODE
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) {
        perror("open /dev/mem");
        return 1;
    }
    void *lw_map = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, LWHPS2FPGA_BASE);
    if (lw_map == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return 1;
    }
    volatile uint8_t *cnn_base = (volatile uint8_t *)lw_map + CNN_ACCEL_OFFSET;
#else
    /* Bare-metal OpenOCD: direct physical MMIO access */
    volatile uint8_t *cnn_base = (volatile uint8_t *)(LWHPS2FPGA_BASE + CNN_ACCEL_OFFSET);
#endif

    int8_t sample_img[1024] = {0};
    int pred_class = 0;
    int8_t scores[10] = {0};

    run_cnn_on_fpga(cnn_base, sample_img, &pred_class, scores);

#ifdef LINUX_USERMODE
    printf("FPGA Predicted Class: %d\n", pred_class);
    munmap(lw_map, 4096);
    close(fd);
#endif
    return 0;
}
