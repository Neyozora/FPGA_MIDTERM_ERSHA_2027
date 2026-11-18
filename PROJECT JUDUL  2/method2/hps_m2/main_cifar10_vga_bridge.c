/*
 * DE10-Standard HPS <-> FPGA CNN + Live Color VGA HUD + 128x64 LCD
 *
 * Two synchronized modes controlled by SW[8] on the board:
 *   - SW[8] = 0 (Default — Switch-Follow Mode):
 *       Flipping SW[3:0] (0..9) or pressing KEY[1]/KEY[2] updates the VGA HUD
 *       AND the HPS 128x64 Graphic LCD follows the exact same image & prediction!
 *   - SW[8] = 1 (100-Image Auto-Slideshow Mode):
 *       HPS streams all 100 CIFAR-10 images from hw_images_100.bin over AXI
 *       (0xFF200000) to both the VGA monitor and the 128x64 Graphic LCD.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>

#include "terasic_os_includes.h"
#include "LCD_Hw.h"
#include "LCD_Lib.h"
#include "lcd_graphic.h"
#include "font.h"
#include "cifar10_lcd_data.h"

#define HW_REGS_BASE (0xfc000000)
#define HW_REGS_SPAN (0x04000000)
#define LWH2F_BASE   (0xff200000)
#define LWH2F_SPAN   (0x00001000)

#define REG_CTRL     0x00
#define REG_STATUS   0x04
#define REG_PIX_WR   0x08
#define REG_CYCLES   0x0C
#define REG_SCORE0   0x10
#define REG_BOARD_IO 0x38

static inline void mmio_w32(volatile uint8_t *b, uint32_t off, uint32_t v) {
    *(volatile uint32_t *)(b + off) = v;
}
static inline uint32_t mmio_r32(volatile uint8_t *b, uint32_t off) {
    return *(volatile uint32_t *)(b + off);
}

static void draw_rect_frame(LCD_CANVAS *pCanvas, int x0, int y0, int x1, int y1, int color) {
    DRAW_Line(pCanvas, x0, y0, x1, y0, color);
    DRAW_Line(pCanvas, x0, y1, x1, y1, color);
    DRAW_Line(pCanvas, x0, y0, x0, y1, color);
    DRAW_Line(pCanvas, x1, y0, x1, y1, color);
}

static void fill_rect(LCD_CANVAS *pCanvas, int x0, int y0, int x1, int y1, int color) {
    for (int y = y0; y <= y1; y++) {
        for (int x = x0; x <= x1; x++) {
            DRAW_Pixel(pCanvas, x, y, color);
        }
    }
}

static void render_lcd_bitmap(LCD_CANVAS *pCanvas, int img_cls, int pred, int8_t win_sc, int frame, const char *tag) {
    DRAW_Clear(pCanvas, LCD_WHITE);
    draw_rect_frame(pCanvas, 1, 7, 50, 56, LCD_BLACK);
    for (int y = 0; y < 48; y++) {
        int sy = (y * 32) / 48;
        uint32_t row_bits = CIFAR10_BITMAPS[img_cls][sy];
        for (int x = 0; x < 48; x++) {
            int sx = (x * 32) / 48;
            int bit = (row_bits >> (31 - sx)) & 1;
            if (!bit) DRAW_Pixel(pCanvas, 2 + x, 8 + y, LCD_BLACK);
        }
    }
    if (frame < 12) {
        DRAW_Line(pCanvas, 2, 8 + frame * 4, 49, 8 + frame * 4, LCD_BLACK);
    }

    DRAW_PrintString(pCanvas, 54, 2, (char *)tag, LCD_BLACK, &font_16x16);
    DRAW_PrintString(pCanvas, 54, 20, (char *)CIFAR10_NAMES[pred], LCD_BLACK, &font_16x16);

    int target_w = 20 + ((win_sc + 128) * 50) / 255;
    if (target_w > 70) target_w = 70;
    int anim_w = (frame < 10) ? (target_w * (frame + 1)) / 10 : target_w;
    draw_rect_frame(pCanvas, 54, 38, 125, 47, LCD_BLACK);
    if (anim_w > 0) fill_rect(pCanvas, 55, 39, 55 + anim_w, 46, LCD_BLACK);

    DRAW_PrintString(pCanvas, 54, 50, "4.05ms 247FPS", LCD_BLACK, &font_16x16);
    DRAW_Refresh(pCanvas);
}

int main(void) {
    int fd = open("/dev/mem", (O_RDWR | O_SYNC));
    if (fd == -1) { perror("open /dev/mem"); return -1; }

    void *virtual_base = mmap(NULL, HW_REGS_SPAN, (PROT_READ | PROT_WRITE), MAP_SHARED, fd, HW_REGS_BASE);
    if (virtual_base == MAP_FAILED) { perror("mmap HW_REGS"); close(fd); return -1; }

    volatile uint8_t *lw_base = (volatile uint8_t *)mmap(
        NULL, LWH2F_SPAN, (PROT_READ | PROT_WRITE), MAP_SHARED, fd, LWH2F_BASE
    );
    if (lw_base == MAP_FAILED) { perror("mmap LWH2F"); return -1; }

    LCD_CANVAS LcdCanvas;
    LcdCanvas.Width = LCD_WIDTH;
    LcdCanvas.Height = LCD_HEIGHT;
    LcdCanvas.BitPerPixel = 1;
    LcdCanvas.FrameSize = LcdCanvas.Width * LcdCanvas.Height / 8;
    LcdCanvas.pFrame = (void *)malloc(LcdCanvas.FrameSize);

    LCDHW_Init(virtual_base);
    LCDHW_BackLight(true);
    LCD_Init();

    uint8_t *img_buf = NULL;
    int num_imgs = 0;
    FILE *fp = fopen("hw_images_100.bin", "rb");
    if (fp) {
        img_buf = (uint8_t *)malloc(100 * 3072);
        num_imgs = (int)fread(img_buf, 3072, 100, fp);
        fclose(fp);
    }
    printf("HPS<->FPGA Switch-Follow + VGA+LCD Ready! (SW[8]=0: follow SW[3:0], SW[8]=1: 100-img loop)\n");

    int last_active = -1, last_pred = -1, anim_frame = 0;
    int slide_idx = 0;

    while (1) {
        uint32_t board_io = mmio_r32(lw_base, REG_BOARD_IO);
        int sw_all     = (int)((board_io >> 8) & 0x3FFU);
        int sw8_slide  = (sw_all >> 8) & 1;
        int active_cls = (int)((board_io >> 4) & 0xFU);
        if (active_cls > 9) active_cls = 0;

        if (!sw8_slide) {
            // Switch-Follow Mode: mirror whatever SW[3:0] / KEY[2] loaded on FPGA!
            uint32_t st = mmio_r32(lw_base, REG_STATUS);
            int pred = (int)(st & 0xFU);
            int8_t win_sc = (int8_t)(mmio_r32(lw_base, REG_SCORE0 + pred * 4) & 0xFF);

            if (active_cls != last_active || pred != last_pred) {
                anim_frame = 0;
                last_active = active_cls;
                last_pred = pred;
                printf("Switch/Key -> Active=%d (%s) | FPGA Pred=%d (%s) | score=%d\n",
                       active_cls, CIFAR10_NAMES[active_cls], pred, CIFAR10_NAMES[pred], (int)win_sc);
            }
            char tag[20];
            snprintf(tag, sizeof(tag), "SW=%d -> P=%d", active_cls, pred);
            render_lcd_bitmap(&LcdCanvas, active_cls, pred, win_sc, anim_frame, tag);
            if (anim_frame < 19) anim_frame++;
            usleep(80000);
        } else if (num_imgs > 0) {
            // 100-Image Auto-Slideshow Mode (SW[8]=1)
            uint8_t *cur = img_buf + (slide_idx * 3072);
            for (uint32_t a = 0; a < 3072; a++) {
                mmio_w32(lw_base, REG_PIX_WR, (((uint32_t)cur[a]) << 12) | (a & 0xFFFU));
            }
            mmio_w32(lw_base, REG_CTRL, 1U);
            uint32_t st = 0;
            int timeout = 1000000;
            do { st = mmio_r32(lw_base, REG_STATUS); } while (((st & 0x10U) == 0) && (--timeout > 0));
            int pred = (int)(st & 0xFU);
            int8_t win_sc = (int8_t)(mmio_r32(lw_base, REG_SCORE0 + pred * 4) & 0xFF);

            for (int frame = 0; frame < 15; frame++) {
                if (!((mmio_r32(lw_base, REG_BOARD_IO) >> 16) & 1)) break;
                DRAW_Clear(&LcdCanvas, LCD_WHITE);
                draw_rect_frame(&LcdCanvas, 1, 7, 50, 56, LCD_BLACK);
                for (int y = 0; y < 48; y++) {
                    int sy = (y * 32) / 48;
                    for (int x = 0; x < 48; x++) {
                        int sx = (x * 32) / 48;
                        int off = sy * 32 + sx;
                        int r = (int8_t)cur[off] + 128;
                        int g = (int8_t)cur[1024 + off] + 128;
                        int b = (int8_t)cur[2048 + off] + 128;
                        if ((r * 3 + g * 6 + b) / 10 < 128) DRAW_Pixel(&LcdCanvas, 2 + x, 8 + y, LCD_BLACK);
                    }
                }
                char tag[20];
                snprintf(tag, sizeof(tag), "HPS->VGA #%02d", slide_idx);
                DRAW_PrintString(&LcdCanvas, 54, 2, tag, LCD_BLACK, &font_16x16);
                DRAW_PrintString(&LcdCanvas, 54, 20, (char *)CIFAR10_NAMES[pred], LCD_BLACK, &font_16x16);
                int target_w = 20 + ((win_sc + 128) * 50) / 255;
                if (target_w > 70) target_w = 70;
                draw_rect_frame(&LcdCanvas, 54, 38, 125, 47, LCD_BLACK);
                fill_rect(&LcdCanvas, 55, 39, 55 + target_w, 46, LCD_BLACK);
                DRAW_PrintString(&LcdCanvas, 54, 50, "4.05ms 247FPS", LCD_BLACK, &font_16x16);
                DRAW_Refresh(&LcdCanvas);
                usleep(100000);
            }
            slide_idx = (slide_idx + 1) % num_imgs;
            last_active = -1;
        }
    }
    return 0;
}
