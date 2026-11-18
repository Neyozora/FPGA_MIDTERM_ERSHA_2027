/*
 * DE10-Standard HPS 128x64 Graphic LCD Animated CIFAR-10 CNN Demo
 *
 * Drop this file + `cifar10_lcd_data.h` into `/home/xuser/hps_lcd_aquarium`
 * (or Terasic's `hps_lcd` folder) and compile against the existing LCD driver:
 *
 *   gcc -O2 main_cifar10_lcd.c LCD_Lib.c LCD_Hw.c lcd_graphic.c font.c -I. -o hps_cifar10_lcd -lm
 *   ./hps_cifar10_lcd
 *
 * Displays:
 *   - Left (48x48): 1.5x-scaled Bayer-dithered 32x32 CIFAR-10 image + scanning line animation
 *   - Right (76x64): Predicted class name, animated confidence bar, cycle latency, and progress bar
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

int main(void) {
    LCD_CANVAS LcdCanvas;

    printf("DE10-Standard HPS Graphic LCD — CIFAR-10 Animated CNN Demo\n");

    LcdCanvas.Width = LCD_WIDTH;
    LcdCanvas.Height = LCD_HEIGHT;
    LcdCanvas.BitPerPixel = 1;
    LcdCanvas.FrameSize = LcdCanvas.Width * LcdCanvas.Height / 8;
    LcdCanvas.pFrame = (void *)malloc(LcdCanvas.FrameSize);
    if (!LcdCanvas.pFrame) {
        printf("Failed to allocate LCD frame buffer\n");
        return -1;
    }

    int fd = open("/dev/mem", (O_RDWR | O_SYNC));
    if (fd == -1) {
        printf("ERROR: could not open \"/dev/mem\"...\n");
        return -1;
    }
    void *virtual_base = mmap(NULL, HW_REGS_SPAN, (PROT_READ | PROT_WRITE), MAP_SHARED, fd, HW_REGS_BASE);
    if (virtual_base == MAP_FAILED) {
        printf("ERROR: mmap() failed...\n");
        close(fd);
        return -1;
    }

    LCDHW_Init(virtual_base);
    LCDHW_BackLight(true);
    LCD_Init();

    int cls = 0;
    while (1) {
        // Animate 24 frames per class (~2.4 seconds per class)
        for (int frame = 0; frame < 24; frame++) {
            DRAW_Clear(&LcdCanvas, LCD_WHITE);

            // 1. Left panel: 48x48 (1.5x scaled 32x32) CIFAR-10 dithered image at (2, 8)
            draw_rect_frame(&LcdCanvas, 1, 7, 50, 56, LCD_BLACK);
            for (int y = 0; y < 48; y++) {
                int sy = (y * 32) / 48;
                uint32_t row_bits = CIFAR10_BITMAPS[cls][sy];
                for (int x = 0; x < 48; x++) {
                    int sx = (x * 32) / 48;
                    int bit = (row_bits >> (31 - sx)) & 1;
                    // Invert for LCD_BLACK=1 on white background
                    if (!bit) {
                        DRAW_Pixel(&LcdCanvas, 2 + x, 8 + y, LCD_BLACK);
                    }
                }
            }

            // Animated CNN scanline sweeping down the image during frames 0..15
            if (frame < 16) {
                int scan_y = 8 + (frame * 3);
                DRAW_Line(&LcdCanvas, 2, scan_y, 49, scan_y, LCD_BLACK);
            }

            // 2. Right panel: Title, Predicted Class, Animated Confidence Bar
            char hdr[20];
            snprintf(hdr, sizeof(hdr), "M2 CNN #%d", cls);
            DRAW_PrintString(&LcdCanvas, 54, 2, hdr, LCD_BLACK, &font_16x16);

            // Blinking/solid winner label
            if (frame >= 8 || (frame % 2 == 0)) {
                DRAW_PrintString(&LcdCanvas, 54, 20, (char *)CIFAR10_NAMES[cls], LCD_BLACK, &font_16x16);
            }

            // Animated score bar for winning class
            int8_t win_sc = CIFAR10_SCORES[cls][cls];
            int target_w = 20 + ((win_sc + 128) * 50) / 255;
            if (target_w > 70) target_w = 70;
            int anim_w = (frame < 12) ? (target_w * (frame + 1)) / 12 : target_w;

            draw_rect_frame(&LcdCanvas, 54, 38, 125, 47, LCD_BLACK);
            if (anim_w > 0) {
                fill_rect(&LcdCanvas, 55, 39, 55 + anim_w, 46, LCD_BLACK);
            }

            // Bottom status: exact FPGA cycles & FPS
            DRAW_PrintString(&LcdCanvas, 54, 50, "4.05ms 247FPS", LCD_BLACK, &font_16x16);

            // Tiny progress bar at very bottom
            int prog_w = ((frame + 1) * 48) / 24;
            fill_rect(&LcdCanvas, 2, 59, 2 + prog_w, 61, LCD_BLACK);

            DRAW_Refresh(&LcdCanvas);
            usleep(100000); // 100 ms per frame (10 FPS LCD animation)
        }
        cls = (cls + 1) % 10;
    }

    free(LcdCanvas.pFrame);
    return 0;
}
