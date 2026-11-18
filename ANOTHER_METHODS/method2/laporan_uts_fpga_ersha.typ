#set page(
  paper: "a4",
  margin: (x: 22mm, y: 22mm),
  numbering: "1 / 1",
)
#set text(font: "New Computer Modern", size: 10.5pt, lang: "id")
#set heading(numbering: "1.1")
#set par(justify: true, leading: 0.65em)

// =============================================================================
// HALAMAN JUDUL / COVER
// =============================================================================
#align(center)[
  #v(12mm)
  #text(size: 13pt, weight: "bold", fill: luma(70))[
    LAPORAN PROYEK UJIAN TENGAH SEMESTER (UTS)
  ]
  #v(6mm)
  #text(size: 20pt, weight: "bold")[
    Perancangan dan Implementasi Akselerator CNN Paralel INT8 4-Lapis Berbasis SoC FPGA (Cyclone V) dengan Integrasi HPS, VGA RGB, dan LCD Grafis
  ]
  #v(8mm)
  #line(length: 80%, stroke: 0.8pt + luma(120))
  #v(8mm)
  #text(size: 12pt)[
    *Disusun Oleh:* \
    #v(4pt)
    *Nama:* Ersha Anandita Larasati \
    *NIM:* `101022300072`
  ]
  #v(10mm)
  #box(
    fill: luma(245),
    stroke: 0.5pt + luma(180),
    radius: 4pt,
    inset: 12pt,
    width: 85%,
  )[
    #align(left)[
      *Ringkasan Spesifikasi Sistem Terverifikasi:* \
      • *Board Target:* Terasic DE10-Standard (Intel Cyclone V SX SoC `5CSXFC6D6F31C6`) \
      • *Dataset:* Fashion-MNIST (Method 1) & CIFAR-10 `32×32×3` RGB (Method 2) \
      • *Arsitektur Method 2:* CNN 4-Lapis (`61.162` parameter), *2-Line Buffer* + *36-MAC Paralel* (`Po = 4`) \
      • *Akurasi Pure INT8:* `83,54%` (`10.000` citra uji CIFAR-10) & `84,00%` (`100` vektor uji hardware) \
      • *Kecepatan Hardware:* `202.538` siklus clock = `4,051 ms` @ `50 MHz` (*`246,9 FPS`*) \
      • *Verifikasi ModelSim 18.1:* `100 / 100` prediksi & `1.000 / 1.000` skor INT8 (*100% bit-exact*)
    ]
  ]
  #v(12mm)
  #text(size: 10.5pt, fill: luma(90))[
    Target Kompetisi: 30th LSI Design Contest in Hokkaido (2027) — Class 1: Image Classification
  ]
]

#pagebreak()

// =============================================================================
// BAB 1: PENDAHULUAN & PERBANDINGAN METHOD 1 VS METHOD 2
// =============================================================================
= Pendahuluan dan Ringkasan Sistem

Proyek ini mengimplementasikan sistem klasifikasi citra *Convolutional Neural Network (CNN)* secara penuh di atas perangkat keras *System-on-Chip (SoC) FPGA* *Terasic DE10-Standard* (`5CSXFC6D6F31C6`). Sistem dikembangkan dalam dua tahap utama:

1. *Method 1 (Baseline Midterm):* CNN 2-lapis sekuensial (1 MAC per siklus) menggunakan dataset *Fashion-MNIST* (`28×28` *grayscale* dipad menjadi `32×32×1`).
2. *Method 2 (Upgrade Paralel + SoC HPS):* CNN 4-lapis paralel (`4×9 = 36` MAC per siklus menggunakan *2-line shift-register line buffer*) dengan dataset citra warna alami *CIFAR-10* (`32×32×3` RGB), terintegrasi langsung dengan prosesor ARM Cortex-A9 (*Hard Processor System / HPS*) melalui jembatan *Lightweight AXI* (`0xFF200000`) untuk menggerakkan layar *VGA 640×480 RGB* dan *LCD Grafis HPS 128×64* secara sinkron.

#table(
  columns: (1.5fr, 1.4fr, 1.5fr),
  inset: 6pt,
  align: (left, center, center),
  stroke: 0.5pt + luma(160),
  [*Parameter*], [*Method 1 (Baseline)*], [*Method 2 (Paralel + HPS)*],
  [Dataset & Ukuran Input], [Fashion-MNIST (`32×32×1` Gray)], [*CIFAR-10 (`32×32×3` RGB)*],
  [Kuantisasi Input], [`q_in = pixel >> 1` (`[0, 127]`)], [*`q_in = pixel - 128` (`[-128, 127]`)*],
  [Topologi CNN], [2-Lapis (`1→8→P→8→16→GAP→FC`)], [*4-Lapis (`3→16→P→32→P→64→64→GAP→FC`)*],
  [Total Parameter], [`1.418`], [*`61.162` (`43,1×` lebih besar)*],
  [Metode Kuantisasi], [Per-Layer INT8 QAT], [*BN-Folding + Per-Channel INT8 QAT*],
  [Akurasi Uji INT8], [`85,87%` (Fashion-MNIST)], [*`83,54%` (CIFAR-10, turun `<0,1%`)*],
  [Arsitektur Konvolusi], [Sekuensial 1-MAC/siklus], [*2-Line Buffer + `4×9` MAC (`Po=4`)*],
  [Siklus Clock / Citra], [`302.978` siklus], [*`202.538` siklus (`1,50×` lebih cepat)*],
  [Latensi & Throughput], [`6,060 ms` (`165,0 FPS`)], [*`4,051 ms` (`246,9 FPS`)*],
  [Utilisasi ALM Cyclone V], [`804 / 41.910` (`2%`)], [`3.815` (`9%` FPGA) / `8.149` (`19%` + HPS)],
  [Blok Memori M10K], [`31 / 553` (`6%`)], [`102` (`18%`) / `103` (`19%` + HPS)],
  [Blok DSP], [`22 / 112` (`20%`)], [*`52 / 112` (`46%`)*],
  [Tampilan Output], [VGA Grayscale + PS/2 Mouse], [*VGA 24-bit RGB + LCD HPS 128×64*],
)

// =============================================================================
// BAB 2: STEP-BY-STEP SETUP DAN DOKUMENTASI HINGGA BERHASIL
// =============================================================================
= Dokumentasi Step-by-Step Setup Hingga Berjalan

Berikut adalah urutan langkah pengerjaan dari awal (*dataset*) hingga sistem berjalan secara *real-time* di papan fisik DE10-Standard.

== Langkah 1: Pengunduhan dan Pra-pemrosesan Dataset CIFAR-10
Skrip `sw_m2/preprocess.py` mengunduh dataset CIFAR-10 (`32×32×3` RGB) dan membentuk pembagian data deterministik (`SEED = 42`):
- *Data Latih:* `45.000` citra (`4.500` per kelas)
- *Data Validasi:* `5.000` citra (`500` per kelas)
- *Data Uji Resmi:* `10.000` citra (`1.000` per kelas)
- *Vektor Uji Hardware:* `100` citra seimbang (`10` citra pertama tiap kelas dari set uji)
```powershell
python -m sw_m2.preprocess
```

== Langkah 2: Pelatihan Model, BatchNorm Folding, dan Per-Channel INT8 QAT
Skrip `sw_m2/train_qat.py` melatih model di GPU (`CUDA`) dalam tiga tahap:
1. *Tahap 1 (Float32 + BatchNorm2d, 80 epoch):* Menggunakan augmentasi `RandomCrop(32, padding=4)` dan `RandomHorizontalFlip(p=0.5)`, optimizer `SGD(lr=0.05, momentum=0.9)`, dan `CosineAnnealingLR`.
2. *Tahap 2 (BatchNorm Folding + QAT, 25 epoch):* Menggabungkan parameter `BatchNorm` ke bobot dan bias konvolusi, lalu melakukan *fine-tuning* QAT (*Straight-Through Estimator*) dengan `Adam(lr=2e-4)`.
3. *Tahap 3 (Konversi Integer Murni):* Mengekstrak bobot `INT8`, bias `INT32`, serta pengali diadik per-kanal `(mult, shift)`.
```powershell
python -u -m sw_m2.train_qat
```

== Langkah 3: Ekspor Memori Hardware (`.mem`) dan Payload HPS (`.bin`)
Skrip `sw_m2/export_mem.py` mengonversi seluruh bobot ke format `$readmemh` (`mem_m2/w_bank0..3.mem` berukuran `1676 × 72-bit`, `q_bank0..3.mem` berukuran `44 × 53-bit`, `fc_weights_m2.mem`, `fc_qparams_m2.mem`, serta `de10_images_10_m2.mem` berisi 10 citra terverifikasi untuk `SW[3:0]`). Selain itu, disiapkan payload biner `hps_m2/hw_images_100.bin` (`307.200` byte) dan header bitmap LCD `hps_m2/cifar10_lcd_data.h`.
```powershell
python -m sw_m2.export_mem
python -m sw_m2.gen_hps_lcd_header
```

== Langkah 4: Verifikasi Bit-Exact pada ModelSim 18.1
Seluruh modul RTL Verilog-2001 (`rtl_m2/*.v`) dikompilasi dan diuji terhadap `100` citra CIFAR-10 menggunakan `sim_m2/tb_cnn_top_m2.v`:
```powershell
C:\intelFPGA_lite\18.1\modelsim_ase\win32aloem\vlib.exe work_m2
C:\intelFPGA_lite\18.1\modelsim_ase\win32aloem\vlog.exe -work work_m2 rtl_m2/*.v sim_m2/tb_cnn_top_m2.v
C:\intelFPGA_lite\18.1\modelsim_ase\win32aloem\vsim.exe -c work_m2.tb_cnn_top_m2 -do "run -all; quit -f"
```
Hasilnya menunjukkan *`100 / 100` kesesuaian prediksi*, *`1.000 / 1.000` kesesuaian skor INT8 bit-exact*, dan waktu eksekusi tepat *`202.538` siklus clock* per citra.

== Langkah 5: Pembuatan Sistem Qsys HPS, Sintesis Quartus, dan Upload Bitstream
1. Menjalankan skrip Qsys `quartus_m2/build_hps_qsys.tcl` untuk membentuk `hps_cnn_qsys` yang mengaktifkan *Lightweight HPS-to-FPGA AXI Bridge* (`0xFF200000`).
2. Menjalankan kompilasi Quartus Prime 18.1 (`quartus_map`, `hps_sdram_p0_pin_assignments.tcl`, `quartus_fit`, `quartus_asm`, `quartus_sta`).
3. Memprogram file bitstream `quartus_m2/output_files/de10_standard_m2.sof` ke FPGA melalui USB-Blaster II (`DE-SoC [USB-1]`):
```powershell
C:\intelFPGA_lite\18.1\quartus\bin64\quartus_pgm.exe -c "DE-SoC [USB-1]" -m JTAG -o "p;quartus_m2\output_files\de10_standard_m2.sof@2"
```

== Langkah 6: Kompilasi dan Eksekusi Program C pada Linux HPS (ARM Cortex-A9)
Folder demo LCD asli `/home/xuser/hps_lcd_aquarium` diduplikasi menjadi `/home/xuser/hps_cifar10_lcd` agar folder asli tetap aman. Program `main_cifar10_vga_bridge.c` dikompilasi langsung di papan DE10-Standard (Angstrom Linux) dan dijalankan:
```bash
cd /home/xuser/hps_cifar10_lcd
gcc -O2 -std=gnu99 -Wall -Dsoc_cv_av \
  -Iintel-socfpga-hwlib/armv7a/hwlib/include/soc_cv_av \
  -Iintel-socfpga-hwlib/armv7a/hwlib/include -I. \
  main_cifar10_vga_bridge.c terasic_lib.c LCD_Lib.c LCD_Driver.c LCD_Hw.c lcd_graphic.c font.c \
  -o hps_vga_bridge -lrt -lm
./hps_vga_bridge
```

== Langkah 7: Menjalankan GUI Python Terintegrasi ModelSim (Opsional di PC)
Untuk menguji foto custom (`.jpg`/`.png`) maupun 100 vektor CIFAR-10 secara interaktif dengan simulasi RTL langsung:
```powershell
python -m sw_m2.gui_app_m2
```

// =============================================================================
// BAB 3: ALGORITMA, MEKANISME KERJA, DAN ARSITEKTUR HARDWARE
// =============================================================================
= Algoritma, Mekanisme Kerja, dan Arsitektur Hardware

== Algoritma Kuantisasi Integer Murni (Zero Floating-Point di FPGA)
Agar FPGA dapat menghitung CNN tanpa unit *floating-point* yang mahal, seluruh operasi diubah menjadi aritmetika integer bertanda:
1. *Kuantisasi Input MSB-Flip (`0` Gerbang Logika):* Setiap piksel RGB `uint8` $p in [0, 255]$ diubah menjadi $q_"in" = p - 128 in [-128, +127]$ (skala $s_"in" = 1/128$). Di Verilog, operasi ini cukup membalik bit paling kiri (*MSB*): `{~pixel[7], pixel[6:0]}`.
2. *BatchNorm Folding:* Parameter `BatchNorm` ($gamma, beta, mu, sigma^2, epsilon$) digabung secara aljabar ke bobot dan bias sebelum kuantisasi:
   $ w_"fold",c = (gamma_c / sqrt(sigma_c^2 + epsilon)) w_c, quad b_"fold",c = beta_c - (gamma_c mu_c) / sqrt(sigma_c^2 + epsilon). $
3. *Rekuantisasi Diadik Per-Kanal:* Setiap kanal keluaran $c$ memiliki pengali integer $M_c in [1, 65535]$ dan pergeseran bit $n_c in [8, 24]$ sehingga pembagian skala $s_"in" dot s_(w,c) / s_"out"$ digantikan perkalian dan pergeseran bit (*round-to-nearest*):
   $ y_(q,c) = "clamp"_(0, 127) ( floor( ( ("acc"_c + b_(q,c)) dot M_c + 2^(n_c - 1) ) / 2^(n_c) ) ). $
4. *Zero-DSP Global Average Pooling (`8×8`):* Karena keluaran `Conv4` berukuran `8 × 8 = 64 = 2^6`, rata-rata global dihitung tanpa pengali DSP, cukup penjumlahan 16-bit dan geser kanan 6 bit: `(sum + 32) >>> 6`.

== Mekanisme Kerja Hardware (`conv_engine_m2.v` & Ping-Pong BRAM)
- *2-Line Shift-Register Line Buffer (`lb0`, `lb1`):* Piksel fitur dialirkan secara raster (`1` piksel/siklus). Dua penyangga baris menyimpan 2 baris sebelumnya sehingga seluruh *9 piksel jendela `3×3`* tersedia bersamaan pada setiap siklus clock dengan `PAD=1` otomatis di keempat tepi citra.
- *Array 36-MAC Paralel ($P_o = 4$):* Empat unit `9-MAC` bekerja paralel menghitung 4 kanal keluaran sekaligus per siklus clock, membaca `4 × 72-bit` bobot dari `w_bank0..3` dalam 1 siklus.
- *Memori Ping-Pong 4-Bank (`fmap_a0..3` & `fmap_b0..3`):* Memori fitur dibagi menjadi 4 bank paralel berdasarkan `oc % 4`, sehingga penulisan 4 kanal hasil konvolusi maupun pembacaan `MaxPool 2×2` terjadi tanpa antrean.

== Mekanisme Integrasi HPS, VGA, dan LCD (`0xFF200000`)
Prosesor ARM Cortex-A9 pada HPS memetakan jembatan *Lightweight AXI* (`0xFF200000`) via `/dev/mem` dan `mmap()` dengan peta register sebagai berikut:
- `0x00` (`REG_CTRL`): Menulis `1` memicu sinyal `start` pada `cnn_top_m2`.
- `0x04` (`REG_STATUS`): Membaca bit selesai (`done`) dan `predicted_class` (`0..9`).
- `0x08` (`REG_PIX_WR`): Mengirim piksel RGB `{data[7:0], addr[11:0]}` ke `img_ram` sekaligus ke memori VGA (`vga_r_ram`, `vga_g_ram`, `vga_b_ram`).
- `0x0C` (`REG_CYCLES`): Membaca jumlah siklus clock presisi (`202.538` siklus).
- `0x10..0x34` (`REG_SCORE0..9`): Membaca skor `INT8` untuk kesepuluh kelas.
- `0x38` (`REG_BOARD_IO`): Membaca posisi sakelar `SW[9:0]` dan tombol `KEY[3:0]` agar layar LCD HPS otomatis mengikuti perubahan sakelar di papan FPGA!

// =============================================================================
// BAB 4: ARTI INDIKATOR LAYAR HPS LCD, VGA HUD, LED, DAN 7-SEGMENT
// =============================================================================
= Penjelasan Arti Tampilan Layar HPS LCD, VGA HUD, LED, dan 7-Segment

== Arti Tampilan pada Layar LCD Grafis HPS (`128 × 64`)
1. *Gambar Kiri (`48×48` piksel):* Tampilan citra CIFAR-10 `32×32` yang diperbesar `1,5×` dan dikonversi ke monokrom 1-bit menggunakan *4×4 Bayer Ordered Dithering*. Garis horizontal yang bergerak turun adalah animasi *scanline* konvolusi.
2. *Judul Kanan Atas:*
   - `SW=X -> P=Y` *(Mode Switch-Follow, `SW[8]=0`)*: `X` adalah indeks citra ROM yang dipilih sakelar (`0..9`), dan `Y` adalah hasil prediksi langsung dari FPGA (`0..9`).
   - `HPS->VGA #XX` *(Mode Slideshow, `SW[8]=1`)*: `XX` adalah nomor urut citra (`00..99`) yang sedang dikirim ARM HPS ke FPGA dan VGA.
3. *Nama Kelas & Batang Skor:* Menampilkan nama kelas prediksi (`AIRPLANE` s.d. `TRUCK`) serta batang skor animasi yang lebarnya dihitung proporsional dari skor `INT8` kelas pemenang (`[-128, 127]`).
4. *Indikator `4.05ms 247FPS`:*
   - *`4.05ms` (Latensi Inferensi):* Waktu komputasi 1 citra oleh FPGA pada clock `50 MHz`:
     $ T_"infer" = (202.538 "siklus") / (50 times 10^6 "Hz") = 4,05076 "ms" approx 4,05 "ms". $
   - *`247FPS` (Throughput):* Jumlah citra yang dapat diklasifikasi per detik:
     $ "FPS" = (50 times 10^6 "Hz") / (202.538 "siklus") = 246,87 "FPS" approx 247 "FPS". $

== Arti Tampilan VGA HUD, LED, dan 7-Segment
- *VGA Monitor (`640×480` @ `60 Hz`):* Panel kiri menampilkan citra warna *24-bit RGB* (`256×256`, zoom `8×`); panel kanan menampilkan diagram batang horizontal 10 kelas (kelas pemenang berwarna hijau zamrud); panel bawah menampilkan `WINNER: <NAMA KELAS>`.
- *`LEDR[9:0]`:* Indikator *one-hot* kelas prediksi (`LEDR[k]` menyala jika prediksi = `k`).
- *`HEX5..HEX0`:* Saat `SW[9] = 0`, menampilkan indeks citra (`HEX5`) & kelas prediksi (`HEX0`) dalam format `X 0 0 0 0 Y`; saat `SW[9] = 1`, menampilkan jumlah siklus clock `03172A` heksadesimal (`202.538` desimal).

== Tabel Kombinasi Sakelar (`SW3 SW2 SW1 SW0`) untuk Method 2 (CIFAR-10)
Tabel berikut merangkum kombinasi sakelar geser `SW[3:0]` (`▼ 0` = Bawah/OFF, `▲ 1` = Atas/ON) pada DE10-Standard saat `SW[8] = 0` dan `SW[9] = 0`, beserta tampilan yang muncul serentak pada `HEX5..HEX0`, LCD HPS `128×64`, `LEDR[9:0]`, dan indeks citra di GUI Python (`sw_m2.gui_app_m2`):

#table(
  columns: (0.55fr, 0.5fr, 0.5fr, 0.5fr, 0.5fr, 1.55fr, 1.15fr, 1.25fr, 0.7fr, 0.9fr),
  inset: 5pt,
  align: (center, center, center, center, center, left, center, center, center, center),
  stroke: 0.6pt + black,
  fill: (col, row) => if row == 0 { rgb("#1e3a8a") } else { none },
  ..([*Pic \#*], [*SW3*], [*SW2*], [*SW1*], [*SW0*], [*Citra CIFAR-10 di FPGA*], [*Tampilan HEX5..0*], [*Layar LCD HPS*], [*LED*], [*GUI Vec \#*]).map(h => text(fill: white, weight: "bold", size: 8.5pt)[#h]),
  [0], [▼ 0], [▼ 0], [▼ 0], [▼ 0], [*Airplane* (Kelas 0)], [`0 0 0 0 0 0`], [`SW=0 -> P=0`], [`LEDR0`], [`Vec #1`],
  [1], [▼ 0], [▼ 0], [▼ 0], [▲ 1], [*Automobile* (Kelas 1)], [`1 0 0 0 0 1`], [`SW=1 -> P=1`], [`LEDR1`], [`Vec #10`],
  [2], [▼ 0], [▼ 0], [▲ 1], [▼ 0], [*Bird* (Kelas 2)], [`2 0 0 0 0 2`], [`SW=2 -> P=2`], [`LEDR2`], [`Vec #20`],
  [3], [▼ 0], [▼ 0], [▲ 1], [▲ 1], [*Cat* (Kelas 3)], [`3 0 0 0 0 3`], [`SW=3 -> P=3`], [`LEDR3`], [`Vec #30`],
  [4], [▼ 0], [▲ 1], [▼ 0], [▼ 0], [*Deer* (Kelas 4)], [`4 0 0 0 0 4`], [`SW=4 -> P=4`], [`LEDR4`], [`Vec #40`],
  [5], [▼ 0], [▲ 1], [▼ 0], [▲ 1], [*Dog* (Kelas 5)], [`5 0 0 0 0 5`], [`SW=5 -> P=5`], [`LEDR5`], [`Vec #50`],
  [6], [▼ 0], [▲ 1], [▲ 1], [▼ 0], [*Frog* (Kelas 6)], [`6 0 0 0 0 6`], [`SW=6 -> P=6`], [`LEDR6`], [`Vec #60`],
  [7], [▼ 0], [▲ 1], [▲ 1], [▲ 1], [*Horse* (Kelas 7)], [`7 0 0 0 0 7`], [`SW=7 -> P=7`], [`LEDR7`], [`Vec #70`],
  [8], [▲ 1], [▼ 0], [▼ 0], [▼ 0], [*Ship* (Kelas 8)], [`8 0 0 0 0 8`], [`SW=8 -> P=8`], [`LEDR8`], [`Vec #80`],
  [9], [▲ 1], [▼ 0], [▼ 0], [▲ 1], [*Truck* (Kelas 9)], [`9 0 0 0 0 9`], [`SW=9 -> P=9`], [`LEDR9`], [`Vec #90`],
)

// =============================================================================
// BAB 5: HASIL TRAINING, AKURASI, DAN VISUALISASI DATA
// =============================================================================
= Hasil Training, Kuantisasi, dan Visualisasi

== Tabel Ringkasan Hasil Training & Kuantisasi (Method 2 — CIFAR-10)
Model 4-lapis dilatih di GPU NVIDIA (`CUDA`) selama `80` epoch (Float32 + BatchNorm2d), dilanjutkan *BatchNorm Folding* dan *Per-Channel Symmetric INT8 QAT* selama `25` epoch, lalu dievaluasi secara *Pure Integer* (`INT8` + pengali diadik) yang identik bit-demi-bit dengan FPGA:

#align(center)[
  #text(weight: "bold", size: 11pt)[HASIL TRAINING (METHOD 2 — CIFAR-10)]
]
#v(4pt)
#table(
  columns: (1.5fr, 1.1fr, 1.1fr, 1.3fr, 1.1fr),
  inset: 6pt,
  align: (left, center, center, center, center),
  stroke: 0.6pt + black,
  fill: (col, row) => if row == 0 { rgb("#1e3a8a") } else { none },
  ..([*Pembagian Data*], [*FP32 Float+BN*], [*QAT Fake-Quant*], [*Pure INT8 (Terpilih)*], [*Selisih vs. FP32*]).map(h => text(fill: white, weight: "bold", size: 9pt)[#h]),
  [Train (`45.000`)], [—], [—], [*`87,93%`*], [—],
  [Validation (`5.000`)], [`84,18%`], [`84,34%`], [*`84,02%`*], [`-0,16%`],
  [Test (`10.000`)], [`83,49%`], [`83,61%`], [*`83,54%`*], [*`+0,05%`*],
  [HW Test Vectors (`100`)], [—], [—], [*`84,00%`*], [—],
)

== Visualisasi 100 Vektor Uji Hardware CIFAR-10 (`10 × 10` RGB)
#figure(
  image("reports_m2/hw_vectors_100_grid_m2.png", width: 88%),
  caption: [10 citra pertama per kelas dari set uji CIFAR-10 (`100` vektor uji hardware, `32×32×3` RGB dengan kuantisasi MSB-flip `[-128, 127]`).]
)

== Sampel 10 Citra ROM Board & Kurva Training Referensi
#figure(
  image("reports_m2/cifar10_samples_m2.png", width: 92%),
  caption: [10 citra uji CIFAR-10 (`32×32×3` RGB) yang tersimpan pada `board_img_rom` DE10-Standard beserta skor `INT8` masing-masing.]
)

#figure(
  image("reports/training/training_curves.png", width: 90%),
  caption: [Referensi kurva *Training & Validation Loss* serta *Accuracy* pada tahap pelatihan baseline Method 1 (Fashion-MNIST).]
)

== Confusion Matrix Pure INT8 & Distribusi Bobot (`Conv1..Conv4`, `FC`)
#figure(
  image("reports_m2/int8_confusion_matrix_m2.png", width: 76%),
  caption: [Confusion Matrix Pure INT8 Method 2 pada seluruh `10.000` citra uji resmi CIFAR-10 (Akurasi = `83,54%`).]
)

#figure(
  image("reports_m2/weight_distributions_m2.png", width: 98%),
  caption: [Perbandingan distribusi bobot *FP32 (BN-Folded)* vs. *Dequantized Per-Channel INT8* pada lapisan `Conv1`, `Conv2`, `Conv3`, `Conv4`, dan `FC`.]
)

== Perbandingan Performa Hardware (Method 1 vs. Method 2)
#figure(
  image("reports_m2/method1_vs_method2_chart.png", width: 92%),
  caption: [Perbandingan jumlah siklus clock dan *throughput* (FPS @ `50 MHz`) antara Method 1 (1-MAC) dan Method 2 (36-MAC).]
)

// =============================================================================
// BAB 6: HASIL OUTPUT VGA DAN LCD HPS (BESERTA AREA FOTO LANGSUNG)
// =============================================================================
= Hasil Output VGA Monitor dan LCD Grafis HPS

== Referensi Layout Simulasi VGA HUD
#figure(
  image("reports/simulation/modelsim_vga_frame.png", width: 85%),
  caption: [Referensi tata letak frame VGA `640×480` hasil render hardware `vga_cnn_hud` (pada Method 2, panel kiri menampilkan citra warna 24-bit RGB CIFAR-10 dan label 10 kelas CIFAR-10).]
)

== Dokumentasi Foto Output Fisik (VGA Monitor & LCD HPS DE10-Standard)
*Catatan:* Kotak di bawah ini disediakan untuk menempelkan foto langsung hasil kamera terhadap monitor VGA dan LCD HPS DE10-Standard (atau ganti blok `#rect` di file `.typ` dengan `#image("nama_foto.jpg", width: 85%)`).

#grid(
  columns: (1fr, 1fr),
  gutter: 12pt,
  [
    #rect(width: 100%, height: 68mm, stroke: (dash: "dashed", paint: luma(120)), radius: 4pt)[
      #align(center + horizon)[
        #text(fill: luma(100), size: 9.5pt)[
          *[AREA FOTO 1: OUTPUT VGA MONITOR]* \
          Tempelkan foto monitor VGA `640×480` \
          (Menampilkan citra RGB CIFAR-10, \
          diagram batang 10 kelas, dan `WINNER`)
        ]
      ]
    ]
  ],
  [
    #rect(width: 100%, height: 68mm, stroke: (dash: "dashed", paint: luma(120)), radius: 4pt)[
      #align(center + horizon)[
        #text(fill: luma(100), size: 9.5pt)[
          *[AREA FOTO 2: OUTPUT LCD HPS `128×64`]* \
          Tempelkan foto LCD DE10-Standard \
          (Menampilkan citra dither `48×48`, \
          `SW=X -> P=Y`, dan `4.05ms 247FPS`)
        ]
      ]
    ]
  ]
)

// =============================================================================
// BAB 7: KESIMPULAN
// =============================================================================
= Kesimpulan

Proyek akselerator CNN pada papan *Terasic DE10-Standard (`5CSXFC6D6F31C6`)* berhasil ditingkatkan dari *baseline* 2-lapis sekuensial (Method 1) menjadi *akselerator paralel 4-lapis INT8 (`36` MAC/siklus)* pada dataset *CIFAR-10 `32×32×3` RGB* (Method 2) yang terintegrasi dengan *ARM Cortex-A9 HPS* via *Lightweight AXI Bridge* (`0xFF200000`). Sistem mencapai akurasi *Pure INT8* *`83,54%`*, kesesuaian *bit-exact `1.000 / 1.000`* di ModelSim 18.1, kecepatan *`246,9 FPS` (`4,051 ms`)* pada `50 MHz`, lolos uji *timing* Quartus Prime 18.1 tanpa pelanggaran (`+3,634 ns` setup slack), serta menampilkan hasil klasifikasi secara sinkron di *VGA Monitor*, *LCD Grafis HPS `128×64`*, *LED*, dan *7-Segment*.
