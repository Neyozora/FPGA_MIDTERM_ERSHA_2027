// ============================================================================
// Laporan Proyek Tengah Semester (Midterm) — Bahasa Indonesia
// Nama: Ersha Anandita Larasati | NIM: 101022300072
// Compile: typst compile laporan_midterm_ersha.typ laporan_midterm_ersha.pdf
// ============================================================================

#set document(
  title: "Laporan Proyek Midterm: Implementasi Akselerator CNN Klasifikasi Citra Fashion-MNIST Berbasis INT8 pada FPGA Intel Cyclone V (Terasic DE10-Standard)",
  author: "Ersha Anandita Larasati",
)

#set page(
  paper: "a4",
  margin: (x: 2cm, y: 2.2cm),
  header: align(right)[
    #text(size: 8.5pt, fill: luma(110))[
      *Laporan Proyek Midterm* • Ersha Anandita Larasati (101022300072) • FPGA Cyclone V DE10-Standard
    ]
  ],
  footer: context [
    #line(length: 100%, stroke: 0.4pt + luma(180))
    #v(2pt)
    #grid(
      columns: (1fr, 1fr),
      align(left)[#text(size: 8.5pt, fill: luma(100))[Metode 1 (Baseline Sequential Single-MAC) • Target LSI Design Contest 2027]],
      align(right)[#text(size: 8.5pt, fill: luma(100))[Halaman #counter(page).display("1 dari 1", both: true)]],
    )
  ],
)

#set text(
  font: ("Calibri", "Arial", "Times New Roman"),
  size: 10.5pt,
  lang: "id",
)

#set par(justify: true, leading: 0.62em)
#set heading(numbering: "1.1.")

#show heading.where(level: 1): it => block(above: 1.35em, below: 0.7em)[
  #text(size: 13.8pt, weight: "bold", fill: rgb("#1e3a8a"))[#it]
  #v(-4pt)
  #line(length: 100%, stroke: 1.2pt + rgb("#1e3a8a"))
]

#show heading.where(level: 2): it => block(above: 1.05em, below: 0.5em)[
  #text(size: 11.4pt, weight: "bold", fill: rgb("#0f172a"))[#it]
]

#let kotak(judul, isi, color: rgb("#1d4ed8"), bg: rgb("#eff6ff")) = block(
  width: 100%,
  fill: bg,
  stroke: (left: 4pt + color, rest: 0.5pt + color.lighten(50%)),
  inset: 10pt,
  radius: (right: 4pt),
  above: 0.8em,
  below: 0.8em,
)[
  #text(weight: "bold", size: 10.8pt, fill: color)[#judul] \
  #isi
]

// ============================================================================
// HALAMAN JUDUL / HEADER IDENTITAS
// ============================================================================
#align(center)[
  #block(
    width: 100%,
    fill: rgb("#0f172a"),
    inset: (x: 18pt, y: 20pt),
    radius: 6pt,
  )[
    #text(size: 10pt, weight: "bold", fill: rgb("#38bdf8"))[
      LAPORAN PROYEK TENGAH SEMESTER (MIDTERM) • METODE 1 (BASELINE)
    ]
    #v(6pt)
    #text(size: 15.5pt, weight: "bold", fill: white)[
      Perancangan dan Implementasi Akselerator CNN Klasifikasi Citra Fashion-MNIST Berbasis INT8 pada FPGA Terasic DE10-Standard dengan Output VGA dan Antarmuka GUI Interaktif
    ]
    #v(10pt)
    #line(length: 60%, stroke: 0.6pt + rgb("#38bdf8"))
    #v(8pt)
    #text(size: 11pt, weight: "bold", fill: white)[
      Nama: Ersha Anandita Larasati \
      NIM: 101022300072
    ]
    #v(4pt)
    #text(size: 9.5pt, fill: rgb("#cbd5e1"))[
      Perangkat Target: Terasic DE10-Standard (Intel Cyclone V SX SoC `5CSXFC6D6F31C6`) \
      Dataset: Fashion-MNIST (`32 × 32`) • Tahun Akademik 2026/2027
    ]
  ]
]

#v(6pt)

#kotak(
  "Ringkasan Eksekutif Proyek (Metode 1)",
  [
    Proyek ini mengimplementasikan sistem klasifikasi citra *Convolutional Neural Network (CNN)* dua lapis secara *end-to-end* pada FPGA *Intel Cyclone V (`5CSXFC6D6F31C6`)* menggunakan aritmetika bilangan bulat bertanda (`INT8` bobot/aktivasi, `INT32` akumulator) dan `100%` memori internal *M10K BRAM* tanpa memori eksternal DDR.
    - *Akurasi Model:* `86.56%` (`FP32`, 120 epoch) dan *`85.87%`* (`INT8` QAT, selisih hanya `-0.69%`), dengan total hanya *`1,418` parameter* (`1,520` byte).
    - *Verifikasi Simulasi (ModelSim 18.1):* `0` kesalahan dari `31,828` pengujian modul unit dan `0` kesalahan dari `11,731` elemen tensor internal pada `cnn_top.v` dengan latensi *`302,978` siklus clock (`6.060 ms` pada `50 MHz` = `165.0 FPS`)*.
    - *Implementasi Hardware (Quartus Prime 18.1 + DE10-Standard):* Menggunakan *`1,942 / 41,910` ALMs (`5%`)*, *`22 / 112` DSP (`20%`)*, *`32 / 553` M10K BRAM (`6%`)*, dengan frekuensi maksimum $F_"max" = 69.73" MHz"$, serta dilengkapi output monitor *VGA (`640×480 @ 60 Hz`)*, simulasi virtual *DESim*, dan *GUI Python* untuk menggambar langsung atau mengunggah foto ke FPGA melalui USB-JTAG.
  ],
)

// ============================================================================
// BAB 1: STEP-BY-STEP SETUP LINGKUNGAN & HARDWARE
// ============================================================================
= Panduan Step-by-Step Setup Lingkungan & Hardware

Bagian ini menjelaskan langkah demi langkah persiapan perangkat lunak (*software*) dan perangkat keras (*hardware*) dari awal hingga proyek dapat dijalankan pada board *Terasic DE10-Standard*.

== Langkah 1: Instalasi Python 3.12 & Pustaka AI/GUI
1. Pasang *Python 3.12 (64-bit)* (pastikan opsi *"Add python.exe to PATH"* dan *Tcl/Tk* dicentang agar `tkinter` berjalan).
2. Buka terminal *PowerShell* pada direktori proyek (`c:\Users\Ersha Anandita\Downloads\fpga_midterm`) dan pasang pustaka yang dibutuhkan:
   ```powershell
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
   pip install numpy matplotlib pillow pytest
   ```
3. Verifikasi GPU CUDA dan Tkinter:
   ```powershell
   python -c "import torch, numpy, matplotlib, PIL, tkinter; print('CUDA:', torch.cuda.is_available(), '| Tk OK')"
   ```

== Langkah 2: Instalasi Intel Quartus Prime Lite 18.1, ModelSim 18.1 & DESim
1. Pasang *Intel Quartus Prime Lite Edition 18.1 (Build 625)* beserta komponen:
   - *Quartus Prime Lite* (`C:\intelFPGA_lite\18.1\quartus\bin64\`)
   - *Cyclone V Device Support* (untuk chip `5CSXFC6D6F31C6` pada DE10-Standard)
   - *ModelSim - Intel FPGA Starter Edition 18.1* (`C:\intelFPGA_lite\18.1\modelsim_ase\win32aloem\`)
2. Pasang *DESim* (`C:\desim\DESim_run.bat`) untuk simulasi interaktif board virtual (Switch, Push Button, 7-Segment, LED, dan layar VGA).
3. Pasang *Typst* untuk kompilasi dokumen `.typ` ke `.pdf` melalui `winget install --id Typst.Typst`.

== Langkah 3: Koneksi Fisik Board Terasic DE10-Standard
1. Sambungkan adaptor daya *`12V DC`* ke board DE10-Standard dan tekan tombol power merah.
2. Sambungkan kabel Micro-USB dari laptop ke port *`USB Blaster II`* pada board (bukan `USB-UART`). Jika driver belum terpasang, arahkan Windows Device Manager ke `C:\intelFPGA_lite\18.1\quartus\drivers\usb-blaster-ii`.
3. Pastikan rantai JTAG terdeteksi (`4BA00477 SOCVHPS` dan `02D020DD Cyclone V`):
   ```powershell
   C:\intelFPGA_lite\18.1\quartus\bin64\jtagconfig.exe
   ```
4. *(Opsional)* Hubungkan kabel *VGA 15-pin* dari port VGA biru ke monitor eksternal (`640×480 @ 60 Hz`).

== Langkah 4: Alur Pengujian & Menjalankan Demo Lengkap
1. Jalankan seluruh `58` pengujian otomatis (`Milestone 1–10`): `python -m pytest -v sw/tests`.
2. Jalankan program board + GUI Python sekaligus dengan membuka `run_demo.bat` (atau jalankan `python -m sw.gui_app` lalu klik tombol *`⚡ Flash FPGA (.sof)`*).

// ============================================================================
// BAB 2: ALUR KERJA 10 MILESTONE & DOKUMENTASI ARSITEKTUR
// ============================================================================
= Dokumentasi Rancangan Sistem (10 Milestone Co-Design)

Pengembangan akselerator CNN ini dibagi ke dalam *10 Milestone* berurutan agar setiap tahap dapat diverifikasi secara matematis sebelum masuk ke level perangkat keras:

#figure(
  table(
    columns: (auto, 1.2fr, 1.8fr, 1.25fr),
    inset: 5pt,
    align: (center, left, left, left),
    fill: (x, y) => if y == 0 { rgb("#1e3a8a") } else if calc.odd(y) { rgb("#f8fafc") } else { white },
    [#text(fill: white, weight: "bold")[Tahap]],
    [#text(fill: white, weight: "bold")[Milestone]],
    [#text(fill: white, weight: "bold")[Penjelasan Proses & Spesifikasi]],
    [#text(fill: white, weight: "bold")[Hasil Terverifikasi]],
    [*M1*], [Dataset & Preprocessing], [Unduh Fashion-MNIST (`zalandoresearch/fashion-mnist`), bagi `54k/6k/10k` (`seed=42`), tambah *2-px zero-padding* (`28×28 → 32×32`), kuantisasi `q_in = pixel >> 1` (`[0, 127]`)], [`1,024` byte/citra (`1` blok M10K)],
    [*M2*], [Arsitektur Model CNN], [`Conv1(3×3, 1→8) → ReLU → Pool(2×2) → Conv2(3×3, 8→16) → ReLU → GAP(13×13) → FC(16→10) → Argmax` (`sw/model/cnn.py`)], [`1,418` parameter; `259,648` MAC],
    [*M3*], [Training Baseline FP32], [Latih `120` epoch di GPU (`AdamW max_lr=0.01`, `OneCycleLR`, `batch=128`) via `sw/train/train.py`], [Akurasi uji *`86.56%`* (Epoch 119)],
    [*M4*], [Kuantisasi `INT8` (QAT)], [Bandingkan PTQ (`85.02%`) vs. QAT `12` epoch (`85.87%`); hitung pengali diadik `(M, n)` (`sw/quant/quantize.py`)], [Akurasi `INT8` *`85.87%`* (`-0.69%`)],
    [*M5*], [Ekspor `.mem` & `.mif`], [Konversi bobot `INT8` & bias `INT32` ke format heksadesimal komplemen-2 (`sw/export/export_mem.py`)], [`16` berkas hex (`0` galat baca)],
    [*M6*], [Model Referensi Integer], [Simulasikan loop hardware murni `INT8`/`INT32` di Python (`sw/refmodel/int8_ref_model.py`) & cetak vektor emas], [`100%` identik pada `10,000` citra],
    [*M7*], [Desain RTL Verilog-2001], [Tulis `7` modul RTL (`conv_engine`, `relu`, `maxpool`, `gap`, `fully_connected`, `argmax`, `cnn_top`) di `rtl/`], [`0` error, `0` warning (`vlog`)],
    [*M8*], [Simulasi Unit ModelSim], [Uji masing-masing `6` modul komputasi terhadap vektor emas M6 (`sim/tb/tb_*.v`)], [`6/6` lulus (`0 / 31,828` galat)],
    [*M9*], [Simulasi Integrasi Penuh], [Uji `cnn_top.v` pada Citra 0 (`11,731` elemen BRAM) & sweep `10` kelas (`Vec 0..90`)], [`0` galat; *`302,978` siklus* (`6.06 ms`)],
    [*M10*], [Sintesis Quartus & Board], [Sintesis `de10_standard_top.v` (CNN + VGA + JTAG + PS/2), program `.sof`, & uji langsung di DE10-Standard], [`1,942` ALMs (`5%`), $F_"max"=69.73" MHz"$],
  ),
  caption: [Ringkasan alur kerja 10 Milestone Proyek Midterm (Metode 1).],
)

== Penjelasan Matematika Kuantisasi & Arsitektur Hardware
1. *Kuantisasi Input Tanpa Gerbang Logika (`0` Gate):* Citra `28×28` (`0..255`) diberi padding `2` piksel nol di keempat sisi menjadi `32×32` (`1,024` piksel), lalu digeser kanan `1` bit: $q_"in" = p >> 1 in [0, 127]$ dengan skala $s_"in" = 2^(-7) = 0.0078125$.
2. *Requantization Diadik (Pengali + Geser Bit):* Agar FPGA tidak memakai pembagian desimal (`FP32`), akumulator `INT32` dikalikan konstanta bulat 16-bit $M$ dan digeser kanan $n$ bit:
   $ q_"out" = "clip" lr(( floor(frac("acc"_"int32" times M + 2^(n-1), 2^n)), q_"min", q_"max" )) $
   - `Conv1`: $M = 23202, n = 19$ (`[0, 127]`) | `Conv2`: $M = 16868, n = 20$ (`[0, 127]`)
   - `GAP (÷ 169)`: $M = 16780, n = 19$ (`[0, 127]`, akurat mutlak untuk semua `21,464` kemungkinan penjumlahan!)
   - `FC (16 → 10)`: $M = 23793, n = 21$ (`[-128, 127]`, $s_"fc" = 0.4488924$)
3. *Keputusan Kelas & Persentase Keyakinan (`Softmax`):* Di dalam FPGA (`argmax.v`), kelas pemenang ditentukan dari nilai skor `INT8` terbesar pada `fc_score_ram[0..9]` (contoh Citra 0: `[+7, -28, -5, -2, -19, -25, -2, -36, -7, -14]`, pemenang = indeks `0` *T-shirt/top*). Pada GUI Python, persentase keyakinan (`%`) dihitung dengan fungsi *Softmax*: $P(k) = exp(q_k dot s_"fc") / sum_j exp(q_j dot s_"fc") times 100\%$.

// ============================================================================
// BAB 3: HASIL TRAINING, KUANTISASI & VISUALISASI GAMBAR
// ============================================================================
= Hasil Training, Kuantisasi & Analisis Gambar

== Perbandingan Akurasi FP32 vs. PTQ vs. QAT INT8
Model dilatih selama `120` epoch menggunakan GPU NVIDIA GeForce RTX 5050 Laptop dengan `AdamW` (`max_lr = 0.01`) dan `OneCycleLR`. Checkpoint terbaik (`fp32_best.pt`) dicapai pada *Epoch 119*.

#figure(
  table(
    columns: (1.4fr, 1fr, 1fr, 1.2fr, 1fr, 1.1fr),
    inset: 6pt,
    align: (left, center, center, center, center, center),
    fill: (x, y) => if y == 0 { rgb("#1e3a8a") } else if calc.odd(y) { rgb("#f8fafc") } else { white },
    [#text(fill: white, weight: "bold")[Pembagian Data]],
    [#text(fill: white, weight: "bold")[FP32 Baseline]],
    [#text(fill: white, weight: "bold")[PTQ INT8]],
    [#text(fill: white, weight: "bold")[QAT INT8 (Terpilih)]],
    [#text(fill: white, weight: "bold")[Selisih vs. FP32]],
    [#text(fill: white, weight: "bold")[Kecocokan FP32]],
    [Train (`54,000`)], [`87.69%`], [–], [*`86.84%`*], [`-0.85%`], [`95.65%`],
    [Validation (`6,000`)], [`87.27%`], [`85.78%`], [*`86.97%`*], [`-0.30%`], [`96.00%`],
    [Test (`10,000`)], [`86.56%`], [`85.02%`], [*`85.87%`*], [*`-0.69%`*], [*`95.49%`*],
    [HW Test Vectors (`100`)], [`85.0%`], [–], [*`82.0%`*], [`-3.0%`], [`93.0%`],
  ),
  caption: [Hasil evaluasi akurasi model `FP32` (`120` epoch) dibandingkan dengan `PTQ` dan `QAT INT8`.],
)

== Kurva Training Loss, Akurasi & Distribusi Bobot
Gambar berikut memperlihatkan penurunan *Cross-Entropy Loss* dan kenaikan akurasi selama `120` epoch (`reports/training/training_curves.png`), serta sampel citra input `32×32` `INT8` (`reports/dataset/samples_int8_grid.png`):

#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 10pt,
    image("reports/training/training_curves.png", width: 100%),
    image("reports/dataset/samples_int8_grid.png", width: 100%),
  ),
  caption: [Kiri: Kurva *Training & Validation Loss* serta *Accuracy* selama `120` epoch (`AdamW + OneCycleLR`). Kanan: Contoh citra `32×32` `INT8` (`[0, 127]`) pada seluruh 10 kelas Fashion-MNIST.],
)

#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 10pt,
    image("reports/quantization/int8_confusion_matrix.png", width: 100%),
    image("reports/quantization/weight_distributions.png", width: 100%),
  ),
  caption: [Kiri: *Confusion Matrix* model `INT8` QAT pada `10,000` citra uji (`85.87%`). Kelas `Bag` (`95.7%`), `Sandal` (`95.6%`), `Trouser` (`95.5%`), dan `Sneaker` (`94.7%`) sangat tinggi, sedangkan `Shirt` (`64.8%`) kerap bertukar dengan `T-shirt`/`Coat`. Kanan: Distribusi bobot `FP32` vs. `INT8` pada `Conv1`, `Conv2`, dan `FC`.],
)

// ============================================================================
// BAB 4: HASIL IMPLEMENTASI FPGA, SIMULASI & OUTPUT VGA INTERAKTIF
// ============================================================================
= Hasil Implementasi FPGA, Simulasi ModelSim & Output VGA

== Hasil Sintesis & Sumber Daya Hardware Quartus Prime (`5CSXFC6D6F31C6`)
Setelah seluruh modul (`cnn_top.v`, `vga_cnn_hud.v`, `ps2_mouse_ctrl.v`, `altsource_probe` 272-bit JTAG, dan controller board `de10_standard_top.v`) dikompilasi di Quartus Prime 18.1:
- *Logic Utilization (ALMs):* `1,942 / 41,910` (*`5%`*) | *Total Registers:* `2,768`
- *On-Chip M10K BRAM:* `32 / 553` blok (*`6%`*, `194,384` bit total; khusus inti `cnn_top` = `104,272` bit / `21` M10K)
- *DSP Blocks:* `22 / 112` (*`20%`*: `9` di `conv_engine`, `6` di `gap`, `7` di `fully_connected`)
- *TimeQuest Timing ($F_"max"$ Slow 85°C):* *`69.73 MHz`* (Setup Slack `+5.659 ns` terhadap clock `50.00 MHz`, `0` pelanggaran timing).
- *Kecepatan Inferensi Nyata:* `302,978` siklus = *`6.060 ms` (`165.0 FPS`)* pada `50 MHz`, atau *`4.345 ms` (`230.1 FPS`)* pada $F_"max"$.

== Hasil Simulasi ModelSim & Render Output VGA (`640×480 @ 60 Hz`)
Gambar di bawah ini menunjukkan bukti verifikasi *ModelSim*: (kiri) gelombang sinyal `fc_score_ram` dan `predicted_class = 0` (`done = 1`) pada `6,060 us`, dan (kanan) hasil render frame nyata **`640×480`** langsung dari sinyal `VGA_R`, `VGA_G`, `VGA_B` milik modul `rtl/vga_cnn_hud.v` (`tb_vga_cnn_hud.v`):

#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 10pt,
    image("reports/simulation/modelsim_waveform.jpg", width: 100%),
    image("reports/simulation/modelsim_vga_frame.png", width: 100%),
  ),
  caption: [Kiri: Waveform ModelSim saat `cnn_top.v` menulis 10 skor kelas dan menyalakan `done = 1`. Kanan: Output frame VGA `640×480` (`307,200` piksel aktif) yang dirender langsung oleh `rtl/vga_cnn_hud.v` di ModelSim.],
)

== Visualisasi Pipeline Internal FPGA & Uji 10 Kelas
#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 10pt,
    image("reports/simulation/fpga_cnn_pipeline_visual.png", width: 100%),
    image("reports/simulation/fpga_10class_detections.png", width: 100%),
  ),
  caption: [Kiri: Isi memori internal FPGA (`img_ram → conv1_relu_ram → pool1_ram → conv2_relu_ram → gap_ram → fc_score_ram`) untuk Citra 0 (`T-shirt/top`). Kanan: Pengujian `10` kelas (`Vec 0..90`) menghasilkan akurasi `10/10` (`100%`) dengan `0` selisih bit antara RTL dan Python.],
)

== Cara Menggunakan Output VGA & Menggambar Langsung ke FPGA
Selain menampilkan 10 citra uji bawaan melalui sakelar fisik `SW[3:0]` (dengan tampilan `HEX5..HEX0` seperti `0 - P A S 0` atau `3 0 2 9 7 8` saat `SW[9]=1`), proyek ini mendukung interaksi menggambar langsung ke layar VGA:
1. *Menggambar Langsung dari Laptop ke Layar VGA (`sw/gui_app.py`):*
   - Jalankan `python -m sw.gui_app`, klik tombol *`✏️ Clear & Draw`* (atau tekan tombol fisik *`KEY[3]`* di board untuk menghapus layar).
   - Klik-kiri dan seret mouse di kotak `32×32` pada GUI; piksel langsung dikirim baris-per-baris melalui *272-bit Row-Burst USB-JTAG* (`sw/jtag_bridge.py`) ke `vga_img_ram` dan `img_ram` di DE10-Standard, sehingga goresan langsung tampil di monitor VGA eksternal dan langsung diklasifikasi oleh CNN hardware dalam `6.06 ms`!
   - Pengguna juga dapat mengunggah foto pakaian (misalnya dari Pinterest) melalui tombol *`📷 Upload Pinterest / Photo...`*.
2. *Simulasi Virtual Lengkap di DESim (`desim_cnn/`):*
   - Buka folder `desim_cnn` di aplikasi *DESim*, klik `Compile Testbench` lalu `Start Simulation` untuk melihat layar VGA virtual `160×120` (lengkap dengan gambar pakaian, label `0 TSHIRT`..`9 BOOT`, bar skor, dan `WINNER`), `HEX5..HEX0`, serta `LEDR` secara interaktif tanpa board fisik.
