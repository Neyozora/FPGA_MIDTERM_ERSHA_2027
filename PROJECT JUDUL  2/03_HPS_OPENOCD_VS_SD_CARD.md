# Running the Cyclone V HPS (ARM Cortex-A9): OpenOCD (No SD Card) vs. SD Card Boot

Your **DE10-Standard** board has an **Intel Cyclone V SX SoC (`5CSXFC6D6F31C6`)** containing **both**:
1. **FPGA Fabric (`02D020DD`)** — where our `cnn_top.v` hardware accelerator lives.
2. **HPS Dual-Core ARM Cortex-A9 (`4BA00477 SOCVHPS`)** — a real `800–925 MHz` ARM CPU wired directly to the FPGA via **AXI Bridges (`0xFF200000` Lightweight HPS-to-FPGA & `0xC0000000` Heavyweight HPS-to-FPGA)**.

You asked: *"I want to use HPS with OpenOCD so I don't need an SD card, but if the SD card method is easier, go ahead."* Here is the honest engineering comparison and exact setup for **both**!

---

## Quick Verdict: Which Should You Choose?

| Feature | Option A: **HPS via OpenOCD (No SD Card)** | Option B: **SD Card Linux Boot (Recommended for Demo!)** |
| :--- | :--- | :--- |
| **Needs SD Card?** | **No!** Loads bare-metal `.elf` directly over USB-Blaster II (`DE-SoC [USB-1]`). | Yes (microSD card `>= 4 GB` flashed with Terasic DE10-Standard Linux image). |
| **Setup Difficulty on Windows** | **Medium-Hard:** Requires `arm-none-eabi-gcc` + OpenOCD + running the Cyclone V **Preloader (`u-boot-spl`)** first to initialize DDR3 & AXI bridges, OR fitting your entire C program inside the `64 KB` On-Chip RAM (`0xFFFF0000`). | **Very Easy:** Flash `.img` once with Rufus/Win32DiskImager, boot Linux in 10 seconds, compile C directly on the board with `gcc`, and talk to `cnn_top.v` via `/dev/mem`! |
| **USB Camera / Wi-Fi / Python on Board?** | No OS (bare-metal only — no USB webcam driver, no Python, no filesystem). | **Full Ubuntu/Angstrom Linux!** You can plug a USB webcam or run Python + OpenCV right on the ARM CPU! |
| **Best Use Case** | Fast bare-metal C debugging, university microprocessor labs, or when you have no microSD card on hand. | **Best for final LSI Contest / Midterm demos** (real camera/image loading + web/GUI + FPGA AXI acceleration). |

---

## Option A: Using HPS with OpenOCD Over USB-Blaster II (Zero SD Card)

Because your DE10-Standard's `USB-Blaster II` chip is already connected (`DE-SoC [USB-1]`), OpenOCD can halt the ARM Cortex-A9 (`4BA00477`) and load a bare-metal C program directly into the HPS's **`64 KB` On-Chip RAM (`0xFFFF0000`)**—without needing an SD card or even DDR3 initialization!

### Step 1: Install Tools on Windows
1. **ARM Bare-Metal Compiler:** Install `arm-none-eabi-gcc` (from Arm GNU Toolchain).
2. **OpenOCD for Windows:** Download OpenOCD (`>= 0.11` with `usb_blaster` / `altera` support).
3. *(Note on USB-Blaster II driver for OpenOCD)*: OpenOCD uses `libusb` (`WinUSB`), whereas Quartus uses Intel's `USB-Blaster II` driver. Switching drivers with Zadig is reversible, OR you can use Intel SoC EDS's built-in `arm-altera-eabi-gdb` / Quartus `System Console` with the existing Quartus driver!

### Step 2: Compile & Load Bare-Metal C Code into On-Chip RAM (`0xFFFF0000`)
We provided a starter config [`hps_openocd_starter/openocd_de10_standard.cfg`](hps_openocd_starter/openocd_de10_standard.cfg) and C driver [`hps_openocd_starter/hps_cnn_bridge.c`](hps_openocd_starter/hps_cnn_bridge.c):
1. Add a **Platform Designer (Qsys)** system in Quartus connecting `hps_0.h2f_lw_axi_master` (`0xFF200000`) to a small Avalon-MM Slave register wrapper around `cnn_top.v`.
2. Compile your bare-metal C code (`-Ttext=0xFFFF0000` so it fits in the `64 KB` On-Chip RAM):
   ```powershell
   arm-none-eabi-gcc -mcpu=cortex-a9 -O2 -nostartfiles -Wl,-Ttext=0xFFFF0000 hps_cnn_bridge.c -o hps_cnn.elf
   ```
3. Load and run via OpenOCD (no SD card required!):
   ```powershell
   openocd -f hps_openocd_starter/openocd_de10_standard.cfg -c "init; halt; load_image hps_cnn.elf; resume 0xFFFF0000; exit"
   ```

---

## Option B: SD Card Method (Much Easier for Full Demos!)

If you have a `4 GB+` microSD card, this is **far easier** and unlocks full Linux + USB camera + Python + `gcc` right on the DE10-Standard:

1. **Flash Terasic's Official DE10-Standard SD Card Image:**
   - Download `DE10_Standard_Linux_Console.img` (or LXDE Desktop `.img`) from Terasic's DE10-Standard Resources page.
   - Write it to a microSD card using **Rufus** or **BalenaEtcher**, insert it into the DE10-Standard's microSD slot, and set the `MSEL[4:0]` DIP switches on the back of the board to `01010` (so HPS boots Linux and programs the FPGA `.rbf` automatically, OR keep `MSEL=10010` and program `.sof` from Quartus!).
2. **Connect Over UART / SSH:**
   - Plug a Micro-USB cable into the **`USB-UART`** port (`115200` baud) and log in as `root`.
3. **Talk to `cnn_top.v` in 10 Lines of C (via `/dev/mem`):**
   - Map the Lightweight HPS-to-FPGA AXI Bridge (`0xFF200000`) using `mmap()` in Linux C (`gcc hps_cnn_bridge.c -DLINUX_USERMODE -o run_cnn && ./run_cnn`).
   - Your ARM CPU can now read JPEG/PNG images (or live USB webcam frames via OpenCV), send the `1,024` pixels over AXI at `> 50 MB/s` into `cnn_top.v`, and read back `predicted_class` and all 10 `fc_score_ram` values in microseconds!
