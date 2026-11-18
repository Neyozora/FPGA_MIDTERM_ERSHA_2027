"""Upload `main_cifar10_vga_bridge.c` + `hw_images_100.bin` to `/home/xuser/hps_cifar10_lcd`, enable lwhps2fpga bridge, compile, and run."""
import base64
import time
import serial
from sw_m2.config import REPO_ROOT


def send_file_b64(ser: serial.Serial, local_path, remote_name: str):
    data = local_path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    ser.write(f"cat << 'EOF_B64' | base64 -d > {remote_name}\n".encode())
    time.sleep(0.05)
    for i in range(0, len(b64), 240):
        ser.write((b64[i : i + 240] + "\n").encode())
        time.sleep(0.022)
        if i % 4800 == 0:
            ser.read(4096)
    ser.write(b"EOF_B64\n")
    time.sleep(0.3)
    ser.read(4096)
    print(f"Uploaded {remote_name} ({len(data)} bytes)")


def main():
    hps_dir = REPO_ROOT / "hps_m2"
    ser = serial.Serial("COM7", 115200, timeout=0.3)
    ser.write(b"\x03\n")
    time.sleep(0.3)
    ser.read(4096)

    ser.write(b"cd /home/xuser/hps_cifar10_lcd\n")
    time.sleep(0.2)
    ser.read(4096)

    # Enable Lightweight HPS-to-FPGA bridge in Linux sysfs after JTAG .sof programming
    ser.write(b"echo 1 > /sys/class/fpga-bridge/lwhps2fpga/enable 2>/dev/null; cat /sys/class/fpga-bridge/lwhps2fpga/enable 2>/dev/null\n")
    time.sleep(0.3)
    print("Bridge status:", ser.read(1024).decode(errors="replace").strip())

    send_file_b64(ser, hps_dir / "main_cifar10_vga_bridge.c", "main_cifar10_vga_bridge.c")
    # hw_images_100.bin is already uploaded on HPS; skip re-sending 307 KB

    cmd = (
        "gcc -O2 -std=gnu99 -Wall -Dsoc_cv_av "
        "-Iintel-socfpga-hwlib/armv7a/hwlib/include/soc_cv_av "
        "-Iintel-socfpga-hwlib/armv7a/hwlib/include "
        "-I. main_cifar10_vga_bridge.c terasic_lib.c LCD_Lib.c LCD_Driver.c LCD_Hw.c lcd_graphic.c font.c "
        "-o hps_vga_bridge -lrt -lm\n"
    )
    ser.write(cmd.encode())
    time.sleep(5.0)
    print("Compile output:\n" + ser.read(8192).decode(errors="replace"))

    ser.write(b"killall hps_lcd hps_cifar10_lcd hps_vga_bridge 2>/dev/null; ./hps_vga_bridge &\n")
    time.sleep(2.0)
    print("Run output:\n" + ser.read(4096).decode(errors="replace"))
    ser.close()


if __name__ == "__main__":
    main()
