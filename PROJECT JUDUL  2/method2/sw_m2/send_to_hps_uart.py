"""Send `cifar10_lcd_data.h` and `main_cifar10_lcd.c` to DE10-Standard HPS over COM7, compile, and run."""
import base64
import time
import serial
from sw_m2.config import REPO_ROOT


def send_file_b64(ser: serial.Serial, local_path, remote_name: str):
    data = local_path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    ser.write(f"cat << 'EOF_B64' | base64 -d > {remote_name}\n".encode())
    time.sleep(0.05)
    for i in range(0, len(b64), 120):
        ser.write((b64[i : i + 120] + "\n").encode())
        time.sleep(0.015)
        ser.read(4096)
    ser.write(b"EOF_B64\n")
    time.sleep(0.2)
    out = ser.read(4096).decode(errors="replace")
    print(f"Uploaded {remote_name} ({len(data)} bytes)")


def main():
    hps_dir = REPO_ROOT / "hps_m2"
    ser = serial.Serial("COM7", 115200, timeout=0.3)
    ser.write(b"\x03\n")
    time.sleep(0.3)
    ser.read(4096)

    ser.write(b"cp -r /home/xuser/hps_lcd_aquarium /home/xuser/hps_cifar10_lcd 2>/dev/null; cd /home/xuser/hps_cifar10_lcd\n")
    time.sleep(0.5)
    ser.read(4096)

    send_file_b64(ser, hps_dir / "cifar10_lcd_data.h", "cifar10_lcd_data.h")
    send_file_b64(ser, hps_dir / "main_cifar10_lcd.c", "main_cifar10_lcd.c")

    # Compile using the exact same HWLIBS flags as Makefile
    cmd = (
        "gcc -O2 -std=gnu99 -Wall -Dsoc_cv_av "
        "-Iintel-socfpga-hwlib/armv7a/hwlib/include/soc_cv_av "
        "-Iintel-socfpga-hwlib/armv7a/hwlib/include "
        "-I. main_cifar10_lcd.c terasic_lib.c LCD_Lib.c LCD_Driver.c LCD_Hw.c lcd_graphic.c font.c "
        "-o hps_cifar10_lcd -lrt -lm\n"
    )
    ser.write(cmd.encode())
    time.sleep(4.0)
    print("Compile output:\n" + ser.read(8192).decode(errors="replace"))

    # Start in background so LCD animates immediately and user can reopen PuTTY
    ser.write(b"killall hps_lcd hps_cifar10_lcd 2>/dev/null; ./hps_cifar10_lcd &\n")
    time.sleep(1.0)
    print("Run output:\n" + ser.read(4096).decode(errors="replace"))
    ser.close()


if __name__ == "__main__":
    main()
