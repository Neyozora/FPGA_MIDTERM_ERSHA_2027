"""
Method 2 RGB Row-Burst USB-JTAG Bridge (`sw_m2/jtag_bridge_m2.py`)
Communicates with the 272-bit `altsource_probe` ("CNN2") inside
`rtl_m2/de10_standard_top_m2.v` via `quartus_stp.exe`.
Streams a 32x32x3 RGB image in 96 row-burst writes (3 channels x 32 rows),
triggers FPGA CNN inference, updates the live VGA RGB HUD (and HPS LCD),
and reads back the 128-bit hardware telemetry probe!
"""
from pathlib import Path
import os
import subprocess
import tempfile
import time
from typing import Dict, List
import numpy as np

QUARTUS_STP_CANDIDATES = [
    Path(r"C:\intelFPGA_lite\18.1\quartus\bin64\quartus_stp.exe"),
    Path(r"C:\altera_lite\24.1std\quartus\bin64\quartus_stp.exe"),
]


def find_quartus_stp() -> Path:
    for p in QUARTUS_STP_CANDIDATES:
        if p.exists():
            return p
    return QUARTUS_STP_CANDIDATES[0]


def _row_to_hex64(row_32: np.ndarray) -> str:
    u8 = np.asarray(row_32, dtype=np.uint8).reshape(32)
    return "".join(f"{int(u8[31 - c]):02X}" for c in range(32))


def send_rgb_image_to_fpga(img_q: np.ndarray) -> Dict:
    """
    Stream (3, 32, 32) int8 RGB image over JTAG to DE10-Standard `CNN2` instance.
    Returns dict with ok, pred, cycles, scores (int8[10]), wall_s.
    """
    stp_exe = find_quartus_stp()
    if not stp_exe.exists():
        return {"ok": False, "error": f"quartus_stp.exe not found at {stp_exe}"}

    t0 = time.time()
    words68: List[str] = []
    wr_tog = 0
    # 3 channels (0=R, 1=G, 2=B) x 32 rows
    for ch in range(3):
        for ry in range(32):
            wr_tog ^= 1
            # ctrl_hi16: bits[12:11]=ch, bit[7]=pc_mode(1), bit[5]=wr_tog, bits[4:0]=ry
            ctrl_hi = ((ch & 3) << 11) | (1 << 7) | (wr_tog << 5) | (ry & 0x1F)
            words68.append(f"{ctrl_hi:04X}{_row_to_hex64(img_q[ch, ry])}")

    # Trigger inference by toggling bit 8 (`jtag_infer`: 0 -> 1)
    ctrl_pre   = (0 << 8) | (1 << 7)
    ctrl_infer = (1 << 8) | (1 << 7)
    pre_word   = f"{ctrl_pre:04X}" + ("0" * 64)
    infer_word = f"{ctrl_infer:04X}" + ("0" * 64)

    tcl = f"""
set hw_list [get_hardware_names]
if {{[llength $hw_list] == 0}} {{
    puts "ERR:NO_CABLE"
    exit 1
}}
set hw_name [lindex $hw_list 0]
foreach h $hw_list {{
    if {{[string match "*DE-SoC*" $h]}} {{ set hw_name $h; break }}
}}
set dev_name ""
foreach d [get_device_names -hardware_name $hw_name] {{
    if {{[string match "*5CSXFC6D6*" $d] || [string match "*@2*" $d]}} {{ set dev_name $d; break }}
}}
if {{$dev_name eq ""}} {{ set dev_name [lindex [get_device_names -hardware_name $hw_name] end] }}
start_insystem_source_probe -hardware_name $hw_name -device_name $dev_name
foreach w {{{' '.join(words68)}}} {{
    write_source_data -instance_index 0 -value $w -value_in_hex
}}
write_source_data -instance_index 0 -value {pre_word} -value_in_hex
write_source_data -instance_index 0 -value {infer_word} -value_in_hex
after 25
set p [read_probe_data -instance_index 0 -value_in_hex]
end_insystem_source_probe
puts "PROBE_HEX:$p"
"""
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".tcl", delete=False) as tf:
            tf.write(tcl)
            tmp_path = tf.name
        proc = subprocess.run(
            [str(stp_exe), "-t", tmp_path],
            capture_output=True,
            text=True,
            timeout=30,
            creationflags=creationflags,
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
    out = proc.stdout + "\n" + proc.stderr
    for line in out.splitlines():
        if line.startswith("PROBE_HEX:"):
            hex_str = line.split("PROBE_HEX:", 1)[1].strip()
            val = int(hex_str, 16)
            scores = np.array(
                [(val >> (i * 8)) & 0xFF for i in range(10)], dtype=np.uint8
            ).view(np.int8)
            pred = (val >> 80) & 0xF
            done = (val >> 84) & 1
            cycles = (val >> 88) & 0xFFFFFF
            return {
                "ok": True,
                "pred": int(pred),
                "done": int(done),
                "cycles": int(cycles),
                "scores": scores,
                "wall_s": time.time() - t0,
            }
    return {"ok": False, "error": "JTAG probe failed", "raw": out[-400:]}
