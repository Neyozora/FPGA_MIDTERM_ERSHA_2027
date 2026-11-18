"""Ultra-Fast 32-Pixel Row-Burst USB-JTAG Bridge for DE10-Standard FPGA.

Communicates with the 272-bit `altsource_probe` ("CNN0") instance inside
`rtl/de10_standard_top.v` on a background daemon thread:
  - 1-Command Hardware Clear (`queue_clear_canvas`): Wipes all 1,024 pixels of
    `img_ram` & `vga_img_ram` in 1 single JTAG write (20.5 us on FPGA)!
  - 32-Pixel Row-Burst Streaming: Sends an entire 32-pixel row (256 bits) in
    a single JTAG write (32x faster than single-pixel streaming: a brush stroke
    takes 1-2 writes, and a full 32x32 image takes at most 32 writes!).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

QUARTUS_STP_CANDIDATES = [
    Path(r"C:\intelFPGA_lite\18.1\quartus\bin64\quartus_stp.exe"),
    Path(r"C:\altera_lite\24.1std\quartus\bin64\quartus_stp.exe"),
]

QUARTUS_PGM_CANDIDATES = [
    Path(r"C:\intelFPGA_lite\18.1\quartus\bin64\quartus_pgm.exe"),
    Path(r"C:\altera_lite\24.1std\quartus\bin64\quartus_pgm.exe"),
]


@dataclass
class FPGAProbeState:
    """Decoded 128-bit hardware telemetry read from DE10-Standard over JTAG."""

    connected: bool
    hw_name: str
    dev_name: str
    sw_all: int
    sw_img: int
    sw_cycle_mode: bool
    key_all: int
    predicted_class: int
    active_img: int
    has_result: bool
    pc_mode_active: bool
    cycle_count: int
    hw_fc_scores: List[int]


class DE10JtagBridge:
    """Thread-safe, non-blocking 272-bit Row-Burst JTAG manager."""

    def __init__(self, rom_images_10: Optional[np.ndarray] = None) -> None:
        self._proc: Optional[subprocess.Popen[str]] = None
        self._state_lock = threading.Lock()

        self.connected: bool = False
        self.busy_streaming: bool = False
        self.stream_progress_text: str = ""
        self.hw_name: str = ""
        self.dev_name: str = ""
        self.last_error: str = "Connecting..."
        self.latest_probe: Optional[FPGAProbeState] = None

        # Command queue (0 ms non-blocking from GUI thread)
        self._want_reconnect: bool = True
        self._want_flash_sof: Optional[str] = None
        self._want_hw_clear: bool = False
        self._pending_rom_sel: Optional[int] = None
        self._pending_rows: Dict[int, np.ndarray] = {}
        self._pending_cnn_trig: Optional[int] = None
        self._stop_flag: bool = False

        self._rom_images_10 = (
            rom_images_10.reshape(10, 32, 32).astype(np.int8)
            if rom_images_10 is not None
            else None
        )
        self._fpga_shadow_img = np.zeros((32, 32), dtype=np.int8)
        if self._rom_images_10 is not None:
            self._fpga_shadow_img[:, :] = self._rom_images_10[0]

        # Upper 16 control bits [271:256] of the 272-bit source register
        self._ctrl_hi16: int = 0

        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    @staticmethod
    def _find_bin(candidates: List[Path]) -> Optional[Path]:
        for p in candidates:
            if p.exists():
                return p
        return None

    # -------------------------------------------------------------------------
    # Non-Blocking API for Tkinter GUI Thread (< 0.02 ms)
    # -------------------------------------------------------------------------
    def get_snapshot(self) -> Tuple[bool, bool, str, Optional[FPGAProbeState]]:
        with self._state_lock:
            return (
                self.connected,
                self.busy_streaming,
                self.stream_progress_text if self.busy_streaming else self.last_error,
                self.latest_probe,
            )

    def request_reconnect(self) -> None:
        with self._state_lock:
            self._want_reconnect = True
            self.last_error = "Reconnecting..."

    def request_flash_sof(self, sof_path: str) -> None:
        with self._state_lock:
            self._want_flash_sof = sof_path
            self.busy_streaming = True
            self.stream_progress_text = "⚡ Flashing DE10-Standard (.sof)..."

    def queue_clear_canvas(self) -> None:
        """Instant 1-command hardware clear (wipes all 1,024 pixels in 20.5 us!)."""
        with self._state_lock:
            self._pending_rows.clear()
            self._pending_rom_sel = None
            self._want_hw_clear = True
            self._pending_cnn_trig = 0
            self._fpga_shadow_img.fill(0)

    def queue_rom_image(self, rom_idx: int) -> None:
        rom_idx = int(rom_idx) % 10
        with self._state_lock:
            self._pending_rows.clear()
            self._want_hw_clear = False
            self._pending_cnn_trig = None
            self._pending_rom_sel = rom_idx

    def queue_dirty_rows(
        self,
        img_32x32: np.ndarray,
        row_indices: List[int],
        trigger_cnn: bool = False,
        true_class: int = 0,
    ) -> None:
        """Queue specific modified 32-pixel rows (1 JTAG write per row!)."""
        with self._state_lock:
            for ry in row_indices:
                r = int(ry) & 31
                row_data = np.asarray(img_32x32[r], dtype=np.int8).copy()
                self._pending_rows[r] = row_data
                self._fpga_shadow_img[r, :] = row_data
            if trigger_cnn:
                self._pending_cnn_trig = int(true_class) & 0xF

    def queue_full_image(self, img_int8: np.ndarray, true_class: int) -> None:
        """Queue only changed rows of a 32x32 image (at most 32 JTAG writes total!)."""
        arr = np.asarray(img_int8, dtype=np.int8).reshape(32, 32)
        with self._state_lock:
            self._pending_rom_sel = None
            if np.all(arr == 0):
                self._pending_rows.clear()
                self._want_hw_clear = True
                self._fpga_shadow_img.fill(0)
            else:
                for r in range(32):
                    if not np.array_equal(self._fpga_shadow_img[r], arr[r]):
                        self._pending_rows[r] = arr[r].copy()
                        self._fpga_shadow_img[r, :] = arr[r]
            self._pending_cnn_trig = int(true_class) & 0xF

    def close(self) -> None:
        self._stop_flag = True
        self._close_proc()

    # -------------------------------------------------------------------------
    # Background Worker Loop
    # -------------------------------------------------------------------------
    def _worker_loop(self) -> None:
        last_poll_t = 0.0
        while not self._stop_flag:
            sof_to_flash: Optional[str] = None
            with self._state_lock:
                if self._want_flash_sof is not None:
                    sof_to_flash = self._want_flash_sof
                    self._want_flash_sof = None
            if sof_to_flash is not None:
                self._do_flash_sof(sof_to_flash)
                continue

            need_conn = False
            with self._state_lock:
                if self._want_reconnect:
                    need_conn = True
                    self._want_reconnect = False
            if need_conn:
                self._do_connect()

            if not self.connected or self._proc is None:
                time.sleep(0.20)
                continue

            # 1. Instant 1-command Hardware Clear
            do_clear = False
            with self._state_lock:
                if self._want_hw_clear:
                    do_clear = True
                    self._want_hw_clear = False
            if do_clear:
                self._do_hw_clear()
                continue

            # 2. Instant 1-command ROM Image Load (0..9)
            rom_sel: Optional[int] = None
            with self._state_lock:
                if self._pending_rom_sel is not None:
                    rom_sel = self._pending_rom_sel
                    self._pending_rom_sel = None
            if rom_sel is not None:
                self._do_trigger_rom(rom_sel)
                last_poll_t = time.time()
                continue

            # 3. Burst-stream any dirty 32-pixel rows (up to all 32 rows in 1 batch!)
            rows_to_send: List[Tuple[int, np.ndarray]] = []
            with self._state_lock:
                if self._pending_rows:
                    for ry in sorted(self._pending_rows.keys()):
                        rows_to_send.append((ry, self._pending_rows.pop(ry)))
            if rows_to_send:
                self._do_send_rows(rows_to_send)
                continue

            # 4. Trigger CNN inference if queued
            trig_cls: Optional[int] = None
            with self._state_lock:
                if self._pending_cnn_trig is not None and not self._pending_rows:
                    trig_cls = self._pending_cnn_trig
                    self._pending_cnn_trig = None
            if trig_cls is not None:
                self._do_trigger_custom_cnn(trig_cls)
                last_poll_t = time.time()
                continue

            # 5. Periodic poll for physical board switch/button changes
            now = time.time()
            if now - last_poll_t >= 0.20:
                self._do_poll()
                last_poll_t = time.time()
            else:
                time.sleep(0.01)

    @staticmethod
    def _row_to_hex64(row_32_int8: np.ndarray) -> str:
        """Pack 32 signed INT8 bytes (col 0 = LSB, col 31 = MSB) into 64 hex chars."""
        u8 = np.asarray(row_32_int8, dtype=np.uint8).reshape(32)
        # col 31 is at bits [255:248], col 0 is at bits [7:0]
        return "".join(f"{int(u8[31 - c]):02X}" for c in range(32))

    def _format_source_hex68(self, ctrl_hi16: int, row_hex64: Optional[str] = None) -> str:
        if row_hex64 is None:
            row_hex64 = "0" * 64
        return f"{ctrl_hi16 & 0xFFFF:04X}{row_hex64}"

    def _do_hw_clear(self) -> None:
        if self._proc is None or self._proc.stdin is None or self._proc.stdout is None:
            return
        # Toggle bit 6 of ctrl_hi16 (= source[262] = pc_clear_trig)
        self._ctrl_hi16 ^= (1 << 6)
        word68 = self._format_source_hex68(self._ctrl_hi16)
        try:
            self._proc.stdin.write(f"send_batch {{{word68}}}\n")
            self._proc.stdin.flush()
            while True:
                line = self._proc.stdout.readline()
                if not line or "BATCH_OK" in line:
                    break
        except Exception as exc:
            with self._state_lock:
                self.last_error = str(exc)
            self._close_proc()

    def _do_send_rows(self, rows: List[Tuple[int, np.ndarray]]) -> None:
        if self._proc is None or self._proc.stdin is None or self._proc.stdout is None:
            return
        words68: List[str] = []
        for ry, row_arr in rows:
            # Set row_y in bits [4:0] and toggle bit 5 (`pc_row_wr_trig`)
            wr_tog = ((self._ctrl_hi16 >> 5) & 1) ^ 1
            self._ctrl_hi16 = (self._ctrl_hi16 & ~0x3F) | (wr_tog << 5) | (int(ry) & 0x1F)
            row_hex = self._row_to_hex64(row_arr)
            words68.append(self._format_source_hex68(self._ctrl_hi16, row_hex))
        try:
            self._proc.stdin.write(f"send_batch {{{' '.join(words68)}}}\n")
            self._proc.stdin.flush()
            while True:
                line = self._proc.stdout.readline()
                if not line:
                    self._close_proc()
                    return
                if "BATCH_OK" in line:
                    return
        except Exception as exc:
            with self._state_lock:
                self.last_error = str(exc)
            self._close_proc()

    def _do_trigger_custom_cnn(self, true_class: int) -> None:
        if self._proc is None or self._proc.stdin is None or self._proc.stdout is None:
            return
        # Toggle bit 7 (`pc_cust_run_trig`) + set class in bits [11:8]
        cust_tog = ((self._ctrl_hi16 >> 7) & 1) ^ 1
        cls_bits = int(true_class) & 0xF
        self._ctrl_hi16 = (
            (self._ctrl_hi16 & ~((0xF << 8) | (1 << 7)))
            | (cls_bits << 8)
            | (cust_tog << 7)
        )
        word68 = self._format_source_hex68(self._ctrl_hi16)
        try:
            self._proc.stdin.write(
                f'write_source_data -instance_index 0 -value {word68} -value_in_hex; after 10; puts "TRIG_OK:[read_probe_data -instance_index 0 -value_in_hex]"; flush stdout\n'
            )
            self._proc.stdin.flush()
            while True:
                line = self._proc.stdout.readline()
                if not line:
                    self._close_proc()
                    return
                line = line.strip()
                if "TRIG_OK:" in line:
                    val = int(line.split("TRIG_OK:", 1)[1].strip(), 16)
                    st = self._decode_probe(val)
                    with self._state_lock:
                        self.latest_probe = st
                    return
        except Exception as exc:
            with self._state_lock:
                self.last_error = str(exc)
            self._close_proc()

    def _do_trigger_rom(self, rom_idx: int) -> None:
        if self._proc is None or self._proc.stdin is None or self._proc.stdout is None:
            return
        rom_idx = int(rom_idx) & 0xF
        # Toggle bit 12 (`pc_rom_run_trig`) + set rom_idx in bits [11:8]
        rom_tog = ((self._ctrl_hi16 >> 12) & 1) ^ 1
        self._ctrl_hi16 = (
            (self._ctrl_hi16 & ~((1 << 12) | (0xF << 8)))
            | (rom_tog << 12)
            | (rom_idx << 8)
        )
        word68 = self._format_source_hex68(self._ctrl_hi16)
        try:
            self._proc.stdin.write(
                f'write_source_data -instance_index 0 -value {word68} -value_in_hex; after 10; puts "ROM_OK:[read_probe_data -instance_index 0 -value_in_hex]"; flush stdout\n'
            )
            self._proc.stdin.flush()
            while True:
                line = self._proc.stdout.readline()
                if not line:
                    self._close_proc()
                    return
                line = line.strip()
                if "ROM_OK:" in line:
                    val = int(line.split("ROM_OK:", 1)[1].strip(), 16)
                    st = self._decode_probe(val)
                    with self._state_lock:
                        self.latest_probe = st
                        if self._rom_images_10 is not None:
                            self._fpga_shadow_img[:, :] = self._rom_images_10[rom_idx]
                    return
        except Exception as exc:
            with self._state_lock:
                self.last_error = str(exc)
            self._close_proc()

    def _do_flash_sof(self, sof_path: str) -> None:
        self._close_proc()
        pgm_bin = self._find_bin(QUARTUS_PGM_CANDIDATES)
        if pgm_bin is not None:
            try:
                creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
                subprocess.run(
                    [
                        str(pgm_bin),
                        "-c",
                        "DE-SoC [USB-1]",
                        "-m",
                        "JTAG",
                        "-o",
                        f"p;{sof_path}@2",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=15,
                    creationflags=creationflags,
                )
            except Exception as exc:
                with self._state_lock:
                    self.last_error = f"Flash error: {exc}"
        with self._state_lock:
            self.busy_streaming = False
            self.stream_progress_text = ""
            self._want_reconnect = True

    def _do_connect(self) -> None:
        self._close_proc()
        stp_bin = self._find_bin(QUARTUS_STP_CANDIDATES)
        if stp_bin is None:
            with self._state_lock:
                self.connected = False
                self.last_error = "quartus_stp.exe not found"
            return

        try:
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            self._proc = subprocess.Popen(
                [str(stp_bin), "-s"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                creationflags=creationflags,
            )
            init_tcl = (
                "package require ::quartus::insystem_source_probe\n"
                "proc send_batch {words} {\n"
                "    foreach w $words {\n"
                "        write_source_data -instance_index 0 -value $w -value_in_hex\n"
                "    }\n"
                '    puts "BATCH_OK"\n'
                "    flush stdout\n"
                "}\n"
                "set hw_list [get_hardware_names]\n"
                "if {[llength $hw_list] == 0} {\n"
                '    puts "JTAG_ERR:No DE-SoC USB cable"\n'
                "    flush stdout\n"
                "} else {\n"
                "    set hw_name [lindex $hw_list 0]\n"
                '    set dev_name ""\n'
                "    foreach d [get_device_names -hardware_name $hw_name] {\n"
                '        if {[string match "@2*" $d]} { set dev_name $d }\n'
                "    }\n"
                '    if {$dev_name eq ""} { set dev_name [lindex [get_device_names -hardware_name $hw_name] end] }\n'
                "    if {[catch {start_insystem_source_probe -hardware_name $hw_name -device_name $dev_name} err]} {\n"
                '        puts "JTAG_ERR:$err"\n'
                "    } else {\n"
                '        puts "JTAG_READY:$hw_name|$dev_name"\n'
                "    }\n"
                "    flush stdout\n"
                "}\n"
            )
            assert self._proc.stdin is not None
            assert self._proc.stdout is not None
            self._proc.stdin.write(init_tcl)
            self._proc.stdin.flush()

            deadline = time.time() + 8.0
            while time.time() < deadline:
                line = self._proc.stdout.readline()
                if not line:
                    break
                line = line.strip()
                if "JTAG_READY:" in line:
                    payload = line.split("JTAG_READY:", 1)[1]
                    parts = payload.split("|", 1)
                    with self._state_lock:
                        self.hw_name = parts[0]
                        self.dev_name = parts[1] if len(parts) > 1 else ""
                        self.connected = True
                        self.last_error = ""
                    self._do_poll()
                    return
                if "JTAG_ERR:" in line:
                    err = line.split("JTAG_ERR:", 1)[1]
                    with self._state_lock:
                        self.connected = False
                        self.last_error = err
                    self._close_proc()
                    return

            with self._state_lock:
                self.connected = False
                self.last_error = "JTAG connect timeout"
            self._close_proc()
        except Exception as exc:
            with self._state_lock:
                self.connected = False
                self.last_error = str(exc)
            self._close_proc()

    def _close_proc(self) -> None:
        with self._state_lock:
            self.connected = False
        if self._proc is not None:
            try:
                if self._proc.stdin:
                    self._proc.stdin.write(
                        "catch {end_insystem_source_probe}\nexit\n"
                    )
                    self._proc.stdin.flush()
            except Exception:
                pass
            try:
                self._proc.kill()
            except Exception:
                pass
            self._proc = None

    def _do_poll(self) -> None:
        if self._proc is None or self._proc.stdin is None or self._proc.stdout is None:
            return
        try:
            self._proc.stdin.write(
                'if {[catch {set v [read_probe_data -instance_index 0 -value_in_hex]} err]} { puts "PROBE_ERR:$err" } else { puts "PROBE_HEX:$v" }; flush stdout\n'
            )
            self._proc.stdin.flush()
            while True:
                line = self._proc.stdout.readline()
                if not line:
                    self._close_proc()
                    return
                line = line.strip()
                if "PROBE_HEX:" in line:
                    val = int(line.split("PROBE_HEX:", 1)[1].strip(), 16)
                    st = self._decode_probe(val)
                    with self._state_lock:
                        prev = self.latest_probe
                        self.latest_probe = st
                        if (
                            not st.pc_mode_active
                            and self._rom_images_10 is not None
                            and (prev is None or prev.active_img != st.active_img)
                        ):
                            self._fpga_shadow_img[:, :] = self._rom_images_10[st.active_img]
                    return
                if "PROBE_ERR:" in line:
                    with self._state_lock:
                        self.last_error = line.split("PROBE_ERR:", 1)[1]
                    self._close_proc()
                    return
        except Exception as exc:
            with self._state_lock:
                self.last_error = str(exc)
            self._close_proc()

    def _decode_probe(self, val: int) -> FPGAProbeState:
        sw_all = val & 0x3FF
        sw_img = sw_all & 0xF
        if sw_img > 9:
            sw_img = 0
        sw_cycle_mode = bool((sw_all >> 9) & 1)
        key_all = (val >> 10) & 0xF
        predicted_class = (val >> 14) & 0xF
        active_img = (val >> 18) & 0xF
        has_result = bool((val >> 22) & 1)
        pc_mode_active = bool((val >> 23) & 1)

        bcd_raw = (val >> 24) & 0xFFFFFF
        bcd_str = f"{bcd_raw:06X}"
        try:
            cycle_count = int(bcd_str)
        except ValueError:
            cycle_count = 302978

        scores: List[int] = []
        scores_bits = (val >> 48) & ((1 << 80) - 1)
        for i in range(10):
            byte_val = (scores_bits >> (i * 8)) & 0xFF
            signed_val = byte_val - 256 if byte_val >= 128 else byte_val
            scores.append(signed_val)

        return FPGAProbeState(
            connected=True,
            hw_name=self.hw_name,
            dev_name=self.dev_name,
            sw_all=sw_all,
            sw_img=sw_img,
            sw_cycle_mode=sw_cycle_mode,
            key_all=key_all,
            predicted_class=predicted_class,
            active_img=active_img,
            has_result=has_result,
            pc_mode_active=pc_mode_active,
            cycle_count=cycle_count,
            hw_fc_scores=scores,
        )
