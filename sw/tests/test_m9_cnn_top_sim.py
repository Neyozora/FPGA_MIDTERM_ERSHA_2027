"""
Milestone 9 automated tests: Verify complete CNN integration testbench
(`sim/tb/tb_cnn_top.v`), `.do` script (`sim/do/run_tb_cnn_top.do`), and the
ModelSim 18.1 end-to-end simulation transcript (`reports/simulation/cnn_top_transcript.log`).

Run:
    python -m sw.tests.test_m9_cnn_top_sim
    python -m pytest sw/tests/test_m9_cnn_top_sim.py
"""
import re
import sys

from sw.config import REPO_ROOT
from sw.tests._runner import run_all

TB_FILE = REPO_ROOT / "sim" / "tb" / "tb_cnn_top.v"
DO_FILE = REPO_ROOT / "sim" / "do" / "run_tb_cnn_top.do"
TRANSCRIPT_LOG = REPO_ROOT / "reports" / "simulation" / "cnn_top_transcript.log"
REPORT_MD = REPO_ROOT / "reports" / "simulation" / "cnn_top_verification_report.md"


def test_cnn_top_testbench_and_do_script_exist():
    """Verify `tb_cnn_top.v`, `run_tb_cnn_top.do`, and `cnn_top_verification_report.md` exist."""
    assert TB_FILE.is_file(), f"Missing {TB_FILE}"
    assert DO_FILE.is_file(), f"Missing {DO_FILE}"
    assert REPORT_MD.is_file(), f"Missing {REPORT_MD}"

    tb_text = TB_FILE.read_text(encoding="utf-8")
    assert "cnn_top dut" in tb_text
    assert "img0_relu1_out.mem" in tb_text
    assert "batch100_expected_scores.mem" in tb_text


def test_cnn_top_modelsim_transcript_bit_exact_and_cycle_count():
    """Verify ModelSim 18.1 `cnn_top_transcript.log` shows 0 mismatches across all 11,731 Image 0 elements and 100% agreement across all 10 classes."""
    assert TRANSCRIPT_LOG.is_file(), f"Missing {TRANSCRIPT_LOG} — run ModelSim first"
    log_text = TRANSCRIPT_LOG.read_text(encoding="utf-8")

    assert "[FAIL]" not in log_text
    assert "[TB_CNN_TOP] STATUS: PASS" in log_text
    assert "[TB_CNN_TOP] Image 0 Intermediate Checks : 11731 / 11731 matched" in log_text
    assert "[TB_CNN_TOP] RTL vs Python Ref Agreement : 10 / 10 (100.0%)" in log_text
    assert "[TB_CNN_TOP] Score Mismatches            : 0 | Max Abs Error: 0" in log_text
    assert re.search(r"\[TB_CNN_TOP\] Measured Cycles per Image\s*:\s*302978 cycles", log_text)


if __name__ == "__main__":
    sys.exit(run_all(globals()))
