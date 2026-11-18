"""
Milestone 8 automated tests: Verify all 6 individual ModelSim unit testbenches
(`sim/tb/tb_*.v`), `.do` scripts (`sim/do/*.do`), and the ModelSim simulation
transcript (`reports/simulation/unit_tests_transcript.log`).

Run:
    python -m sw.tests.test_m8_unit_sim
    python -m pytest sw/tests/test_m8_unit_sim.py
"""
import re
import sys

from sw.config import REPO_ROOT
from sw.tests._runner import run_all

TB_DIR = REPO_ROOT / "sim" / "tb"
DO_DIR = REPO_ROOT / "sim" / "do"
SIM_REPORT_DIR = REPO_ROOT / "reports" / "simulation"
TRANSCRIPT_LOG = SIM_REPORT_DIR / "unit_tests_transcript.log"
UNIT_REPORT_MD = SIM_REPORT_DIR / "unit_verification_report.md"

UNIT_TBS = [
    ("tb_relu.v", "run_tb_relu.do", "[TB_RELU] STATUS: PASS"),
    ("tb_maxpool.v", "run_tb_maxpool.do", "[TB_MAXPOOL] STATUS: PASS"),
    ("tb_gap.v", "run_tb_gap.do", "[TB_GAP] STATUS: PASS"),
    ("tb_fully_connected.v", "run_tb_fully_connected.do", "[TB_FC] STATUS: PASS"),
    ("tb_argmax.v", "run_tb_argmax.do", "[TB_ARGMAX] STATUS: PASS"),
    ("tb_conv_engine.v", "run_tb_conv_engine.do", "[TB_CONV_ENGINE] STATUS: PASS"),
]


def test_all_unit_testbenches_and_do_scripts_exist():
    """Verify all 6 unit testbenches and their corresponding ModelSim .do scripts exist."""
    assert (DO_DIR / "run_all_unit_tbs.do").is_file()
    for tb_file, do_file, _ in UNIT_TBS:
        tb_path = TB_DIR / tb_file
        do_path = DO_DIR / do_file
        assert tb_path.is_file(), f"Missing unit testbench {tb_path}"
        assert do_path.is_file(), f"Missing .do script {do_path}"
        tb_text = tb_path.read_text(encoding="utf-8")
        assert "$readmemh" in tb_text, f"{tb_file} must load golden test vectors via $readmemh"
        assert "STATUS: PASS" in tb_text and "STATUS: FAIL" in tb_text


def test_modelsim_unit_transcript_all_six_passed_zero_errors():
    """Verify the recorded ModelSim 18.1 transcript confirms PASS and 0 mismatches for all 6 modules."""
    assert TRANSCRIPT_LOG.is_file(), f"Missing {TRANSCRIPT_LOG} — run ModelSim first"
    assert UNIT_REPORT_MD.is_file(), f"Missing {UNIT_REPORT_MD}"

    log_text = TRANSCRIPT_LOG.read_text(encoding="utf-8")

    # No [FAIL] lines anywhere in the transcript
    assert "[FAIL]" not in log_text, "Found [FAIL] in ModelSim unit test transcript"

    # All 6 unit testbenches must report STATUS: PASS
    for _, _, pass_banner in UNIT_TBS:
        assert pass_banner in log_text, f"Missing `{pass_banner}` in {TRANSCRIPT_LOG}"

    # Verify exact cycle counts and output counts from ModelSim transcript
    assert re.search(r"\[TB_RELU\] Checks: 10160\b", log_text)
    assert re.search(r"\[TB_MAXPOOL\] Pool1 Cycles: 10801 \| Max Abs Error: 0 \| Mismatches: 0", log_text)
    assert re.search(r"\[TB_GAP\] Channels Checked: 16 \| Cycles: 2753 \| Max Abs Error: 0 \| Mismatches: 0", log_text)
    assert re.search(r"\[TB_FC\] Classes Checked: 10 \| Cycles: 191 \| Max Abs Error: 0 \| Mismatches: 0", log_text)
    assert re.search(r"\[TB_ARGMAX\] Cases Checked: 4 \| Cycles/Run: 13 \| Mismatches: 0", log_text)
    assert re.search(r"\[TB_CONV_ENGINE\] Conv1: 7200 outputs \| Cycles: 86401 \| Max Abs Err: 0 \| Mismatches: 0", log_text)
    assert re.search(r"\[TB_CONV_ENGINE\] Conv2: 2704 outputs \| Cycles: 202801 \| Max Abs Err: 0 \| Mismatches: 0", log_text)


if __name__ == "__main__":
    sys.exit(run_all(globals()))
