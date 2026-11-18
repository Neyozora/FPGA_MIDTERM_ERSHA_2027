"""
Method 2 project-wide constants and paths (CIFAR-10 32x32x3 RGB).
Kept completely separate from Method 1 (`sw/`) for side-by-side comparison.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = REPO_ROOT / "data"
RAW_CIFAR_DIR = DATA_DIR / "raw" / "cifar10"
PROCESSED_M2_DIR = DATA_DIR / "processed_m2"
SPLITS_M2_DIR = DATA_DIR / "splits_m2"

REPORTS_M2_DIR = REPO_ROOT / "reports_m2"
CHECKPOINTS_M2_DIR = REPO_ROOT / "checkpoints_m2"
MEM_M2_DIR = REPO_ROOT / "mem_m2"
SIM_M2_DIR = REPO_ROOT / "sim_m2"
SIM_M2_VECTORS_DIR = SIM_M2_DIR / "vectors"

# Hardware Target (Terasic DE10-Standard / Intel Cyclone V SoC)
FPGA_FAMILY = "Cyclone V"
FPGA_PART_DE10_STANDARD = "5CSXFC6D6F31C6"

# Dataset constants (CIFAR-10 native 32x32x3 RGB)
IMG_H = 32
IMG_W = 32
IMG_C = 3
NUM_CLASSES = 10

CLASS_NAMES = [
    "airplane",     # 0
    "automobile",   # 1
    "bird",         # 2
    "cat",          # 3
    "deer",         # 4
    "dog",          # 5
    "frog",         # 6
    "horse",        # 7
    "ship",         # 8
    "truck",        # 9
]

NUM_TRAIN_OFFICIAL = 50_000
NUM_TEST_OFFICIAL = 10_000
SEED = 42
VAL_PER_CLASS = 500          # 500 * 10 = 5,000 val, 45,000 train
TEST_VECTORS_PER_CLASS = 10  # 100 balanced hardware test vectors

# Input representation: MSB-flipped INT8 [-128, 127] (0 logic gates in Verilog!)
# q_in = int8(pixel - 128) = {~pixel[7], pixel[6:0]}
# x = q_in * (1/128) in [-1.0, 127/128]
INPUT_FRAC_BITS = 7
INPUT_SCALE = 2.0 ** -INPUT_FRAC_BITS
INPUT_QMIN = -128
INPUT_QMAX = 127
