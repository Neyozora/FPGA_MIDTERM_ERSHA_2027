"""
Project-wide constants and paths.

Every script in the `sw` package imports paths and numeric conventions from
here so that the dataset pipeline, training, quantization, export and the
integer reference model all agree on a single source of truth.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "fashion_mnist"   # downloaded .gz IDX files (git-ignored)
PROCESSED_DIR = DATA_DIR / "processed"         # regenerated .npz files     (git-ignored)
SPLITS_DIR = DATA_DIR / "splits"               # train/val split indices    (tracked)

REPORTS_DIR = REPO_ROOT / "reports"
DATASET_REPORT_DIR = REPORTS_DIR / "dataset"
TRAINING_REPORT_DIR = REPORTS_DIR / "training"
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"

# ---------------------------------------------------------------------------
# Hardware Target (Terasic DE10-Standard / Intel Cyclone V SoC)
# ---------------------------------------------------------------------------
# Note: The physical DE10-Standard board uses 5CSXFC6D6F31C6N (41,509 ALMs,
# 576 M10Ks, 112 variable-precision DSP blocks), while the initial prompt
# specification listed 5CSEBA6U23I7 (DE10-Nano, 41,910 ALMs, 553 M10Ks,
# 112 DSP blocks). Both share the exact same Cyclone V M10K (10,240-bit) and
# variable-precision DSP (18x18 / dual 9x9 / 27x27) fabric architecture.
# Quartus synthesis (Milestone 10) is parameterized to support both parts.
FPGA_FAMILY = "Cyclone V"
FPGA_PART_DE10_STANDARD = "5CSXFC6D6F31C6"
FPGA_PART_SPEC = "5CSEBA6U23I7"
M10K_BITS = 10_240


# ---------------------------------------------------------------------------
# Dataset constants
# ---------------------------------------------------------------------------
RAW_IMG_H = 28
RAW_IMG_W = 28
INPUT_PAD = 2               # 2-pixel zero-padding on top, bottom, left, right
IMG_H = RAW_IMG_H + 2 * INPUT_PAD   # 32 (LSI Design Contest 2027 input size)
IMG_W = RAW_IMG_W + 2 * INPUT_PAD   # 32
IMG_C = 1
NUM_CLASSES = 10

CLASS_NAMES = [
    "T-shirt/top",  # 0
    "Trouser",      # 1
    "Pullover",     # 2
    "Dress",        # 3
    "Coat",         # 4
    "Sandal",       # 5
    "Shirt",        # 6
    "Sneaker",      # 7
    "Bag",          # 8
    "Ankle boot",   # 9
]

NUM_TRAIN_OFFICIAL = 60_000
NUM_TEST_OFFICIAL = 10_000

# Validation split: VAL_PER_CLASS images of every class are taken out of the
# official 60k training set (stratified, seeded). 600 x 10 = 6,000 images.
SEED = 42
VAL_PER_CLASS = 600

# Representative hardware test vectors: the first N test images of each class
# (in official test-set order -> fully deterministic, no RNG involved).
TEST_VECTORS_PER_CLASS = 10

# ---------------------------------------------------------------------------
# Input representation (THE contract between Python and the FPGA)
# ---------------------------------------------------------------------------
# Raw pixel p in [0, 255] (uint8) is converted to a signed INT8 activation
#
#       q_in = p >> INPUT_SHIFT            ->  q_in in [0, 127]
#
# and the real value the float model is trained on is defined as
#
#       x = q_in * INPUT_SCALE,  INPUT_SCALE = 2^-INPUT_FRAC_BITS = 1/128
#
# i.e. the float model sees *exactly* the values the hardware sees (up to a
# power-of-two scale, which is exact in float32). Input zero point = 0.
INPUT_SHIFT = 1
INPUT_FRAC_BITS = 7
INPUT_SCALE = 2.0 ** -INPUT_FRAC_BITS
INPUT_ZERO_POINT = 0
INPUT_QMIN = 0      # smallest value produced by p >> 1
INPUT_QMAX = 127    # largest value produced by p >> 1 (255 >> 1)
