# Upgrading to CIFAR-10 & CIFAR-100 (`32×32×3` RGB) for LSI Design Contest 2027

**CIFAR-10** and **CIFAR-100** (from Alex Krizhevsky, University of Toronto — `torchvision.datasets.CIFAR10` / `CIFAR100`) are **natively `32 × 32 × 3` RGB** images—making them an *exact* resolution match for the **30th LSI Design Contest in Hokkaido (2027)** without any zero-padding!

---

## 1. Why CIFAR-10 / CIFAR-100 Is Much Better Than Fashion-MNIST for Method 2
| Property | Method 1 (Fashion-MNIST) | Method 2 (**CIFAR-10**, Recommended) | Method 2+ (**CIFAR-100**) |
| :--- | :---: | :---: | :---: |
| **Native Resolution** | `28 × 28 × 1` (needed `2px` padding) | **`32 × 32 × 3` RGB (Native!)** | **`32 × 32 × 3` RGB (Native!)** |
| **Classes** | `10` clothing silhouettes | **`10` real-world objects:** `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck` | **`100` fine-grained classes** (or `20` coarse superclasses) |
| **Dataset Split** | `54k` train / `6k` val / `10k` test | **`45k` train / `5k` val / `10k` test** | **`45k` train / `5k` val / `10k` test** |
| **Input Memory (`img_ram`)** | `1,024` bytes (`1` M10K BRAM) | **`3,072` bytes (`3` M10K BRAMs)** | **`3,072` bytes (`3` M10K BRAMs)** |
| **Real-World Photos** | Only silhouettes | **Natural color photos** (cars, planes, cats, dogs, ships!) | Natural color photos |

---

## 2. Hardware Input Quantization for `32×32×3` RGB (`0` Gates!)
Each CIFAR-10/100 pixel has 3 channels (`R, G, B`) in `uint8 [0, 255]`:
- **Option A — Zero-Gate Shift (`s_in = 2^-7`):**
  $$q_{\text{in}}(c, r, x) = p(c, r, x) \gg 1 \in [0, 127]$$
  Requires **0 logic gates** in Verilog, fits in signed `INT8`, and stores the entire `3 × 32 × 32` RGB image in `3,072` bytes (`3` Cyclone V M10K blocks).
- **Option B — Channel-Mean Subtracted `INT8` (`[-128, 127]`):**
  Subtract `128` (`{~pixel[7], pixel[6:0]}` — just flipping the MSB bit in hardware!) so `0..255` maps to `-128..+127` centered around `0`, which helps `BatchNorm`-folded `Conv1` converge faster!

---

## 3. Recommended CNN Architecture for CIFAR-10 (`> 85%–90%` Accuracy on Cyclone V)
Because **30th LSI Design Contest in Hokkaido (2027)** allows **any number of CNN layers**, a 2-layer Fashion-MNIST CNN is too shallow for RGB CIFAR-10. Use a **3- or 4-layer VGG-Tiny / ResNet-Tiny** architecture:
1. **`Conv1` (`3×3`, `3 → 16`, PAD=1) + `BN` + `ReLU`** $\rightarrow$ `16 × 32 × 32`
2. **`Conv2` (`3×3`, `16 → 32`, PAD=1) + `BN` + `ReLU` + `MaxPool(2×2)`** $\rightarrow$ `32 × 16 × 16`
3. **`Conv3` (`3×3`, `32 → 64`, PAD=1) + `BN` + `ReLU` + `MaxPool(2×2)`** $\rightarrow$ `64 × 8 × 8`
4. **`Conv4` (`3×3`, `64 → 64`, PAD=1) + `BN` + `ReLU`** $\rightarrow$ `64 × 8 × 8`
5. **`GAP (8×8)`** $\rightarrow$ `64` features $\rightarrow$ **`FC (64 → 10)`** (or `64 → 100` for CIFAR-100)

*With **BatchNorm folding** + **Per-Channel `INT8` QAT** + **Data Augmentation** (`RandomCrop(32, padding=4)` + `RandomHorizontalFlip()`), this compact architecture achieves **`86%–90%` INT8 test accuracy on CIFAR-10** while still fitting inside Cyclone V on-chip M10K BRAM (`< 15%` of `553` M10Ks)!*
