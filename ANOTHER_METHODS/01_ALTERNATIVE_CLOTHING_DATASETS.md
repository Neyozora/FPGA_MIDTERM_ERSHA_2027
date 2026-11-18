# Alternative Clothing Datasets for LSI Design Contest 2027 (`32×32`)

In **Method 1**, we used **Fashion-MNIST** (`28×28` grayscale padded to `32×32`). While Fashion-MNIST is great for hardware verification, its low resolution and monochrome silhouettes make `Shirt`, `Coat`, and `Pullover` hard to distinguish, and real Pinterest/camera photos look different from studio-centered Fashion-MNIST silhouettes.

Below are **4 better clothing datasets** you can use for **Method 2+**, along with exact preprocessing rules to keep them **LSI 2027 (`32×32`) compliant**.

---

## 1. Recommended Clothing Datasets

| Dataset | Raw Size & Color | Classes | Why Use It for Method 2? |
| :--- | :---: | :---: | :--- |
| **1. Apparel Images Dataset (Kaggle)** | `~200×200` RGB (`~16,000` images) | `Shirt`, `Dress`, `Pants`, `Shoes`, `Shorts` (or `12–24` sub-classes) | Real-world e-commerce & Pinterest-style photos! Matches real camera/uploaded photos far better than Fashion-MNIST. |
| **2. Fashion Product Images Small (Kaggle)** | `60×80` RGB (`44,441` images) | `Topwear`, `Bottomwear`, `Footwear`, `Bags`, `Watches`, `Eyewear` | Huge, clean dataset with rich metadata (`masterCategory`, `subCategory`, `articleType`). Easy to filter to `10` balanced clothing classes! |
| **3. DeepFashion (Category & Attribute Prediction Subset)** | Variable RGB (`200k+` images) | `50` clothing categories (`Tee`, `Jacket`, `Jeans`, `Sneakers`, etc.) | Industry-standard academic fashion benchmark; selecting the top `10` classes gives realistic textures and collars/zippers. |
| **4. Zalando / Custom Pinterest Scraped + Fashion-MNIST Hybrid** | Mixed RGB + Grayscale | Same `10` Fashion-MNIST classes | **Best for keeping 100% compatibility with Method 1!** Mixes Fashion-MNIST with augmented real photos so your Pinterest uploader works much more accurately! |

---

## 2. How to Preprocess Real Clothing Photos to LSI 2027 `32×32` Format

### Option A: `32×32×1` Grayscale (Drop-In Compatible with Method 1 BRAM)
1. **Center-Crop & Pad to Square:** Preserve aspect ratio so tall dresses or wide bags are never squished.
2. **Background Suppression (Optional):** For white-background e-commerce photos, invert (`255 - gray`) or use adaptive thresholding so the background becomes `0` (`black`), matching Fashion-MNIST polarity.
3. **Resize to `32×32` (`LANCZOS`):** No need for `28×28 + 2px pad` on native high-res datasets—resize directly to `32×32` (`1,024` bytes = `1` M10K BRAM).
4. **Quantize:** `q_in = pixel >> 1` (`INT8 [0, 127]`, zero hardware gates).

### Option B: `32×32×3` RGB (For Higher Accuracy on Real-World Clothes)
If you want the CNN to tell a **blue denim jacket** apart from a **white dress shirt**, color matters!
- Input tensor becomes **`3 × 32 × 32` (`3,072` bytes = `3` M10K BRAM blocks)**.
- Quantize each R, G, B channel with `pixel >> 1` (`INT8 [0, 127]`).
- `Conv1` changes from `1 → 8` channels (`72` weights) to **`3 → 16` channels (`432` weights)**—still tiny enough for 100% on-chip M10K BRAM!

---

## 3. Essential Training Upgrades for Real Clothing Datasets
1. **Data Augmentation (During Training Only):**
   - `RandomHorizontalFlip(p=0.5)` (a T-shirt flipped left-right is still a T-shirt!)
   - `RandomAffine(degrees=10, translate=(0.08, 0.08), scale=(0.9, 1.1))`
   - `ColorJitter(brightness=0.2, contrast=0.2)` (makes the FPGA robust to dim/bright Pinterest photos!)
2. **Batch Normalization + Folding:**
   - Add `nn.BatchNorm2d` after `Conv1` and `Conv2` during training, then mathematically **fold** $\gamma, \beta, \mu, \sigma$ into the `Conv` weights & `INT32` biases before exporting `.mem` files—giving **`+2% to +4%` higher accuracy with ZERO extra FPGA hardware!**
