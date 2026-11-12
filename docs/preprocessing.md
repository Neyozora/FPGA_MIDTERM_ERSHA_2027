# Dataset Preprocessing Specification

## Source

- Fashion-MNIST, Zalando Research: https://github.com/zalandoresearch/fashion-mnist
- Four gzipped IDX files are downloaded by `python -m sw.data.download`. The
  first URL tried is the official GitHub repository, the second is the
  official S3 mirror.
- Each file is checked against the MD5 published in the official README:

| File | MD5 |
|---|---|
| train-images-idx3-ubyte.gz | 8d4fb7e6c68d591d4c3dfef9ec88bf0d |
| train-labels-idx1-ubyte.gz | 25c81989df183df01b3e8a0aad5dffbe |
| t10k-images-idx3-ubyte.gz | bef4ecab320f06d8554ea6380940ec79 |
| t10k-labels-idx1-ubyte.gz | bb300cfdad3c16e7a12a480ee83cd310 |

The IDX files are parsed by our own reader, [`sw/data/idx.py`](../sw/data/idx.py).
It does not use torchvision, so this step is transparent and does not depend
on PyTorch. The reader checks the magic bytes, the dimension header and the
payload length.

## Splits

| Split | Source | Size | Selection |
|---|---|---|---|
| train | official train set | 54,000 | everything not in val |
| val | official train set | 6,000 | 600 per class, `numpy.random.default_rng(42)`, without replacement |
| test | official test set (`t10k`) | 10,000 | unchanged, used only for final evaluation |

- The split indices are saved to `data/splits/split_seed42.npz` and reused on
  every later run. Commit this file so the split stays the same even if
  NumPy's RNG stream changes in a future version.
- `data/processed/manifest.json` records a SHA-256 fingerprint of the indices
  and of every processed array.

## Pixel → INT8 input conversion

$$
q_{in} = p \gg 1 = \left\lfloor p / 2 \right\rfloor, \quad p \in [0,255] \Rightarrow q_{in} \in [0,127]
$$

$$
x = q_{in} \cdot s_{in}, \quad s_{in} = 2^{-7} = 1/128, \quad z_{in} = 0
$$

- `q_in` is the signed INT8 value that the FPGA reads from `test_image.mem`.
- `x` is the value the floating-point model is trained on. Because `s_in` is
  a power of two, `x` is exact in float32, so the FP32/QAT model and the
  hardware see the same input. There is no separate float-only normalization.
- Implemented in [`sw/data/preprocess.py`](../sw/data/preprocess.py). Every
  later stage must call `pixels_to_int8()` / `int8_to_float()` rather than
  re-implementing the conversion.

### Why this convention

| Option | Pros | Cons | Decision |
|---|---|---|---|
| `x = p/255`, then mean/std normalization | textbook | non-zero mean needs a zero point or negative inputs; the float and integer inputs would differ | rejected |
| `q = p − 128` (asymmetric, zp = −128) | keeps all 8 bits | zero-point correction terms in every conv1 MAC | rejected (complexity) |
| `q = round(p·127/255)` | uses the full INT8 range | rounding is done offline, and a later camera/stream input would need a multiplier | possible alternative |
| **`q = p >> 1`, s = 1/128** | trivial in hardware (wire shift), exact in float, zero point 0 | drops the pixel LSB (256 → 128 grey levels) | **chosen** |

Mean/std normalization would not help anyway. An affine input normalization
`x' = (x − μ)/σ` is absorbed exactly by conv1 (`w' = w/σ`,
`b' = b − μ·Σw/σ`), so the network can learn it.

The cost of dropping the LSB is reported in
`reports/dataset/dataset_report.md`. Classification accuracy is trained and
measured *on the converted data*, so there is no train/deploy mismatch.

## Hardware test vectors

`data/processed/test_vectors.npz` contains the first 10 official test images
of each class (100 images, ordered by class, then by test index). It stores:

- `test_idx`: official test-set index
- `labels`: ground-truth label
- `images_u8`: raw pixels
- `images_int8`: hardware input codes

The selection is deterministic and uses no RNG. In Milestone 5 these are
written to `test_image.mem` / `.mif`.

## Reproduce

```powershell
python -m sw.data.prepare_dataset   # download (if needed), split, convert, save
python -m sw.data.inspect_dataset   # statistics + figures -> reports/dataset/
python -m sw.tests.test_m1_data     # automated checks
```
