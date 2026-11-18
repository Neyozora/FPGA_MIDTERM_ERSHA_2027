"""
Download and preprocess CIFAR-10 (32x32x3 RGB) for Method 2.
Creates:
  - 45,000 train / 5,000 val (stratified, SEED=42) / 10,000 test
  - 100 balanced hardware test vectors (10 per class, first 10 of each class in test set)
  - Input quantization: q_in = int8(pixel - 128) in [-128, 127] (MSB-flip in hardware)
"""
import json
import numpy as np
import torchvision

from sw_m2.config import (
    RAW_CIFAR_DIR,
    PROCESSED_M2_DIR,
    SPLITS_M2_DIR,
    REPORTS_M2_DIR,
    NUM_CLASSES,
    CLASS_NAMES,
    SEED,
    VAL_PER_CLASS,
    TEST_VECTORS_PER_CLASS,
)


def preprocess_cifar10():
    RAW_CIFAR_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_M2_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_M2_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_M2_DIR.mkdir(parents=True, exist_ok=True)

    train_ds = torchvision.datasets.CIFAR10(root=str(RAW_CIFAR_DIR), train=True, download=True)
    test_ds = torchvision.datasets.CIFAR10(root=str(RAW_CIFAR_DIR), train=False, download=True)

    # train_ds.data is (50000, 32, 32, 3) uint8; convert to NCHW (N, 3, 32, 32)
    train_imgs_u8 = np.transpose(train_ds.data, (0, 3, 1, 2)).copy()
    train_labels = np.array(train_ds.targets, dtype=np.int64)

    test_imgs_u8 = np.transpose(test_ds.data, (0, 3, 1, 2)).copy()
    test_labels = np.array(test_ds.targets, dtype=np.int64)

    # Stratified 45k train / 5k val split
    rng = np.random.default_rng(SEED)
    train_idx_list, val_idx_list = [], []
    for c in range(NUM_CLASSES):
        c_idx = np.where(train_labels == c)[0]
        rng.shuffle(c_idx)
        val_idx_list.append(c_idx[:VAL_PER_CLASS])
        train_idx_list.append(c_idx[VAL_PER_CLASS:])

    train_idx = np.sort(np.concatenate(train_idx_list))
    val_idx = np.sort(np.concatenate(val_idx_list))

    np.savez(SPLITS_M2_DIR / "cifar10_split.npz", train_idx=train_idx, val_idx=val_idx)

    # Quantize to INT8 [-128, 127] via MSB flip: pixel - 128
    def to_int8(u8_arr):
        return (u8_arr.astype(np.int16) - 128).astype(np.int8)

    x_train_q = to_int8(train_imgs_u8[train_idx])
    y_train = train_labels[train_idx]

    x_val_q = to_int8(train_imgs_u8[val_idx])
    y_val = train_labels[val_idx]

    x_test_q = to_int8(test_imgs_u8)
    y_test = test_labels

    # 100 balanced hardware test vectors (first 10 of each class in official test set)
    hw_indices = []
    for c in range(NUM_CLASSES):
        c_test = np.where(test_labels == c)[0][:TEST_VECTORS_PER_CLASS]
        hw_indices.append(c_test)
    hw_indices = np.concatenate(hw_indices)
    x_hw_q = x_test_q[hw_indices]
    y_hw = y_test[hw_indices]

    np.savez_compressed(
        PROCESSED_M2_DIR / "cifar10_int8.npz",
        x_train_q=x_train_q,
        y_train=y_train,
        x_val_q=x_val_q,
        y_val=y_val,
        x_test_q=x_test_q,
        y_test=y_test,
        x_hw_q=x_hw_q,
        y_hw=y_hw,
        hw_indices=hw_indices,
    )

    summary = {
        "dataset": "CIFAR-10",
        "shape": [3, 32, 32],
        "train_count": int(len(y_train)),
        "val_count": int(len(y_val)),
        "test_count": int(len(y_test)),
        "hw_test_vectors": int(len(y_hw)),
        "input_quant": "int8(pixel - 128) in [-128, 127], scale=1/128",
        "class_names": CLASS_NAMES,
    }
    (REPORTS_M2_DIR / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"CIFAR-10 preprocessed: train={x_train_q.shape}, val={x_val_q.shape}, test={x_test_q.shape}, hw={x_hw_q.shape}")


if __name__ == "__main__":
    preprocess_cifar10()
