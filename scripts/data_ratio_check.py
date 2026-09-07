"""Verify the class (0/1) balance of the train/val/test subsamples.

Reproduces the exact subsampling used in training (seed=42, 10k/2k/2k)
so the reported class ratios can be independently validated.

Run:  python class_balance_check.py
"""
from datasets import load_dataset

SEED = 42 # must match cfg.seed in conf/config.yaml
TRAIN_N, VAL_N, TEST_N = 10000, 2000, 2000
DATASET = "dbzadnen/breast-histopathology-images"


def subsample(split, n, seed):
    """Match data.py::_subsample — shuffle(seed) then take first n."""
    return split.shuffle(seed=seed).select(range(min(n, len(split))))


def main():
    ds = load_dataset(DATASET)
    splits = {
        "train": subsample(ds["train"], TRAIN_N, SEED),
        "val":   subsample(ds["validation"], VAL_N, SEED),
        "test":  subsample(ds["test"], TEST_N, SEED),
    }
    for name, split in splits.items():
        labels = split["label"]
        n1 = sum(labels)
        n0 = len(labels) - n1
        print(f"{name:5s} | n={len(labels):5d} | "
              f"label0={n0:5d} ({n0/len(labels):.1%}) | "
              f"label1={n1:5d} ({n1/len(labels):.1%})")


if __name__ == "__main__":
    main()