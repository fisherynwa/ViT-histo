"""Tests that the validation/test transform is deterministic (no augmentation)
while the train transform is stochastic (augmentation active).

This locks down a real correctness property: the held-out test set must be
evaluated on clean, un-augmented images. If augmentation were accidentally
applied to the val/test pipeline, the reported metrics would be measuring the
wrong thing.

CI-safe: uses a synthetic PIL image, no dataset download, no GPU.
"""

import numpy as np
import torch
from PIL import Image

from vit_histo.data import _build_transforms


class _FakeProcessor:
    """Stand-in for AutoImageProcessor with the fields _build_transforms reads."""
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]
    size = {"height": 224, "width": 224}


def _dummy_image():
    """A non-symmetric random image, so flips/rotations would visibly change it."""
    arr = np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def test_val_transform_is_deterministic():
    """The val/test transform must produce identical output on repeat calls
    (i.e. no random augmentation is applied)."""
    _, val_tf = _build_transforms(_FakeProcessor())
    img = _dummy_image().convert("RGB")
    out1 = val_tf(img)
    out2 = val_tf(img)
    assert torch.equal(out1, out2), "val/test transform is NOT deterministic (augmentation leaked in?)"


def test_train_transform_is_stochastic():
    """Paired sanity check: the train transform DOES vary across calls, which
    makes the determinism test above meaningful (rules out both transforms
    being accidentally deterministic)."""
    train_tf, _ = _build_transforms(_FakeProcessor())
    torch.manual_seed(0)
    img = _dummy_image().convert("RGB")
    outs = [train_tf(img) for _ in range(8)]
    all_identical = all(torch.equal(outs[0], o) for o in outs[1:])
    assert not all_identical, "train transform did not vary — augmentation may be disabled"