import matplotlib.pyplot as plt
from datasets import load_dataset
from torchvision import transforms
import os

os.makedirs("figures", exist_ok=True)

# --- load and SHUFFLE (handles class-sorted datasets) ---
ds = load_dataset("dbzadnen/breast-histopathology-images", split="train").shuffle(seed=42)

# --- find first 6 tumor (label==1) patches ---
pos_idxs = []
for i in range(6000):
    if ds[i]["label"] == 1:
        pos_idxs.append(i)
    if len(pos_idxs) == 6:
        break
assert len(pos_idxs) >= 1, "No label==1 patches found — check label field/value"

# ============================================================
# FIGURE 1: row of tumor patches
# ============================================================
n = len(pos_idxs)
fig, axes = plt.subplots(1, n, figsize=(2.2*n, 2.6))
for ax, idx in zip(axes, pos_idxs):
    ax.imshow(ds[idx]["image"])
    ax.set_title("label = 1", fontsize=9)
    ax.axis("off")
plt.suptitle("Tumor patches (label = 1)", fontsize=12)
plt.tight_layout()
plt.savefig("label1_samples.png", dpi=150, bbox_inches="tight")
plt.show()

# ============================================================
# FIGURE 2: one tumor patch + its augmentations
# ============================================================
img = ds[pos_idxs[0]]["image"].convert("RGB")

aug = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomChoice([
        transforms.RandomRotation((0, 0)),
        transforms.RandomRotation((90, 90)),
        transforms.RandomRotation((180, 180)),
        transforms.RandomRotation((270, 270)),
    ]),
])

m = 6
fig, axes = plt.subplots(1, m, figsize=(2.2*m, 2.6))
axes[0].imshow(img); axes[0].set_title("original", fontsize=9); axes[0].axis("off")
for i in range(1, m):
    axes[i].imshow(aug(img))
    axes[i].set_title(f"aug {i}", fontsize=9); axes[i].axis("off")
plt.suptitle("Augmentation of a tumor patch (flips + 90° rotations)", fontsize=12)
plt.tight_layout()
plt.savefig("augmentation_demo.png", dpi=150, bbox_inches="tight")
plt.show()

plt.savefig("figures/label1_samples.png", dpi=150, bbox_inches="tight")
plt.savefig("figures/augmentation_demo.png", dpi=150, bbox_inches="tight")

print("Saved: figures/label1_samples.png, figures/augmentation_demo.png")