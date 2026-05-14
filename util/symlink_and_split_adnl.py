#!/usr/bin/env python3
import random
import os
from pathlib import Path

# SRC_ROOT = Path("/Users/logan/Developer/vibes/WORK/LIPLA/LPDGAN/dataset")
SRC_ROOT = Path("/mnt/data/nblong-t04/LPDGAN/dataset")

# ── adnl: andan + ninhloc ─────────────────────────────────────────
ADNL_ROOT = SRC_ROOT / "adnl"
for split in ["train", "test"]:
    (ADNL_ROOT / split / "sharp").mkdir(parents=True, exist_ok=True)

andan_ninhloc = []
for name in ["andan", "ninhloc"]:
    for subdir in SRC_ROOT.glob(f"{name}/*/"):
        for img in subdir.glob("*.jpg"):
            andan_ninhloc.append((name, subdir.name, img.name, img))

random.seed(42)
random.shuffle(andan_ninhloc)
split_idx = int(len(andan_ninhloc) * 0.9)
adnl_train = andan_ninhloc[:split_idx]
adnl_test  = andan_ninhloc[split_idx:]

def symlink_adnl(images, split):
    dst = ADNL_ROOT / split / "sharp"
    for name, subdir_name, filename, img_path in images:
        link_name = f"{name}-{subdir_name}-{filename}"
        target = os.path.relpath(img_path, dst)
        link = dst / link_name
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(target)

symlink_adnl(adnl_train, "train")
symlink_adnl(adnl_test,  "test")
print(f"adnl: train={len(adnl_train)}, test={len(adnl_test)}")

# ── LPBlur: separate train/test split ───────────────────────────────
LPBLUR_ROOT = SRC_ROOT / "LPBlur"
for split in ["train", "test"]:
    for variant in ["sharp", "blur"]:
        (LPBLUR_ROOT / split / variant).mkdir(parents=True, exist_ok=True)

# Collect by variant
for variant in ["sharp", "blur"]:
    images = []
    for subdir in SRC_ROOT.glob(f"LPBlur/{variant}/"):
        for img in subdir.glob("*.jpg"):
            images.append((variant, subdir.name, img.name, img))

    random.seed(42)
    random.shuffle(images)
    split_idx = int(len(images) * 0.9)
    lp_train = images[:split_idx]
    lp_test  = images[split_idx:]

    def symlink_lpblur(images, split, variant):
        dst = LPBLUR_ROOT / split / variant
        for variant, _, filename, img_path in images:
            link_name = filename  # just the filename, no LPBlur/blur/sharp prefix
            target = os.path.relpath(img_path, dst)
            link = dst / link_name
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(target)

    symlink_lpblur(lp_train, "train", variant)
    symlink_lpblur(lp_test,  "test",  variant)
    print(f"LPBlur/{variant}: train={len(lp_train)}, test={len(lp_test)}")

print("\nDone.")
print(f"  adnl/train/sharp/{'*'}: {len(list((ADNL_ROOT/'train'/'sharp').iterdir()))}")
print(f"  adnl/test/sharp/{'*'}:  {len(list((ADNL_ROOT/'test'/'sharp').iterdir()))}")
print(f"  LPBlur/train/sharp/{'*'}: {len(list((LPBLUR_ROOT/'train'/'sharp').iterdir()))}")
print(f"  LPBlur/train/blur/{'*'}:  {len(list((LPBLUR_ROOT/'train'/'blur').iterdir()))}")
print(f"  LPBlur/test/sharp/{'*'}:  {len(list((LPBLUR_ROOT/'test'/'sharp').iterdir()))}")
print(f"  LPBlur/test/blur/{'*'}:   {len(list((LPBLUR_ROOT/'test'/'blur').iterdir()))}")