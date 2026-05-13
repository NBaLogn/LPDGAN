# Plan: Symlink dataset images to flat adnl + LPBlur structures, split into train/test

## Context
Three source datasets (`andan`, `ninhloc`, `LPBlur`) with different subdir structures.
Final structure: `adnl/[train|test]/sharp/` for andan+ninhloc, `LPBlur/[train|test]/[sharp|blur]/` for LPBlur.

## Source datasets
| Dataset | Source structure | Images |
|---------|------------------|--------|
| `andan` | `andan/00-23/*.jpg` | 10,532 |
| `ninhloc` | `ninhloc/00-23/*.jpg` | 10,186 |
| `LPBlur` | `LPBlur/sharp/*.jpg`, `LPBlur/blur/*.jpg` | 10,408 each |

## Final structure
```
dataset/
  adnl/
    train/sharp/  ← andan + ninhloc symlinks, naming: [dataset]-[subdir]-[image].jpg (18,646)
    test/sharp/   ← andan + ninhloc symlinks (2,072)
  LPBlur/
    train/sharp/  ← LPBlur/sharp symlinks, naming: [image].jpg (9,367)
    train/blur/   ← LPBlur/blur symlinks, naming: [image].jpg (9,367)
    test/sharp/   ← LPBlur/sharp symlinks (1,041)
    test/blur/    ← LPBlur/blur symlinks (1,041)
```

## Approach
Python script `scripts/symlink_and_split_adnl.py`:

1. **adnl**: collect andan + ninhloc → shuffle with `seed(42)` → 90/10 split → symlinks in `adnl/[train|test]/sharp/`
2. **LPBlur**: separately for each variant (sharp/blur) → shuffle with `seed(42)` → 90/10 split → symlinks in `LPBlur/[train|test]/[sharp|blur]/`

Non-destructive: existing symlinks are replaced if they exist.

## Verification
```bash
ls dataset/adnl/train/sharp/ | wc -l   # 18646
ls dataset/adnl/test/sharp/  | wc -l   # 2072
ls dataset/LPBlur/train/sharp/ | wc -l  # 9367
ls dataset/LPBlur/train/blur/  | wc -l  # 9367
ls dataset/LPBlur/test/sharp/  | wc -l  # 1041
ls dataset/LPBlur/test/blur/   | wc -l  # 1041
file dataset/adnl/train/sharp/andan-00-L1_Lpn_20251012000033280.jpg
file dataset/LPBlur/train/sharp/grab10.jpg   # LPBlur: naming is just [image].jpg, not LPBlur-sharp-[image].jpg
```