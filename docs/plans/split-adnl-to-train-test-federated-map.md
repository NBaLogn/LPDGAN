# Split ADNL to train/test

## Context

ADNL dataset has 20,718 symlinks in `dataset/adnl/`. User wants 19,000 for training, remaining for test.

## Plan

### Step 1: Create directory structure

```
dataset/adnl/
  train/
    plate_info.txt
  test/
```

### Step 2: Split symlinks

- 19,000 → `dataset/adnl/train/`
- 1,718 → `dataset/adnl/test/`

Use file listing sorted deterministically. Move symlinks (not copy) to preserve storage.

```bash
cd dataset/adnl
ls | sort > /tmp/adnl_sorted.txt

head -19000 /tmp/adnl_sorted.txt | while read f; do
  mv "$f" train/
done

mv *.jpg test/ 2>/dev/null || true
```

### Step 3: Generate plate_info.txt

Train set needs `plate_info.txt` mapping filenames to plate numbers. Extract from symlink names (format: `{dataset}-{subdir}-{plate_number}.jpg`).

Parse `{plate_number}` from symlink filename pattern. Write to `train/plate_info.txt` with format:
```
filename|plate_number
```

### Step 4: Verify

- `ls dataset/adnl/train/ | wc -l` → 19000
- `ls dataset/adnl/test/ | wc -l` → 1718
- `wc -l dataset/adnl/train/plate_info.txt` → 19000

## Files touched

- `dataset/adnl/train/` (new dir + symlinks moved)
- `dataset/adnl/test/` (new dir + symlinks moved)
- `dataset/adnl/train/plate_info.txt` (generated)
- `scripts/symlink_adnl.sh` (update to recreate split)

## Verification

Run after script:
```bash
ls dataset/adnl/train/ | wc -l
ls dataset/adnl/test/ | wc -l
wc -l dataset/adnl/train/plate_info.txt
```