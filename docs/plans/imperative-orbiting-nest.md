# Plan — Crop ADNL plates + classify into square / rect

## Context

`dataset/adnl/` aggregates ~20,718 full-frame 1440×1080 captures
(`dataset/adnl/{train,test}/sharp/`, symlinked from `andan/` and `ninhloc/`).
Unlike `dataset/quan_lp_dataset/` (already pre-cropped, ~200–300 px wide),
ADNL frames contain the full vehicle scene with no bounding-box annotations.

To train per-plate-type deblurring models on ADNL the way `quan_lp` already
supports (via `util/split_lp_by_shape.py`), we need to:

1. **Crop** license plates out of each full frame (no detector currently lives
   in the repo).
2. **Classify** each crop as `rect` (1-row plate) or `square` (2-row plate).
3. Land outputs at `dataset/adnl_cropped/{square,rect}/{train,test}/sharp/`
   so existing loaders (`data/LPBlur_dataset.py` family) can consume them
   without changes.

Choices already locked in by the user:
- Detector: a third-party pretrained Vietnamese-LP YOLO model
  (`plate_yolo12n_640_2025.pt` from `tungedng2710/AI-Traffic-Analysis`).
- Output layout: `dataset/adnl_cropped/{square,rect}/{train,test}/sharp/` —
  real JPEGs (not symlinks) since crops are derived.
- Classification: aspect ratio **plus** OCR row-count verification.
- Multi-plate frames: keep every detection, suffix filename with `_<idx>`.

## Approach

One new script (`util/crop_and_classify_adnl.py`) that walks ADNL, runs the
YOLO detector, crops each plate, classifies it via row-count + aspect-ratio,
and writes the result into the target layout. Reuse the PaddleOCR pattern
already established in `util/generate_plate_info.py`.

### Pipeline

```
for split in {train, test}:
  for frame in dataset/adnl/<split>/sharp/*.jpg (resolve symlink → real path):
    detections = yolo(frame, conf=CONF_THR, iou=IOU_THR)
    if not detections:        # log to _no_plate.txt, continue
       continue
    for idx, (x1,y1,x2,y2,score) in enumerate(detections):
       crop = frame[y1:y2, x1:x2]
       cls  = classify(crop)   # OCR rows + AR fallback
       out  = f"dataset/adnl_cropped/{cls}/{split}/sharp/{stem}_{idx}.jpg"
       cv2.imwrite(out, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
```

### Classification rule (`classify(crop)`)

1. `ocr_lines = paddle.ocr(crop, cls=True)` — returns `[polygon, (text, conf)]`.
2. Filter to lines with `conf >= 0.5` and `len(text) >= 2`.
3. Compute `y_center` for each surviving polygon (mean of polygon Y-coords,
   normalized by crop height).
4. **Row count**: greedy 1D cluster of `y_center` values with gap threshold
   `0.30 * crop_h` (tunable constant `ROW_GAP_FRAC`).
   - 1 cluster → `rect`
   - ≥2 clusters → `square`
5. **AR fallback** when OCR yields **zero** usable lines:
   `w / h >= 2.0` → `rect`, else `square` (mirrors
   `util/split_lp_by_shape.RECT_AR_THRESHOLD = 2.0`).
6. Log every disagreement between OCR row-count and AR rule to
   `dataset/adnl_cropped/_ambiguous.csv` for later manual review
   (columns: `stem,idx,w,h,ar,row_count,ocr_chosen,ar_chosen`).

### Detector setup

- Pretrained weights: `plate_yolo12n_640_2025.pt` from
  https://github.com/tungedng2710/AI-Traffic-Analysis  (path
  `weights/plate/plate_yolo12n_640_2025.pt`). Download once and store at
  `checkpoints/lp_detector/plate_yolo12n_640_2025.pt` (gitignored).
- Provide a one-line download helper in the script:
  - `--weights /path/to/file.pt` CLI flag (default
    `checkpoints/lp_detector/plate_yolo12n_640_2025.pt`).
  - If file is missing, print the canonical download URL and exit (do **not**
    auto-fetch — keep the script offline-safe).
- Inference: `ultralytics.YOLO(weights).predict(img, imgsz=640, conf=0.25,
  iou=0.5, device='cpu')` (or `mps` if available on M1 — `torch.backends.mps`
  is auto-detected by ultralytics).

### Files to add / modify

| Path | Change |
|------|--------|
| `util/crop_and_classify_adnl.py` | **NEW** — pipeline script |
| `requirements.txt` | **ADD** `ultralytics>=8.2.28` and `paddleocr>=2.7.0` (latter is already used implicitly by `util/generate_plate_info.py`; make it explicit) |
| `.gitignore` | **ADD** `checkpoints/lp_detector/` (binary weights shouldn't be tracked) |
| `util/scripts.sh` | **OPTIONAL** — add one line under the quan_lp split step: `uv run util/crop_and_classify_adnl.py --dataroot dataset/adnl --out dataset/adnl_cropped` (only if user wants pipeline integration) |

### CLI signature

```
uv run util/crop_and_classify_adnl.py \
  [--dataroot dataset/adnl]               # ADNL source (default)
  [--out dataset/adnl_cropped]            # output root (default)
  [--weights checkpoints/lp_detector/plate_yolo12n_640_2025.pt]
  [--conf 0.25] [--iou 0.5] [--imgsz 640]
  [--ar-threshold 2.0]
  [--row-gap-frac 0.30]
  [--device auto]                         # auto|cpu|mps
  [--batch 8]                             # YOLO batch size
  [--dry-run]                             # plan only, no I/O
  [--limit N]                             # process first N frames only (smoke test)
```

### Reused helpers (no duplication)

- `util/split_lp_by_shape.py::RECT_AR_THRESHOLD` (2.0) — import as the AR
  fallback constant so the two pipelines stay aligned.
- `util/generate_plate_info.py` — copy the PaddleOCR setup boilerplate
  (`os.environ["HOME"] = TMPDIR` workaround, `PaddleOCR(lang="en",
  use_angle_cls=True)`).

### Hardware notes (M1 Pro 16 GB, 8-core CPU)

- YOLO `plate_yolo12n_640_2025.pt` is the *nano* variant — small enough that
  CPU/MPS inference at imgsz=640 is fine.
- Expected wall-clock for full dataset (20,718 frames + ~1.3× detections):
  ~1–2 h on MPS, ~2–4 h on CPU. Use `--limit 200` for a smoke test first.
- PaddleOCR on plate crops is fast (~150–300 ms each on CPU).

## Verification

Run end-to-end on a 200-frame subset first:

```bash
# 1. Install deps in venv (no system-wide)
uv pip install ultralytics paddleocr

# 2. Drop the weights file in place
mkdir -p checkpoints/lp_detector
# manually: scp / curl plate_yolo12n_640_2025.pt -> checkpoints/lp_detector/

# 3. Smoke test (200 frames)
uv run util/crop_and_classify_adnl.py --limit 200 --device auto

# 4. Inspect counts + sample 20 crops manually
ls dataset/adnl_cropped/square/train/sharp | wc -l
ls dataset/adnl_cropped/rect/train/sharp | wc -l
open dataset/adnl_cropped/square/train/sharp/{some,sample,files}.jpg

# 5. Review ambiguous classifications
cat dataset/adnl_cropped/_ambiguous.csv | head -20

# 6. Full run
uv run util/crop_and_classify_adnl.py --device auto
```

Success criteria:
- Crop yield ≥ 0.85 × frame count (some frames may have no plate visible).
- Manual sampling: of 20 random `square/` crops ≥ 18 are visually 2-row; of
  20 random `rect/` crops ≥ 18 are visually 1-row.
- `_ambiguous.csv` ≤ 5% of total crops (otherwise tune
  `ROW_GAP_FRAC` / `AR_THRESHOLD`).
- Output layout is loader-compatible: `data/LPBlur_dataset.py` can be pointed
  at `dataset/adnl_cropped/square` or `dataset/adnl_cropped/rect` with no
  code changes.

## Out of scope

- Producing a paired `blur/` half — that's a separate step using the existing
  kernel-bank pipeline (`util/apply_disk_blur_mod.py`) once these sharp crops
  exist.
- Generating `plate_info.txt` for the new split — handled later by
  `util/generate_plate_info.py` pointed at `dataset/adnl_cropped/<type>/`.
- Training itself.
