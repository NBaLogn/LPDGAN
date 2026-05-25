# Extract License Plates from `dataset/adnl`

## Context

`dataset/adnl/{train,test}/sharp/` holds **20,718 raw vehicle photos** (18,646 train +
2,072 test) — dark Vietnamese night-time ANPR camera captures at 1440×1080, *not*
license-plate crops. They were assembled by `util/symlink_and_split_adnl.py` from
`dataset/andan/` + `dataset/ninhloc/` (90/10 split, seed 42).

LPDGAN is a license-plate **deblurring** GAN. It has **no detector** and assumes its
inputs are already-cropped plate images. There is no detection or cropping code
anywhere in the repo or venv — `dataset/quan_lp`'s `*_crop_0.jpg` plates were cut by an
external tool not present here.

**Goal:** add one standalone util script that, for every adnl photo, detects the plate,
crops it, classifies it as `square` (2-line) or `rect` (1-line), and writes the crop into
the LPDGAN-ready layout — so the crops are immediately reusable by the deblur pipeline.

## Approach

New script **`util/extract_lp_from_vehicle.py`** — argparse CLI, run via
`uv run util/extract_lp_from_vehicle.py`. Per-image pipeline:

```
load photo -> YOLO LP detect -> pick best box -> pad -> crop -> classify (AR) -> save
```

- **Detector:** pretrained YOLO LP model from the `open-image-models` PyPI package
  (`LicensePlateDetector`, YOLOv9 ONNX, runs on `onnxruntime` — no conflict with the
  project's `torch>=2.5.1`). Weights auto-download on first construction.
- **Classification:** reuse the existing rule — aspect ratio `width/height >= 2.0` →
  `rect`, else `square`.
- **Output layout:** `dataset/adnl/{square,rect}/{train,test}/sharp/{basename}.jpg` —
  mirrors what `util/split_lp_by_shape.py` produces and what `data/LPBlur_dataset.py`
  expects.

## Reuse existing code

| What | Where |
|------|-------|
| AR classification rule + threshold | `util/split_lp_by_shape.py:38` (`RECT_AR_THRESHOLD = 2.0`), `:75-89` (`classify`) |
| CLI / `cv2` / `Path` script template | `util/apply_disk_blur_mod.py` |
| Docstring, helper layout, `format_summary`, anomaly list | `util/split_lp_by_shape.py:142-177` |
| Output dir layout (`{type}/{mode}/sharp`) | `util/split_lp_by_shape.py:129-139` |

Source filenames are already globally unique (`andan-16-L5_Lpn_…jpg`,
`util/symlink_and_split_adnl.py:29`), so output crops will not collide across folders.

## New dependency

```bash
uv add open-image-models      # pulls onnxruntime (CPU); confirmed on PyPI v0.5.1
# optional, only for --gpu:  uv add onnxruntime-gpu
```

**One-time weights download caveat:** the first run downloads the ONNX detector weights
and needs general internet. The dev sandbox whitelists only `pypi.org`, so run the first
invocation with the sandbox disabled (or pre-warm the weight cache once on an
unrestricted network). All later runs are fully offline.

## File to create: `util/extract_lp_from_vehicle.py`

### CLI arguments

| Arg | Default | Purpose |
|-----|---------|---------|
| `--dataroot` | `dataset/adnl` | Root holding `{split}/sharp/*.jpg` |
| `--splits` | `train test` | `nargs="+"` — splits to process |
| `--model` | `yolo-v9-t-640-license-plate-end2end` | `open-image-models` model id (640-input — see note) |
| `--conf` | `0.25` | Min detection confidence to accept |
| `--padding` | `0.08` | Fractional bbox padding per side |
| `--ar-threshold` | `2.0` | `w/h >= t` → rect (reuses `RECT_AR_THRESHOLD`) |
| `--limit` | `None` | Process first N images per split (testing) |
| `--dry-run` | flag | Detect + classify + summarize, write nothing |
| `--overwrite` | flag | Re-crop even if output JPEG exists (default: skip) |
| `--gpu` | flag | Use CUDA ONNX provider |
| `--clahe` | flag | CLAHE-enhance a *detection-only* copy of dark frames |
| `--anomaly-log` | `dataset/adnl/lp_extract_anomalies.txt` | Problem-file log |

> **Model resolution note:** adnl frames are 1440×1080 but the plate is only ~150×40 px.
> A 384-input model downscales the plate too far to detect — default to a **640**-input
> variant. If recall is still low on dark frames, try `yolo-v9-s-608-license-plate-end2end`
> (larger model). Valid model ids are listed by `open-image-models` — verify at build time.

### Detector wrapper (isolates the package — swap point for `ultralytics`)

```python
def build_detector(model: str, conf: float, gpu: bool):
    from open_image_models import LicensePlateDetector
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if gpu else ["CPUExecutionProvider"])
    return LicensePlateDetector(detection_model=model, conf_thresh=conf,
                                providers=providers)

def run_detect(detector, img_bgr) -> list[tuple[int, int, int, int, float]]:
    """Return normalized [(x1, y1, x2, y2, conf), ...] in original pixel coords."""
    out = []
    for d in detector.predict(img_bgr):
        b = d.bounding_box                       # VERIFY attr names at build time
        out.append((int(b.x1), int(b.y1), int(b.x2), int(b.y2), float(d.confidence)))
    return out
```

`build_detector` + `run_detect` are the **only** two functions that know the package.
Verify the `DetectionResult` / `bounding_box` attribute names once via
`uv run python -c "import open_image_models; help(open_image_models)"` when wiring this.

### Per-image loop

For each `split`, glob `sorted(dataroot/split/sharp/*.jpg)` (apply `--limit`):

1. **Load** — `cv2.imread(str(path))` (reads through symlinks). `None` → corrupt anomaly.
2. **Skip if done** — output JPEG already in `square/` or `rect/` and not `--overwrite`
   → count `skipped`, continue.
3. **Optional preprocess** (`--clahe`) — LAB convert, CLAHE on L, merge — used **only**
   for detection. The saved crop always comes from the *original* image.
4. **Detect** — `run_detect`; keep boxes with `conf >= --conf`.
5. **Select best box** — 0 boxes → `no detection` anomaly; multiple → take `max` by
   confidence (tie-break: larger area), record a `multi:N` note.
6. **Pad** — expand bbox by `padding * w` / `padding * h`, clamp to image bounds.
7. **Crop** — `img[y1:y2, x1:x2]`; near-zero area → `degenerate bbox` anomaly.
8. **Classify** — `ar = (x2-x1)/(y2-y1)`; `"rect" if ar >= ar_threshold else "square"`
   (same rule as `util/split_lp_by_shape.py:75-89`).
9. **Save** — `dataset/adnl/{ptype}/{split}/sharp/{path.name}` via
   `cv2.imwrite(..., [cv2.IMWRITE_JPEG_QUALITY, 95])`.

Create the 4 output dirs (`{square,rect}/{train,test}/sharp`) up front with
`os.makedirs(exist_ok=True)`; skip under `--dry-run`. Wrap each image body in
`try/except Exception` → anomaly with `repr(e)` so one bad file never aborts the run.

### Edge cases

- **No plate detected** → anomaly `(split, basename, "no detection")`, skip.
- **Multiple plates** → highest-confidence box wins; log `multi:N`.
- **Dark frames** → `--clahe` flag (detection-only); default off for predictability.
- **Corrupt image** → `imread` returns `None` → anomaly, skip.
- **Degenerate bbox** after clamp → anomaly, skip.

### Output: anomaly log + summary

- Anomalies → `--anomaly-log`, one `repr`-style tuple per line (matches the
  `dataset/anomalies.txt` convention). Skipped under `--dry-run`.
- Final `format_summary` (modeled on `util/split_lp_by_shape.py:142-177`): table of
  `square`/`rect` counts per `train`/`test`, grand total, plus undetected / corrupt /
  `multi` / skipped counts and the conf / AR / padding values used.

### Idempotency

Outputs are **real JPEGs** — do **not** copy `split_lp_by_shape.py`'s symlink-clearing
`clear_pass`, and never delete prior outputs. Default behavior skips already-cropped
images; `--overwrite` forces a re-crop. Safe to re-run.

## Files

**Create:** `util/extract_lp_from_vehicle.py`
**Modify:** `pyproject.toml` (dependency added automatically by `uv add open-image-models`)
**Read-only references:** `util/split_lp_by_shape.py`, `util/apply_disk_blur_mod.py`,
`data/LPBlur_dataset.py`, `dataset/anomalies.txt`

No existing files are edited by hand. `dataset/adnl/{train,test}/` raw photos are
untouched; only sibling `square/` and `rect/` dirs are added.

## Verification

1. **Detector loads + weights download** (sandbox disabled for this first run):
   `uv run util/extract_lp_from_vehicle.py --dry-run --limit 20`
   → prints per-split detection counts, writes nothing.
2. **Small real run:** `uv run util/extract_lp_from_vehicle.py --splits test --limit 50`
   → inspect ~10 crops in `dataset/adnl/{square,rect}/test/sharp/`: plate fully inside,
   body text (`ISUZU`/`QKR`) excluded.
3. **Classification spot-check:** a known 2-line plate lands in `square/`, a 1-line plate
   in `rect/`.
4. **Idempotency:** re-run step 2 → every image reports `skipped`; then add `--overwrite`
   → files rewritten.
5. **Dark-frame recall:** run with `--clahe` on the darkest captures; compare the
   undetected count against the no-CLAHE run. If still poor, bump `--model` to
   `yolo-v9-s-608-license-plate-end2end`.
6. **Anomaly review:** open `dataset/adnl/lp_extract_anomalies.txt`; sanity-check a few
   "no detection" files.
7. **Full run:** `uv run util/extract_lp_from_vehicle.py` (both splits; ~20–50 min on
   CPU, faster with `--gpu`). Review the printed summary.
8. **Downstream compatibility:** confirm `data/LPBlur_dataset.py:18-22` globs the new
   `sharp/` dirs correctly. To make a shape subset trainable later, run
   `uv run util/apply_disk_blur_mod.py dataset/adnl/square` (and `rect`) to synth the
   matching `blur/` images.

## Out of scope

- De-skew / perspective correction of crops (plain axis-aligned crop only).
- Deblurring the crops (separate LPDGAN `--mode test` step).
- OCR of plate text (`util/generate_plate_info.py` already exists for that).
