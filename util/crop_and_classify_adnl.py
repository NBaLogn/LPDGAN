"""Crop license plates from ADNL full-frame captures and classify each crop
as ``square`` (2-row) or ``rect`` (1-row).

Source layout (input)
---------------------
    dataset/adnl/<split>/sharp/*.jpg          # 1440x1080 full frames
        split in {train, test}

Output layout (one real JPEG per detection, suffixed by detection index)
----------------------------------------------------------------------
    dataset/adnl_cropped/<plate_type>/<split>/sharp/<stem>_<idx>.jpg
        plate_type in {square, rect}

Pipeline
--------
1. Walk every frame under ``--dataroot/<split>/sharp/``.
2. Run an ultralytics YOLO detector (Vietnamese-LP trained, by default
   ``checkpoints/lp_detector/plate_yolo12n_640_2025.pt``). Keep every
   detection above ``--conf``.
3. Crop, then classify via two signals:
     (a) PaddleOCR text-line detection -> count of row clusters
         (y-centers grouped with gap ``--row-gap-frac * crop_h``).
         1 cluster -> rect, >=2 clusters -> square.
     (b) Aspect-ratio fallback when OCR returns zero usable lines:
         ``w / h >= --ar-threshold`` -> rect, else square.
4. Write the crop into the corresponding output bucket.
5. Disagreements between OCR-row-count and AR-rule are logged to
   ``_ambiguous.csv`` so they can be reviewed later. Frames with no
   detection at all go to ``_no_plate.txt``.

Re-running is safe: existing files in the output tree are overwritten in
place for any (stem, idx) re-emitted. Stale files from prior runs are not
auto-cleaned -- pass ``--clean`` to wipe the output root before writing.

Example
-------
    uv run util/crop_and_classify_adnl.py --limit 200 --device auto
    uv run util/crop_and_classify_adnl.py                       # full run
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
import time
from pathlib import Path

# PaddleOCR / Matplotlib both touch $HOME; redirect to a writable tmpdir.
_TMP = os.environ.get("TMPDIR", tempfile.gettempdir())
os.environ.setdefault("HOME", _TMP)
os.environ.setdefault("MPLCONFIGDIR", _TMP)

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DATAROOT = REPO_ROOT / "dataset" / "adnl"
DEFAULT_OUT = REPO_ROOT / "dataset" / "adnl_cropped"
DEFAULT_WEIGHTS = (
    REPO_ROOT / "checkpoints" / "lp_detector" / "plate_yolo12n_640_2025.pt"
)
WEIGHTS_URL = (
    "https://raw.githubusercontent.com/tungedng2710/AI-Traffic-Analysis/"
    "main/weights/plate/plate_yolo12n_640_2025.pt"
)

SPLITS = ("train", "test")
PLATE_TYPES = ("square", "rect")
JPEG_QUALITY = 95

# Defaults for false-positive guards. All are overridable via CLI.
DEFAULT_CONF = 0.5
DEFAULT_AR_THRESHOLD = 2.0
DEFAULT_ROW_GAP_FRAC = 0.30
DEFAULT_MIN_AREA = 2000        # px^2; YOLO boxes smaller than this are noise
DEFAULT_MIN_AR = 1.0           # crops outside [MIN_AR, MAX_AR] are non-plate
DEFAULT_MAX_AR = 6.0
DEFAULT_HARD_RECT_AR = 3.0     # AR >= this -> force rect, skip OCR
DEFAULT_HARD_SQUARE_AR = 1.3   # AR <= this -> force square, skip OCR
DEFAULT_OCR_MIN_TEXT_LEN = 3   # drop OCR lines with <3 chars (artifacts)
DEFAULT_OCR_MIN_CONF = 0.5


# ---------------------------------------------------------------------------
# OCR row-count helpers
# ---------------------------------------------------------------------------


def _extract_polys_v3(
    result_obj,
    min_text_len: int = DEFAULT_OCR_MIN_TEXT_LEN,
    min_conf: float = DEFAULT_OCR_MIN_CONF,
) -> list[np.ndarray]:
    """Pull rec_polys from a PaddleOCR>=3.x result object.

    Filters lines whose recognized text fails the min-length / min-conf gate
    so phantom short-text detections do not inflate the row count.
    """
    polys: list[np.ndarray] = []
    payload = getattr(result_obj, "json", None)
    if payload is None:
        return polys
    if isinstance(payload, dict):
        res = payload.get("res", payload)
    else:
        res = payload
    raw_polys = res.get("rec_polys") or res.get("dt_polys") or []
    raw_texts = res.get("rec_texts") or []
    raw_scores = res.get("rec_scores") or []
    for i, p in enumerate(raw_polys):
        text = raw_texts[i] if i < len(raw_texts) else ""
        score = float(raw_scores[i]) if i < len(raw_scores) else 0.0
        if score < min_conf or len(str(text).strip()) < min_text_len:
            continue
        polys.append(np.asarray(p, dtype=np.float32))
    return polys


def _extract_polys_v2(
    item,
    min_text_len: int = DEFAULT_OCR_MIN_TEXT_LEN,
    min_conf: float = DEFAULT_OCR_MIN_CONF,
) -> list[np.ndarray]:
    """Pull polys from the legacy PaddleOCR (v2.x) ``[[poly, (text, score)], ...]`` format."""
    polys: list[np.ndarray] = []
    if not item:
        return polys
    for entry in item:
        if not entry:
            continue
        poly = entry[0] if len(entry) >= 1 else None
        text_score = entry[1] if len(entry) >= 2 else (None, 0.0)
        text = text_score[0] if text_score else None
        score = text_score[1] if text_score and len(text_score) > 1 else 0.0
        if poly is None or text is None:
            continue
        if score < min_conf or len(str(text).strip()) < min_text_len:
            continue
        polys.append(np.asarray(poly, dtype=np.float32))
    return polys


def count_rows(polys: list[np.ndarray], crop_h: int, gap_frac: float) -> int:
    """Cluster polygon y-centers; return the number of distinct rows."""
    if not polys:
        return 0
    y_centers = sorted(float(p[:, 1].mean()) for p in polys)
    gap_thr = max(1.0, gap_frac * crop_h)
    rows = 1
    for prev, cur in zip(y_centers, y_centers[1:]):
        if cur - prev > gap_thr:
            rows += 1
    return rows


class OCRBackend:
    """Thin wrapper that hides PaddleOCR v2 vs v3 differences."""

    def __init__(
        self,
        min_text_len: int = DEFAULT_OCR_MIN_TEXT_LEN,
        min_conf: float = DEFAULT_OCR_MIN_CONF,
    ) -> None:
        from paddleocr import PaddleOCR  # noqa: WPS433 (lazy import)

        self.min_text_len = min_text_len
        self.min_conf = min_conf

        # paddleocr 2.10 silently accepts unknown kwargs; choose the API by
        # probing whether ``predict`` exists (3.x) versus only ``ocr`` (2.x).
        try:
            self._ocr = PaddleOCR(
                lang="en",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        except TypeError:
            self._ocr = PaddleOCR(lang="en", use_angle_cls=True)
        if hasattr(self._ocr, "predict") and callable(self._ocr.predict):
            self._api = "v3"
        else:
            self._api = "v2"

    def polys(self, image_bgr: np.ndarray) -> list[np.ndarray]:
        if self._api == "v3":
            results = self._ocr.predict(image_bgr)
            polys: list[np.ndarray] = []
            for r in results:
                polys.extend(_extract_polys_v3(r, self.min_text_len, self.min_conf))
            return polys
        out = self._ocr.ocr(image_bgr, cls=True)
        if not out:
            return []
        return _extract_polys_v2(out[0], self.min_text_len, self.min_conf)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def classify_crop(
    crop_bgr: np.ndarray,
    ocr: OCRBackend,
    ar_threshold: float,
    row_gap_frac: float,
    hard_rect_ar: float,
    hard_square_ar: float,
) -> tuple[str, dict]:
    """Return (plate_type, debug_info) for one cropped plate.

    Hard-AR override (geometry is decisive at the extremes):
      * AR >= ``hard_rect_ar``   -> always ``rect``   (real squares cap at AR ~2.2)
      * AR <= ``hard_square_ar`` -> always ``square`` (real 1-row plates start AR ~2.0)
    In both cases the OCR call is skipped entirely.
    """
    h, w = crop_bgr.shape[:2]
    ar = (w / h) if h > 0 else 0.0
    ar_choice = "rect" if ar >= ar_threshold else "square"

    if ar >= hard_rect_ar:
        return "rect", {
            "w": w, "h": h, "ar": round(ar, 3),
            "row_count": -1, "ocr_choice": "", "ar_choice": ar_choice,
            "n_polys": 0, "reason": "hard_ar",
        }
    if 0 < ar <= hard_square_ar:
        return "square", {
            "w": w, "h": h, "ar": round(ar, 3),
            "row_count": -1, "ocr_choice": "", "ar_choice": ar_choice,
            "n_polys": 0, "reason": "hard_ar",
        }

    polys = ocr.polys(crop_bgr)
    rows = count_rows(polys, crop_h=h, gap_frac=row_gap_frac)
    if rows >= 2:
        ocr_choice: str | None = "square"
    elif rows == 1:
        ocr_choice = "rect"
    else:
        ocr_choice = None

    chosen = ocr_choice if ocr_choice is not None else ar_choice
    reason = "ocr" if ocr_choice is not None else "ar_fallback"
    return chosen, {
        "w": w,
        "h": h,
        "ar": round(ar, 3),
        "row_count": rows,
        "ocr_choice": ocr_choice or "",
        "ar_choice": ar_choice,
        "n_polys": len(polys),
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


def load_detector(weights: Path, device: str):
    from ultralytics import YOLO  # noqa: WPS433

    if not weights.exists():
        sys.stderr.write(
            f"weights not found at {weights}\n"
            f"download with:\n"
            f"  mkdir -p {weights.parent}\n"
            f"  curl -fsSL -o {weights} {WEIGHTS_URL}\n"
        )
        sys.exit(2)

    model = YOLO(str(weights))
    if device == "auto":
        import torch

        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    return model, device


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def iter_frames(dataroot: Path, limit: int | None) -> list[tuple[str, Path]]:
    """Yield (split, image_path) for every frame under dataroot/<split>/sharp/."""
    out: list[tuple[str, Path]] = []
    for split in SPLITS:
        sharp = dataroot / split / "sharp"
        if not sharp.is_dir():
            continue
        for entry in sorted(sharp.iterdir()):
            if entry.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            out.append((split, entry))
    if limit is not None:
        out = out[:limit]
    return out


def prepare_out(out_root: Path, clean: bool) -> None:
    if clean and out_root.exists():
        import shutil

        shutil.rmtree(out_root)
    for plate in PLATE_TYPES:
        for split in SPLITS:
            (out_root / plate / split / "sharp").mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataroot", type=Path, default=DEFAULT_DATAROOT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--conf", type=float, default=DEFAULT_CONF)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--ar-threshold", type=float, default=DEFAULT_AR_THRESHOLD)
    p.add_argument("--row-gap-frac", type=float, default=DEFAULT_ROW_GAP_FRAC)
    p.add_argument("--min-area", type=int, default=DEFAULT_MIN_AREA,
                   help="reject crops smaller than this in px^2")
    p.add_argument("--min-ar", type=float, default=DEFAULT_MIN_AR,
                   help="reject crops with aspect ratio below this")
    p.add_argument("--max-ar", type=float, default=DEFAULT_MAX_AR,
                   help="reject crops with aspect ratio above this")
    p.add_argument("--hard-rect-ar", type=float, default=DEFAULT_HARD_RECT_AR,
                   help="AR at or above this -> always rect (skip OCR)")
    p.add_argument("--hard-square-ar", type=float, default=DEFAULT_HARD_SQUARE_AR,
                   help="AR at or below this -> always square (skip OCR)")
    p.add_argument("--ocr-min-text-len", type=int, default=DEFAULT_OCR_MIN_TEXT_LEN,
                   help="drop OCR lines whose recognized text is shorter than this")
    p.add_argument("--ocr-min-conf", type=float, default=DEFAULT_OCR_MIN_CONF,
                   help="drop OCR lines whose confidence is below this")
    p.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--limit", type=int, default=None, help="process only first N frames")
    p.add_argument("--clean", action="store_true", help="wipe --out before writing")
    p.add_argument("--dry-run", action="store_true", help="detect + classify but skip writes")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    frames = iter_frames(args.dataroot, args.limit)
    if not frames:
        print(f"no frames found under {args.dataroot}", file=sys.stderr)
        return 1
    print(f"frames to process: {len(frames)}")

    detector, device = load_detector(args.weights, args.device)
    print(
        f"detector ready (device={device}, weights={args.weights.name}, "
        f"conf={args.conf}, iou={args.iou})"
    )

    ocr = OCRBackend(
        min_text_len=args.ocr_min_text_len,
        min_conf=args.ocr_min_conf,
    )
    print(
        f"ocr ready (api={ocr._api}, min_text_len={args.ocr_min_text_len}, "
        f"min_conf={args.ocr_min_conf})"
    )
    print(
        f"guards: min_area={args.min_area}px^2  ar in "
        f"[{args.min_ar}, {args.max_ar}]  hard_rect>={args.hard_rect_ar}  "
        f"hard_square<={args.hard_square_ar}"
    )

    if not args.dry_run:
        prepare_out(args.out, args.clean)

    ambiguous_path = args.out / "_ambiguous.csv"
    no_plate_path = args.out / "_no_plate.txt"
    filtered_path = args.out / "_filtered.csv"
    ambiguous_rows: list[dict] = []
    filtered_rows: list[dict] = []
    no_plate: list[str] = []

    counts = {pt: {sp: 0 for sp in SPLITS} for pt in PLATE_TYPES}
    n_detected = 0
    n_filtered = 0
    n_hard_ar = 0
    t0 = time.time()

    # Batch frames through YOLO for throughput; classify per crop.
    for start in range(0, len(frames), args.batch):
        chunk = frames[start : start + args.batch]
        paths = [str(p) for _, p in chunk]
        results = detector.predict(
            paths,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=device,
            verbose=False,
        )
        for (split, src_path), res in zip(chunk, results):
            frame = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
            if frame is None:
                no_plate.append(f"{split}/{src_path.name}\tunreadable")
                continue
            H, W = frame.shape[:2]
            boxes = res.boxes
            if boxes is None or len(boxes) == 0:
                no_plate.append(f"{split}/{src_path.name}")
                continue

            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy() if boxes.conf is not None else None
            stem = src_path.stem
            for idx, (x1, y1, x2, y2) in enumerate(xyxy):
                x1i = max(0, int(round(float(x1))))
                y1i = max(0, int(round(float(y1))))
                x2i = min(W, int(round(float(x2))))
                y2i = min(H, int(round(float(y2))))
                if x2i <= x1i or y2i <= y1i:
                    continue
                crop = frame[y1i:y2i, x1i:x2i]
                if crop.size == 0:
                    continue

                cw, ch = crop.shape[1], crop.shape[0]
                cAR = (cw / ch) if ch > 0 else 0.0
                det_conf = float(confs[idx]) if confs is not None else -1.0

                # Hard geometry guards - drop before OCR even runs.
                drop_reason: str | None = None
                if cw * ch < args.min_area:
                    drop_reason = "min_area"
                elif cAR < args.min_ar:
                    drop_reason = "min_ar"
                elif cAR > args.max_ar:
                    drop_reason = "max_ar"
                if drop_reason is not None:
                    filtered_rows.append({
                        "split": split, "stem": stem, "idx": idx,
                        "w": cw, "h": ch, "ar": round(cAR, 3),
                        "det_conf": round(det_conf, 3), "reason": drop_reason,
                    })
                    n_filtered += 1
                    continue

                plate_type, dbg = classify_crop(
                    crop,
                    ocr=ocr,
                    ar_threshold=args.ar_threshold,
                    row_gap_frac=args.row_gap_frac,
                    hard_rect_ar=args.hard_rect_ar,
                    hard_square_ar=args.hard_square_ar,
                )
                counts[plate_type][split] += 1
                n_detected += 1
                if dbg["reason"] == "hard_ar":
                    n_hard_ar += 1

                if dbg["ocr_choice"] and dbg["ocr_choice"] != dbg["ar_choice"]:
                    ambiguous_rows.append(
                        {
                            "split": split,
                            "stem": stem,
                            "idx": idx,
                            "w": dbg["w"],
                            "h": dbg["h"],
                            "ar": dbg["ar"],
                            "row_count": dbg["row_count"],
                            "ocr_chosen": dbg["ocr_choice"],
                            "ar_chosen": dbg["ar_choice"],
                            "chosen": plate_type,
                            "reason": dbg["reason"],
                            "det_conf": round(det_conf, 3),
                        }
                    )

                if args.dry_run:
                    continue
                out_path = args.out / plate_type / split / "sharp" / f"{stem}_{idx}.jpg"
                cv2.imwrite(str(out_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])

        if (start // args.batch) % 10 == 0:
            elapsed = time.time() - t0
            done = min(start + args.batch, len(frames))
            print(
                f"  [{done}/{len(frames)}] detected={n_detected}  "
                f"square_train={counts['square']['train']} square_test={counts['square']['test']}  "
                f"rect_train={counts['rect']['train']} rect_test={counts['rect']['test']}  "
                f"elapsed={elapsed:.1f}s"
            )

    elapsed = time.time() - t0
    print("---")
    print(f"frames processed   : {len(frames)}")
    print(f"detections (crops) : {n_detected}")
    print(f"filtered out       : {n_filtered} (min_area / ar bounds)")
    print(f"hard-AR fast path  : {n_hard_ar}")
    print(f"no-plate frames    : {len(no_plate)}")
    print(f"ambiguous crops    : {len(ambiguous_rows)}")
    for pt in PLATE_TYPES:
        for sp in SPLITS:
            print(f"  {pt:6s} / {sp:5s} : {counts[pt][sp]}")
    print(f"elapsed            : {elapsed:.1f}s")

    if not args.dry_run:
        if ambiguous_rows:
            with ambiguous_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(ambiguous_rows[0].keys()))
                writer.writeheader()
                writer.writerows(ambiguous_rows)
            print(f"ambiguous log      : {ambiguous_path}")
        if filtered_rows:
            with filtered_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(filtered_rows[0].keys()))
                writer.writeheader()
                writer.writerows(filtered_rows)
            print(f"filtered log       : {filtered_path}")
        if no_plate:
            no_plate_path.write_text("\n".join(no_plate) + "\n")
            print(f"no-plate log       : {no_plate_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
