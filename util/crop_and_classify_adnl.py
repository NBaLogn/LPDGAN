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


# ---------------------------------------------------------------------------
# OCR row-count helpers
# ---------------------------------------------------------------------------


def _extract_polys_v3(result_obj) -> list[np.ndarray]:
    """Pull rec_polys from a PaddleOCR>=3.x result object."""
    polys: list[np.ndarray] = []
    payload = getattr(result_obj, "json", None)
    if payload is None:
        return polys
    if isinstance(payload, dict):
        res = payload.get("res", payload)
    else:
        res = payload
    raw = res.get("rec_polys") or res.get("dt_polys") or []
    for p in raw:
        polys.append(np.asarray(p, dtype=np.float32))
    return polys


def _extract_polys_v2(item) -> list[np.ndarray]:
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
        if score < 0.5 or len(str(text).strip()) < 2:
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

    def __init__(self) -> None:
        from paddleocr import PaddleOCR  # noqa: WPS433 (lazy import)

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
                polys.extend(_extract_polys_v3(r))
            return polys
        out = self._ocr.ocr(image_bgr, cls=True)
        if not out:
            return []
        return _extract_polys_v2(out[0])


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def classify_crop(
    crop_bgr: np.ndarray,
    ocr: OCRBackend,
    ar_threshold: float,
    row_gap_frac: float,
) -> tuple[str, dict]:
    """Return (plate_type, debug_info) for one cropped plate."""
    h, w = crop_bgr.shape[:2]
    ar = (w / h) if h > 0 else 0.0
    ar_choice = "rect" if ar >= ar_threshold else "square"

    polys = ocr.polys(crop_bgr)
    rows = count_rows(polys, crop_h=h, gap_frac=row_gap_frac)
    if rows >= 2:
        ocr_choice: str | None = "square"
    elif rows == 1:
        ocr_choice = "rect"
    else:
        ocr_choice = None

    chosen = ocr_choice if ocr_choice is not None else ar_choice
    return chosen, {
        "w": w,
        "h": h,
        "ar": round(ar, 3),
        "row_count": rows,
        "ocr_choice": ocr_choice or "",
        "ar_choice": ar_choice,
        "n_polys": len(polys),
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
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--ar-threshold", type=float, default=2.0)
    p.add_argument("--row-gap-frac", type=float, default=0.30)
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
    print(f"detector ready (device={device}, weights={args.weights.name})")

    ocr = OCRBackend()
    print(f"ocr ready (api={ocr._api})")

    if not args.dry_run:
        prepare_out(args.out, args.clean)

    ambiguous_path = args.out / "_ambiguous.csv"
    no_plate_path = args.out / "_no_plate.txt"
    ambiguous_rows: list[dict] = []
    no_plate: list[str] = []

    counts = {pt: {sp: 0 for sp in SPLITS} for pt in PLATE_TYPES}
    n_detected = 0
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
                plate_type, dbg = classify_crop(
                    crop,
                    ocr=ocr,
                    ar_threshold=args.ar_threshold,
                    row_gap_frac=args.row_gap_frac,
                )
                counts[plate_type][split] += 1
                n_detected += 1

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
        if no_plate:
            no_plate_path.write_text("\n".join(no_plate) + "\n")
            print(f"no-plate log       : {no_plate_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
