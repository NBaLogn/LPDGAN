"""OCR HTTP API — PaddleOCR-backed license-plate recognition.

Detections are clustered into rows top->bottom (y-center, threshold =
median box height * ROW_GAP_FACTOR) and sorted left->right within each
row, so 2-row square plates reconstruct as "row1row2" — same convention
as ``util.generate_plate_info``.

Run:
    uv run uvicorn util.ocr_api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any, TypeAlias

TMPDIR = os.environ.get("TMPDIR", "/tmp")
os.environ.setdefault("HOME", TMPDIR)

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from paddleocr import PaddleOCR
from starlette.concurrency import run_in_threadpool

from util.generate_plate_info import make_ocr, text_to_indices

ROW_GAP_FACTOR: float = 0.3
MAX_UPLOAD_BYTES: int = 8 * 1024 * 1024

Detection: TypeAlias = tuple[list[list[float]], tuple[str, float]]

_ocr: PaddleOCR | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _ocr
    _ocr = make_ocr()
    yield
    _ocr = None


app = FastAPI(title="LPDGAN OCR API", version="0.1.0", lifespan=lifespan)


def _require_ocr() -> PaddleOCR:
    if _ocr is None:
        raise HTTPException(status_code=503, detail="OCR not initialised")
    return _ocr


def _decode_image(data: bytes) -> np.ndarray:
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"upload exceeds {MAX_UPLOAD_BYTES} bytes",
        )
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="cannot decode image")
    h, w = img.shape[:2]
    if max(h, w) < 300:
        img = cv2.resize(img, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
    return img


def _ocr_with_split_fallback(ocr: PaddleOCR, img: np.ndarray) -> list[Detection]:
    """Run OCR on full image; if ≤1 row detected, retry on top/bottom halves."""
    result = ocr.ocr(img, cls=True)
    detections: list[Detection] = result[0] if (result and result[0] is not None) else []
    if len(detections) > 1:
        return detections
    mid = img.shape[0] // 2
    top_r = ocr.ocr(img[:mid], cls=True)
    bot_r = ocr.ocr(img[mid:], cls=True)
    top_dets: list[Detection] = top_r[0] if (top_r and top_r[0] is not None) else []
    bot_dets: list[Detection] = bot_r[0] if (bot_r and bot_r[0] is not None) else []
    bot_dets = [([[p[0], p[1] + mid] for p in box], tc) for box, tc in bot_dets]
    combined = top_dets + bot_dets
    return combined if combined else detections


async def _load_and_detect(file: UploadFile) -> list[Detection]:
    ocr = _require_ocr()
    img = _decode_image(await file.read())
    return await run_in_threadpool(_ocr_with_split_fallback, ocr, img)


def _cluster_rows(detections: list[Detection]) -> tuple[list[str], list[float]]:
    """Cluster detections into rows top->bottom, sort each row left->right."""
    if not detections:
        return [], []

    items: list[dict[str, Any]] = []
    for box, (text, conf) in detections:
        pts = np.asarray(box, dtype=np.float32)
        items.append(
            {
                "y": float(pts[:, 1].mean()),
                "x": float(pts[:, 0].min()),
                "h": float(pts[:, 1].max() - pts[:, 1].min()),
                "text": text,
                "conf": float(conf),
            }
        )

    items.sort(key=lambda it: it["y"])
    median_h = float(np.median([it["h"] for it in items])) or 1.0
    gap = median_h * ROW_GAP_FACTOR

    rows: list[list[dict[str, Any]]] = []
    for it in items:
        if not rows or (it["y"] - rows[-1][-1]["y"]) > gap:
            rows.append([it])
        else:
            rows[-1].append(it)

    row_texts: list[str] = []
    confs: list[float] = []
    for row in rows:
        row.sort(key=lambda it: it["x"])
        row_texts.append("".join(it["text"] for it in row).strip("."))
        confs.extend(it["conf"] for it in row)
    return row_texts, confs


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "model_loaded": _ocr is not None}


@app.post("/ocr")
async def ocr(file: UploadFile = File(...)) -> dict[str, Any]:
    detections = await _load_and_detect(file)
    rows, confs = _cluster_rows(detections)
    full_text = "".join(rows)
    return {
        "text": full_text,
        "rows": rows,
        "num_rows": len(rows),
        "indices": text_to_indices(full_text),
        "confidences": confs,
    }


@app.post("/ocr/raw")
async def ocr_raw(file: UploadFile = File(...)) -> dict[str, Any]:
    detections = await _load_and_detect(file)
    payload = [
        {
            "box": [[float(p[0]), float(p[1])] for p in box],
            "text": text,
            "confidence": float(conf),
        }
        for box, (text, conf) in detections
    ]
    return {"detections": payload, "num_detections": len(payload)}


@app.post("/ocr/batch")
async def ocr_batch(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    """Upload multiple plate images → JSON with per-image OCR results and plate_info content.

    Reads and decodes images concurrently; OCR is sequential (PaddleOCR shared
    state is not thread-safe).
    """
    if not files:
        raise HTTPException(status_code=400, detail="no files provided")
    ocr = _require_ocr()

    raw_data = await asyncio.gather(*(f.read() for f in files))
    imgs = await asyncio.gather(*(
        run_in_threadpool(lambda d=d: _decode_image(d))
        for d in raw_data
    ), return_exceptions=True)

    results: list[dict[str, Any]] = []
    lines: list[str] = []
    failed = 0

    for file, img in zip(files, imgs):
        filename = file.filename or "unknown"
        fallback = text_to_indices("")
        if isinstance(img, Exception):
            failed += 1
            results.append({
                "filename": filename,
                "text": "",
                "rows": [],
                "num_rows": 0,
                "indices": fallback,
                "confidences": [],
                "avg_confidence": None,
            })
            lines.append(f"{filename},{' '.join(str(i) for i in fallback)}")
            continue
        try:
            detections = await run_in_threadpool(_ocr_with_split_fallback, ocr, img)
            rows, confs = _cluster_rows(detections)
            full_text = "".join(rows)
            avg_conf = float(np.mean(confs)) if confs else None
            indices = text_to_indices(full_text)
            results.append({
                "filename": filename,
                "text": full_text,
                "rows": rows,
                "num_rows": len(rows),
                "indices": indices,
                "confidences": confs,
                "avg_confidence": avg_conf,
            })
            lines.append(f"{filename},{' '.join(str(i) for i in indices)}")
            if not full_text:
                failed += 1
        except Exception:
            failed += 1
            results.append({
                "filename": filename,
                "text": "",
                "rows": [],
                "num_rows": 0,
                "indices": fallback,
                "confidences": [],
                "avg_confidence": None,
            })
            lines.append(f"{filename},{' '.join(str(i) for i in fallback)}")

    return {
        "results": results,
        "plate_info": "\n".join(lines) + "\n" if lines else "",
        "stats": {"processed": len(lines), "failed": failed},
    }
