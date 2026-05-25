"""OCR HTTP API — PaddleOCR-backed license-plate recognition.

Detections are clustered into rows top->bottom (y-center, threshold =
median box height * ROW_GAP_FACTOR) and sorted left->right within each
row, so 2-row square plates reconstruct as "row1row2" — same convention
as ``util.generate_plate_info``.

Run:
    uv run uvicorn util.ocr_api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

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

ROW_GAP_FACTOR: float = 0.6
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
    return img


async def _load_and_detect(file: UploadFile) -> list[Detection]:
    ocr = _require_ocr()
    img = _decode_image(await file.read())
    result = await run_in_threadpool(ocr.ocr, img, cls=True)
    if not result or result[0] is None:
        return []
    return result[0]


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
        row_texts.append("".join(it["text"] for it in row))
        confs.extend(it["conf"] for it in row)
    return row_texts, confs


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
