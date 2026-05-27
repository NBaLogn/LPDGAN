"""OCR HTTP API — fast-plate-ocr-backed license-plate recognition.

Expects a cropped plate image (JPEG/PNG). Unlike the PaddleOCR API, this
backend performs recognition only (no text-region detection), so the caller
must supply a tightly-cropped plate.

Model is selected via ``FAST_OCR_MODEL`` env var (default: cct-s-v2-global-model).

Run:
    uv run uvicorn util.fast_ocr_api:app --host 0.0.0.0 --port 8001
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

# Prepend nvidia CUDA 12 lib paths so onnxruntime-gpu finds libcublas.so.12 / libcublasLt.so.12.
# Must happen before fast_plate_ocr (and thus onnxruntime) is imported.
import ctypes as _ctypes
import glob as _glob
import site as _site

# onnxruntime-gpu needs CUDA 12 .so files. Preload them via ctypes so the dynamic
# linker finds them already mapped when libonnxruntime_providers_cuda.so is opened.
for _lib_dir in _glob.glob(os.path.join(_site.getsitepackages()[0], "nvidia", "*", "lib")):
    for _so in _glob.glob(os.path.join(_lib_dir, "*.so*")):
        if not os.path.islink(_so):  # load real files only, skip symlinks
            try:
                _ctypes.cdll.LoadLibrary(_so)
            except OSError:
                pass
del _ctypes, _glob, _site, _lib_dir, _so

import asyncio

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fast_plate_ocr import LicensePlateRecognizer, PlatePrediction
from starlette.concurrency import run_in_threadpool

from util._image_io import decode_image
from util.generate_plate_info import text_to_indices

MODEL_NAME: str = os.environ.get("FAST_OCR_MODEL", "cct-s-v2-global-model")

_recognizer: LicensePlateRecognizer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _recognizer
    _recognizer = LicensePlateRecognizer(MODEL_NAME)
    yield
    _recognizer = None


app = FastAPI(title="LPDGAN Fast-OCR API", version="0.1.0", lifespan=lifespan)


def _require_recognizer() -> LicensePlateRecognizer:
    if _recognizer is None:
        raise HTTPException(status_code=503, detail="OCR not initialised")
    return _recognizer


async def _load_preds(file: UploadFile) -> list[PlatePrediction]:
    rec = _require_recognizer()
    img = decode_image(await file.read(), as_rgb=True)
    return await run_in_threadpool(lambda: rec.run(img, return_confidence=True))


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": _recognizer is not None,
        "model": MODEL_NAME,
    }


@app.post("/ocr")
async def ocr(file: UploadFile = File(...)) -> dict[str, Any]:
    preds = await _load_preds(file)
    pred = preds[0] if preds else None

    text = pred.plate if pred else ""
    char_probs = pred.char_probs if (pred and pred.char_probs is not None) else None

    return {
        "filename": file.filename,
        "text": text,
        "indices": text_to_indices(text),
        "confidences": char_probs.tolist() if char_probs is not None else [],
        "avg_confidence": float(np.mean(char_probs)) if char_probs is not None else None,
    }


@app.post("/ocr/batch")
async def ocr_batch(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    if not files:
        raise HTTPException(status_code=400, detail="no files provided")
    rec = _require_recognizer()
    raw_data = await asyncio.gather(*(f.read() for f in files))
    imgs = await asyncio.gather(*(
        run_in_threadpool(lambda d=d: decode_image(d, as_rgb=True))
        for d in raw_data
    ))
    preds: list[PlatePrediction] = await run_in_threadpool(
        lambda: rec.run(list(imgs), return_confidence=True)
    )
    results = []
    for f, p in zip(files, preds):
        char_probs = p.char_probs if p.char_probs is not None else None
        results.append({
            "filename": f.filename,
            "text": p.plate,
            "indices": text_to_indices(p.plate),
            "confidences": char_probs.tolist() if char_probs is not None else [],
            "avg_confidence": float(np.mean(char_probs)) if char_probs is not None else None,
        })
    return {"results": results, "count": len(results)}


@app.post("/ocr/raw")
async def ocr_raw(file: UploadFile = File(...)) -> dict[str, Any]:
    preds = await _load_preds(file)

    payload: list[dict[str, Any]] = []
    for p in preds:
        entry: dict[str, Any] = {
            "plate": p.plate,
            "char_probs": p.char_probs.tolist() if p.char_probs is not None else None,
        }
        if p.region is not None:
            entry["region"] = p.region
            entry["region_prob"] = float(p.region_prob) if p.region_prob is not None else None
        payload.append(entry)

    return {"filename": file.filename, "predictions": payload, "num_predictions": len(payload)}
