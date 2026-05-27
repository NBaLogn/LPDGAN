from __future__ import annotations

import cv2
import numpy as np
from fastapi import HTTPException

MAX_UPLOAD_BYTES: int = 8 * 1024 * 1024


def decode_image(data: bytes, *, as_rgb: bool = False) -> np.ndarray:
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
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if as_rgb else img
