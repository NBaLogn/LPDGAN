"""Smoke test for util.ocr_api — runs the ASGI app in-process via TestClient.

Avoids needing a live HTTP listener (sandbox blocks loopback in the dev env).
"""

from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from util.ocr_api import app  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SQUARE_DIR = ROOT / "dataset" / "quan_lp" / "square" / "test" / "sharp"
RECT_DIR = ROOT / "dataset" / "quan_lp" / "rect" / "test" / "sharp"


def _pick(dirpath: Path) -> Path:
    for p in sorted(dirpath.iterdir()):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            return p
    raise FileNotFoundError(f"no image found in {dirpath}")


def _post(client: TestClient, endpoint: str, image: Path) -> dict:
    with image.open("rb") as f:
        resp = client.post(endpoint, files={"file": (image.name, f, "image/jpeg")})
    resp.raise_for_status()
    return resp.json()


def _make_zip(image_paths: list[Path]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in image_paths:
            zf.write(p, p.name)
    return buf.getvalue()


def test_batch(client: TestClient, square_img: Path, rect_img: Path) -> int:
    errors = 0

    print("\n--- /ocr/batch (2 images in zip) ---")
    zip_bytes = _make_zip([square_img, rect_img])
    resp = client.post(
        "/ocr/batch",
        files={"file": ("plates.zip", zip_bytes, "application/zip")},
    )
    print("STATUS:", resp.status_code, "HEADER:", dict(resp.headers))
    assert resp.status_code == 200, resp.status_code

    body = resp.text
    lines = [l for l in body.strip().split("\n") if l]
    assert len(lines) == 2, f"expected 2 lines, got {len(lines)}"

    stats = resp.headers.get("x-ocr-stats", "")
    print("  X-OCR-Stats:", stats)
    assert "processed=2" in stats, stats
    assert "failed=0" in stats, stats

    for line in lines:
        print("  ", line)
        assert "," in line, f"missing comma in line: {line}"
        name_part, idx_part = line.split(",", 1)
        assert name_part.endswith(".jpg"), f"bad name: {name_part}"
        parts = idx_part.strip().split()
        assert len(parts) == 21, f"expected 21 indices, got {len(parts)}"
        for p in parts:
            assert p.isdigit(), f"non-digit index: {p}"

    print("\n--- /ocr/batch (empty zip, expect 400) ---")
    empty_zip = _make_zip([])
    resp = client.post(
        "/ocr/batch", files={"file": ("empty.zip", empty_zip, "application/zip")}
    )
    print("STATUS:", resp.status_code, "BODY:", resp.json())
    if resp.status_code != 400:
        errors += 1

    print("\n--- /ocr/batch (no images zip, expect 400) ---")
    noimg_buf = io.BytesIO()
    with zipfile.ZipFile(noimg_buf, "w") as zf:
        zf.writestr("readme.txt", "no images here")
    resp = client.post(
        "/ocr/batch",
        files={"file": ("noimg.zip", noimg_buf.getvalue(), "application/zip")},
    )
    print("STATUS:", resp.status_code, "BODY:", resp.json())
    if resp.status_code != 400:
        errors += 1

    print("\n--- /ocr/batch (corrupt zip, expect 400) ---")
    resp = client.post(
        "/ocr/batch",
        files={"file": ("bad.zip", b"not-a-zip-content", "application/zip")},
    )
    print("STATUS:", resp.status_code, "BODY:", resp.json())
    if resp.status_code != 400:
        errors += 1

    return errors


def main() -> int:
    square_img = _pick(SQUARE_DIR)
    rect_img = _pick(RECT_DIR)

    with TestClient(app) as client:
        health = client.get("/health").json()
        print("HEALTH:", json.dumps(health))
        assert health["status"] == "ok" and health["model_loaded"], health

        print(f"\n--- /ocr (square, 2-row): {square_img.name} ---")
        sq = _post(client, "/ocr", square_img)
        print(json.dumps(sq, indent=2))

        print(f"\n--- /ocr/raw (square, 2-row): {square_img.name} ---")
        sq_raw = _post(client, "/ocr/raw", square_img)
        print(json.dumps(sq_raw, indent=2))

        print(f"\n--- /ocr (rect, 1-row): {rect_img.name} ---")
        rc = _post(client, "/ocr", rect_img)
        print(json.dumps(rc, indent=2))

        print("\n--- /ocr (empty upload, expect 400) ---")
        resp = client.post("/ocr", files={"file": ("empty.jpg", b"", "image/jpeg")})
        print("STATUS:", resp.status_code, "BODY:", resp.json())
        assert resp.status_code == 400, resp.status_code

        errs = test_batch(client, square_img, rect_img)

    if errs:
        print(f"\nFAILED ({errs} errors)")
        return 1

    print("\nSMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
