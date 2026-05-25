"""Smoke test for util.ocr_api — runs the ASGI app in-process via TestClient.

Avoids needing a live HTTP listener (sandbox blocks loopback in the dev env).
"""

from __future__ import annotations

import json
import sys
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

    print("\nSMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
