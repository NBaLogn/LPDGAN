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


def test_batch(client: TestClient, square_img: Path, rect_img: Path) -> int:
    errors = 0

    print("\n--- /ocr/batch (2 images as files) ---")
    with square_img.open("rb") as sq, rect_img.open("rb") as rc:
        resp = client.post(
            "/ocr/batch",
            files=[
                ("files", (square_img.name, sq, "image/jpeg")),
                ("files", (rect_img.name, rc, "image/jpeg")),
            ],
        )
    print("STATUS:", resp.status_code)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    print(json.dumps(body, indent=2))

    assert "results" in body and "plate_info" in body and "stats" in body, body
    assert body["stats"]["processed"] == 2, body["stats"]
    results = body["results"]
    assert len(results) == 2, f"expected 2 results, got {len(results)}"

    plate_lines = [l for l in body["plate_info"].strip().split("\n") if l]
    assert len(plate_lines) == 2, f"expected 2 plate_info lines, got {len(plate_lines)}"

    for r in results:
        assert "filename" in r and "text" in r and "indices" in r, r
        assert "avg_confidence" in r, r
        assert len(r["indices"]) == 21, f"expected 21 indices, got {len(r['indices'])}"

    print("\n--- /ocr/batch (no files, expect 400) ---")
    resp = client.post("/ocr/batch", files=[])
    print("STATUS:", resp.status_code, "BODY:", resp.text[:200])
    if resp.status_code not in (400, 422):
        errors += 1

    print("\n--- /ocr/batch (corrupt image, expect graceful fallback) ---")
    resp = client.post(
        "/ocr/batch",
        files=[("files", ("bad.jpg", b"not-an-image", "image/jpeg"))],
    )
    print("STATUS:", resp.status_code)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["stats"]["failed"] == 1, body["stats"]
    assert body["results"][0]["text"] == "", body["results"]

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
