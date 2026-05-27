# Fast-OCR API

FastAPI HTTP service for license-plate recognition backed by [`fast-plate-ocr`](https://github.com/ankandrew/fast-plate-ocr) (ONNX, CUDA EP).

Recognition only — caller supplies a tightly-cropped plate image. No text-region detection.

## Run

```bash
uv run uvicorn util.fast_ocr_api:app --host 0.0.0.0 --port 8001
```

### Environment variables

| Var              | Default                  | Purpose                             |
|------------------|--------------------------|-------------------------------------|
| `FAST_OCR_MODEL` | `cct-s-v2-global-model`  | Pre-trained model name from the hub |

### GPU

CUDA 12 nvidia packages are bundled (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`, etc.).
`util/fast_ocr_api.py` preloads them via `ctypes` at import so `onnxruntime-gpu` finds
`libcublas.so.12` even on CUDA 13 hosts.

TensorRT errors at startup are harmless — TensorRT is not installed; the runtime falls back to `CUDAExecutionProvider`.

## Limits

| Limit              | Value      | Source                               |
|--------------------|------------|--------------------------------------|
| Per-file size      | 8 MiB      | `MAX_UPLOAD_BYTES` in `_image_io.py` |
| Files per `/batch` | 1000       | Starlette `MultiPartParser.max_files` |

## Endpoints

### `GET /health`

Liveness + readiness.

```bash
curl http://100.111.0.111:8001/health
```

**Response 200**:
```json
{
  "status": "ok",
  "model_loaded": true,
  "model": "cct-s-v2-global-model"
}
```

---

### `POST /ocr`

Single-plate recognition.

**Request**: `multipart/form-data`, field `file` (JPEG/PNG, ≤ 8 MiB).

```bash
curl -X POST http://100.111.0.111:8001/ocr \
  -F "file=@plate.jpg"
```

**Response 200**:
```json
{
  "filename": "plate.jpg",
  "text": "ABC1234",
  "indices": [11, 12, 13, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  "confidences": [0.999, 0.998, 0.997, 0.999, 0.999, 1.0, 0.999],
  "avg_confidence": 0.9987
}
```

| Field            | Type            | Notes                                                       |
|------------------|-----------------|-------------------------------------------------------------|
| `filename`       | string          | Original upload filename                                    |
| `text`           | string          | Plate string (uppercase ASCII)                              |
| `indices`        | int[21]         | Index encoding from `util.generate_plate_info.CHAR_SET`     |
| `confidences`    | float[]         | Per-character probabilities (length = `len(text)`)          |
| `avg_confidence` | float \| null   | Mean of `confidences`, or null if model returned no probs   |

**Errors**:
- `400` — empty upload or undecodable image
- `413` — exceeds 8 MiB
- `503` — recognizer not initialised (server still warming up)

---

### `POST /ocr/batch`

Multi-plate recognition in a single batched GPU call.

**Request**: `multipart/form-data`, repeated field `files` (1–1000 files).

```bash
curl -X POST http://100.111.0.111:8001/ocr/batch \
  -F "files=@plate1.jpg" \
  -F "files=@plate2.jpg" \
  -F "files=@plate3.jpg"
```

**Response 200**:
```json
{
  "results": [
    {
      "filename": "plate1.jpg",
      "text": "ABC1234",
      "indices": [...],
      "confidences": [...],
      "avg_confidence": 0.998
    },
    {
      "filename": "plate2.jpg",
      "text": "XYZ9876",
      "indices": [...],
      "confidences": [...],
      "avg_confidence": 0.995
    }
  ],
  "count": 2
}
```

`results[i]` corresponds to `files[i]` (same order). Each entry has the same schema as `/ocr` minus the wrapping.

**Errors**:
- `400` — empty file list
- `413` — any single file exceeds 8 MiB
- `503` — recognizer not initialised

> Postman caps at 1000 multipart files — that matches the server limit. Above 1000, split into multiple requests.

---

### `POST /ocr/raw`

Raw prediction payload — includes region detection when the loaded model supports it.

**Request**: same shape as `/ocr`.

```bash
curl -X POST http://100.111.0.111:8001/ocr/raw \
  -F "file=@plate.jpg"
```

**Response 200** (model with region head):
```json
{
  "filename": "plate.jpg",
  "predictions": [
    {
      "plate": "ABC1234",
      "char_probs": [0.999, 0.998, 0.997, 0.999, 0.999, 1.0, 0.999],
      "region": "USA",
      "region_prob": 0.987
    }
  ],
  "num_predictions": 1
}
```

`region` / `region_prob` are omitted when the model doesn't predict region.

---

## Python client example

```python
import httpx

with open("plate.jpg", "rb") as f:
    r = httpx.post(
        "http://100.111.0.111:8001/ocr",
        files={"file": ("plate.jpg", f, "image/jpeg")},
    )
print(r.json()["text"])
```

Batch:

```python
import httpx
from pathlib import Path

paths = list(Path("plates/").glob("*.jpg"))[:1000]
files = [("files", (p.name, p.read_bytes(), "image/jpeg")) for p in paths]
r = httpx.post("http://100.111.0.111:8001/ocr/batch", files=files, timeout=120)
for item in r.json()["results"]:
    print(item["filename"], "->", item["text"])
```

## Character set

`indices` is fixed-length 21. Vocabulary from `util.generate_plate_info.CHAR_SET` (32 classes):

```
#0123456789ABCDEFGHKLMNPRSTUVXYZ
```

| Index | Char(s)                  |
|-------|--------------------------|
| 0     | `#` (padding / unknown)  |
| 1–10  | digits `0`–`9`           |
| 11–31 | letters `A`–`Z` minus `I`, `J`, `O`, `Q`, `W` |

Any input char outside the set maps to `0`. Output is right-padded with `0` to length 21.
