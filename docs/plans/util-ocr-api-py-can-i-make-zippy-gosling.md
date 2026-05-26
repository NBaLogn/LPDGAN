# Plan: Batch endpoint JSON response + plate_info field

## Context

`POST /ocr/batch` currently returns a `plate_info.txt` file download (PlainTextResponse). User wants per-image JSON results (same shape as `POST /ocr`) while still exposing the plate_info.txt content — both in one JSON response.

## File to modify

`util/ocr_api.py` — only `ocr_batch` function (lines 223–270) and its imports.

## New response shape

```json
{
  "results": [
    {
      "filename": "img001.jpg",
      "text": "51G123456",
      "rows": ["51G", "123456"],
      "num_rows": 2,
      "indices": [12, 34, 56, 78],
      "confidences": [0.99, 0.97]
    }
  ],
  "plate_info": "img001.jpg,12 34 56 78\nimg002.jpg,0 0 0 0\n",
  "stats": { "processed": 2, "failed": 0 }
}
```

- `results`: list of per-image dicts, each identical to `POST /ocr` response + `filename` key
- `plate_info`: same string content as previous txt download
- `stats`: moved from `X-OCR-Stats` header into body

## Implementation

1. In `ocr_batch`, build two parallel lists:
   - `results`: append `{filename: p.name, **res}` per image (reuse `_process_single_image_file`)
   - `lines`: same `f"{p.name},{' '.join(...)}"` logic as before
2. Return `dict` (FastAPI auto-serializes) instead of `PlainTextResponse`
3. Remove unused `Response` / `PlainTextResponse` / `FileResponse` from imports if no longer needed elsewhere

## Reused

- `_process_single_image_file` (`util/ocr_api.py:163`) — unchanged
- `_validate_and_extract_zip` (`util/ocr_api.py:129`) — unchanged

## Verification

```bash
# start server
uvr uvicorn util.ocr_api:app --host 0.0.0.0 --port 8000

# create test zip of plate images, then:
curl -s -X POST http://localhost:8000/ocr/batch \
  -F "file=@test_plates.zip" | python3 -m json.tool

# confirm keys: results, plate_info, stats
# confirm results[0] has: filename, text, rows, num_rows, indices, confidences
```
