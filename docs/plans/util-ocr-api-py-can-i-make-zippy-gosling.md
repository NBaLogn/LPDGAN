# Plan: Align /ocr/batch with fast_ocr_api.py style

## Context

`fast_ocr_api.py` (LPDGAN.fast-plate-ocr) accepts `list[UploadFile]` for batch, does concurrent reads + decodes via `asyncio.gather`. Current `ocr_api.py /ocr/batch` accepts a single zip, extracts to tempdir, processes sequentially — more ceremony, awkward for callers used to the fast-ocr style. Goal: adopt the same multi-file upload pattern and concurrent I/O, while keeping `plate_info` and `stats` fields (training-pipeline-specific) and sequential OCR (PaddleOCR is not thread-safe).

## File to modify

`util/ocr_api.py` only.

## Changes

### 1. Input: zip → `list[UploadFile]`

```python
# Before
async def ocr_batch(file: UploadFile = File(...)) -> dict[str, Any]:

# After
async def ocr_batch(files: list[UploadFile] = File(...)) -> dict[str, Any]:
```

### 2. Concurrency: concurrent reads + concurrent decodes, sequential OCR

```python
if not files:
    raise HTTPException(status_code=400, detail="no files provided")
ocr = _require_ocr()
raw_data = await asyncio.gather(*(f.read() for f in files))
imgs = await asyncio.gather(*(
    run_in_threadpool(lambda d=d: _decode_image(d))
    for d in raw_data
))
# OCR sequential — PaddleOCR shared state not thread-safe
for file, img in zip(files, imgs):
    detections = await run_in_threadpool(_ocr_with_split_fallback, ocr, img)
    ...
```

### 3. Response: keep plate_info + stats, add avg_confidence

`filename` from `file.filename`. `avg_confidence = float(np.mean(confs)) if confs else None`.

```json
{
  "results": [{"filename": "lp161.jpg", "text": "29-V7504.44", "rows": [...],
               "num_rows": 2, "indices": [...], "confidences": [...], "avg_confidence": 0.98}],
  "plate_info": "lp161.jpg,3 10 ...\n",
  "stats": {"processed": 1, "failed": 0}
}
```

### 4. Remove dead code

- `_validate_and_extract_zip` — delete (zip upload gone)
- `_process_single_image_file` — delete (logic inlined into loop)
- Unused imports: `io`, `shutil`, `tempfile`, `zipfile`
- Unused constants: `BATCH_MAX_UPLOAD_BYTES`, `IMAGE_EXTS`

### 5. Add import

`import asyncio`

## Verification

```bash
uv run uvicorn util.ocr_api:app --host 0.0.0.0 --port 8000

curl -s http://localhost:8000/ocr/batch \
  -F "files=@dataset/quan_lp/train/sharp/lp161.jpg" \
  -F "files=@dataset/quan_lp/train/sharp/lp167.jpg" | python3 -m json.tool
# expect: results[*].filename, results[*].avg_confidence, plate_info string, stats keys
```
