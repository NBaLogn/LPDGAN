# Plan: Batch OCR API — Zip Upload → plate_info.txt Download

## Context

User needs to run PaddleOCR over many images at once and get `plate_info.txt` — dataset annotation file that `data/LPBlur_dataset.py` consumes during training (format: `image_name,idx0 idx1 ... idx20`, comma-delimited, no header, read via `pd.read_csv(sep=',')`).

Since HTTP can't send folders, zip upload is practical choice. A single new endpoint handles extraction, OCR, and returns `plate_info.txt` as downloadable file.

## Changes

Single file: **`util/ocr_api.py`** — add 1 endpoint + 1 helper.

## New endpoint: `POST /ocr/batch`

Upload `.zip` of plate images → download `plate_info.txt`.

**Request:** `multipart/form-data` with `file` field (`.zip` up to 200 MB).

**Response:** `FileResponse` (`text/plain`, `Content-Disposition: attachment; filename="plate_info.txt"`).
Header `X-OCR-Stats: processed=N, failed=N`.

plate_info.txt format per line:
```
<basename>,<idx0> <idx1> ... <idx20>\n
```
Matches `LPBlur_dataset.py:26` (`pd.read_csv(sep=',')`). Failed images → all-# indices.

## Helper: `_validate_and_extract_zip(file) -> (tmp_dir, sorted_image_paths)`

1. `await file.read()` → validate size ≤ `BATCH_MAX_UPLOAD_BYTES` (200 MB)
2. `tempfile.mkdtemp(prefix="ocr_batch_")`
3. `zipfile.ZipFile(io.BytesIO(data))` → extract only `.jpg/.jpeg/.png` members
4. `tmp_dir.rglob("*")` → find all images, sort, return
5. On error: `shutil.rmtree` before raising `HTTPException`

## Helper: `_process_single_image_file(path) -> dict`

Reuses `_decode_image` (bytes→cv2), `run_in_threadpool(ocr.ocr)`, `_cluster_rows`, `text_to_indices`.
Returns same shape as `/ocr` response.

## OCR processing loop

```
sem = asyncio.Semaphore(8)  # limit concurrent PaddleOCR threads
tasks = [_bounded(p) for p in image_paths]
results = await asyncio.gather(*tasks)
```

Each image wrapped in try/except → `null` on failure (doesn't kill batch).

## Cleanup

`try/finally: shutil.rmtree(tmp_dir, ignore_errors=True)` guarantees temp deletion.

## Imports added (stdlib, no new deps)

`asyncio`, `io`, `shutil`, `tempfile`, `zipfile`, `Path` (already used indirectly via `Path` patterns in codebase; explicit import for `rglob`).

## Verification

1. Run `uv run util/_smoke_ocr_api.py` — existing tests must still pass
2. Update `_smoke_ocr_api.py` with batch test:
   - Create in-memory zip with 2-3 test images from `dataset/quan_lp/{square,rect}/test/sharp/`
   - POST to `/ocr/batch` → assert status 200, Content-Type `text/plain`, non-empty body
   - Assert body lines match `basename,<space-separated-indices>` pattern
   - Verify header `X-OCR-Stats: processed=N, failed=0`
3. Test error cases: empty zip → 400, non-image zip → 400, corrupt zip → 400
