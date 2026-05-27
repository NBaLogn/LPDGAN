# License-Plate OCR — Postman User Guide

How to call the OCR API from Postman. No code required.

## Before you start

Ask whoever set up the server for:

1. **Server URL** — e.g. `http://10.0.0.5:8001` (replace `localhost:8001` below if needed).
2. **Sample plate images** to test with. Each image must be a tightly-cropped license plate (JPEG or PNG, under 8 MB).

> If the image shows the whole car or has a lot of background, results will be poor. Crop to the plate first.

---

## Step 1 — Check the server is up

**Method**: `GET`
**URL**: `http://localhost:8001/health`

Click **Send**.

Expected response:
```json
{
  "status": "ok",
  "model_loaded": true,
  "model": "cct-s-v2-global-model"
}
```

If you get a connection error, the server is down or the URL is wrong. Stop here and check with your admin.

---

## Step 2 — Read one plate (`/ocr`)

**Method**: `POST`
**URL**: `http://localhost:8001/ocr`

### Body tab

1. Select **Body**.
2. Choose **form-data**.
3. Add a row:
   - **Key**: `file`
   - Hover the key, change the type dropdown from **Text** to **File**.
   - **Value**: click **Select Files**, pick your plate image.

Click **Send**.

Expected response:
```json
{
  "filename": "plate.jpg",
  "text": "ABC1234",
  "indices": [11, 12, 13, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  "confidences": [0.999, 0.998, 0.997, 0.999, 0.999, 1.0, 0.999],
  "avg_confidence": 0.9987
}
```

### What the fields mean

| Field            | What it tells you                                                                |
|------------------|----------------------------------------------------------------------------------|
| `filename`       | The image you uploaded (so you can match results back if scripted).              |
| `text`           | The recognised plate. **This is the main answer.**                               |
| `avg_confidence` | How sure the model is overall (0.0–1.0). Anything < 0.8 is worth a manual check. |
| `confidences`    | Per-character confidence — useful if one letter is suspicious.                   |
| `indices`        | Numeric encoding for downstream training code. End-users can ignore.             |

---

## Step 3 — Read many plates in one shot (`/ocr/batch`)

Faster than calling `/ocr` repeatedly — the server runs them as one GPU batch.

**Method**: `POST`
**URL**: `http://localhost:8001/ocr/batch`

### Body tab

1. Select **Body** → **form-data**.
2. Add **multiple rows, all with the same key name** `files`:
   - **Key**: `files` (type **File**)
   - **Value**: pick the first image.
   - Click **+** to add a new row, key `files` again, pick the next image.
   - Repeat for each image, up to **1000 files per request**.

> Postman tip: in newer Postman versions you can multi-select files when the key is set to `files`. Hold Ctrl/Cmd while picking files.

Click **Send**.

Expected response:
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

- `results[i]` is in the **same order** as the files you uploaded.
- Use `filename` to match each result back to your local file.
- `count` is just `len(results)`.

### Limits

| Limit                | Value | What happens if exceeded                                     |
|----------------------|-------|--------------------------------------------------------------|
| Files per request    | 1000  | Postman shows "Too many files" / server rejects the upload   |
| Size of any one file | 8 MB  | Server returns `413 upload exceeds 8388608 bytes`            |

For more than 1000 plates, split into multiple `/ocr/batch` calls.

---

## Common errors

| Status | Body example                                | What to do                                                   |
|--------|---------------------------------------------|--------------------------------------------------------------|
| 400    | `{"detail": "empty upload"}`                | The file field is empty. Re-attach the image.                |
| 400    | `{"detail": "cannot decode image"}`         | Not a valid JPEG/PNG. Re-export or use a different file.     |
| 413    | `{"detail": "upload exceeds 8388608 bytes"}`| File > 8 MB. Re-save at lower quality / smaller dimensions. |
| 503    | `{"detail": "OCR not initialised"}`         | Server is still loading the model. Wait ~10 s and retry.     |
| —      | Connection refused / timeout                | Server is down or wrong URL. Ask the admin.                  |

---

## Tips for good results

- **Crop tight** to the plate. Any extra background degrades accuracy.
- **Brightness / blur**: very dark, motion-blurred, or low-resolution images fail more often. Check `avg_confidence` — values under 0.8 mean the answer might be wrong even if `text` looks plausible.
- **Resolution**: the model auto-resizes; you don't need a specific size. ~200 px wide is plenty.
- **Filename matters**: the API echoes it back. Use meaningful names so batch results are easy to match.
