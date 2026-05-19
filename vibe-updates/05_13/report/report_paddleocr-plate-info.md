# Report: PaddleOCR plate_info Generation

**Commit:** `40f4a26` (05-13). Refactor of `scripts/generate_plate_info.py` (+70/-43, net 113 LOC).

## Change Summary
| Item | Before | After |
|------|--------|-------|
| OCR call | (legacy variants in `util/`) | `PaddleOCR(lang="ch", use_textline_orientation=True).predict(img_path)` |
| Outputs | 1 file (`plate_info.txt`) | 2 files: `plate_info.txt` (indices) + `plate_info_text.txt` (raw text) |
| Existing files | overwritten | backed up to `*.bak` |
| Char set | (legacy) | 33 classes: `#` + 0-9 + 20 letters (A-Z excluding I,J,O,Q,W) + `.` + `-` |

## Charset Detail
```
CHAR_SET = "#0123456789ABCDEFGHKLMNPRSTUVXYZ"  # 33 chars
# Index 0:    # (padding/unknown)
# Index 1-10: 0-9
# Index 11-30: A-Z excluding I,J,O,Q,W
# Index 31:   .
# Index 32:   -
```
Note: README/comment claims "A-Z excluding I,O" but actual string also excludes J, Q, W. Doc drift — needs reconciliation.

## Pipeline
1. `collect_all_images(dataroot)` walks `{dataroot}/{train,val,test}/sharp/*` (sorted).
2. For each image, `ocr.predict(img_path)` → reads `rec_texts` (attribute or dict key) → joined.
3. Empty / failed detection → `"#" * 21`.
4. `text_to_indices()` upper-cases, maps unknown → `#` (idx 0), pads/truncates to 21.
5. Writes two newline-delimited files at dataset root.

## Risks / Gaps
- `lang="ch"` for Western plates is almost certainly wrong — at minimum needs an A/B against `lang="en"`.
- `dataroot` hard-coded in `__main__` to `dataset/quan_lp_dataset`.
- No progress bar beyond a print every 500 — long datasets blind.
- `os.environ["HOME"] = TMPDIR` before importing PaddleOCR — works around model cache permissions; document why.
- Failure counter conflates "no result" with "exception" — separate buckets would aid triage.

## Verification
- After run, `wc -l plate_info.txt plate_info_text.txt` should equal total images in `train+val+test`.
- `awk '{print NF}' plate_info.txt | sort -u` → must be `22` (image_name + 21 indices).
- Spot-check 10 entries against ground-truth (if available) for OCR accuracy.
