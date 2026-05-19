# Daily 05_13

## Summary
Dataset pipeline groundwork: adnl + LPBlur symlink scripts with 90/10 train/test split, PaddleOCR-driven `plate_info.txt`/`plate_info_text.txt` generation, and a sweep of legacy `util/` scripts into `scripts/` alongside an OMC skill + agent doc refresh.

## Done
- Added [scripts/symlink_and_split_adnl.py](../../../scripts/symlink_and_split_adnl.py) — seeded (42) shuffle, 90/10 split for `adnl/` (andan+ninhloc merged) and per-variant `LPBlur/{sharp,blur}/` symlinks under `train/`/`test/`. Commit `5fd3d1d`.
- Added bash helper [scripts/symlink_adnl.sh](../../../scripts/symlink_adnl.sh) and federated-split planning doc (later removed). Commit `dace65d`.
- Rewrote [scripts/generate_plate_info.py](../../../scripts/generate_plate_info.py) — switched to `PaddleOCR.predict()`, emits two artifacts at dataset root: `plate_info.txt` (21 indices/line, 33-class Western charset) and `plate_info_text.txt` (raw OCR text). Backs up existing files to `.bak`. Commit `40f4a26`.
- Config + housekeeping pass (commit `ad5caed`, +570/-248):
  - Moved `util/{apply_disk_blur_mod.py,generate_plate_info.py}` → `scripts/`; deleted legacy `util/generate_plate_info_chunked.py` + `generate_plate_info_resume.py`.
  - Added [AGENTS.md](../../../AGENTS.md) (+136) and refreshed [CLAUDE.md](../../../CLAUDE.md) (+72).
  - Added OMC skill stubs: `debug-issue`, `explore-codebase`, `refactor-safely`, `review-changes`, `omc-reference` under `.claude/skills/`.
  - `.gitignore` widened for dataset/image formats.

## Evidence
- [vibe-updates/05_13/report/report_dataset-symlinks.md](../report/report_dataset-symlinks.md)
- [vibe-updates/05_13/report/report_paddleocr-plate-info.md](../report/report_paddleocr-plate-info.md)
- [vibe-updates/05_13/report/report_config-cleanup.md](../report/report_config-cleanup.md)
- [vibe-updates/05_13/artifact/manifest_dataset-symlinks.json](../artifact/manifest_dataset-symlinks.json)
- [vibe-updates/05_13/artifact/manifest_paddleocr-plate-info.json](../artifact/manifest_paddleocr-plate-info.json)
- [vibe-updates/05_13/artifact/manifest_config-cleanup.json](../artifact/manifest_config-cleanup.json)

## Blockers
- `generate_plate_info.py` hard-codes `dataroot = "dataset/quan_lp_dataset"` in `__main__` — runner is non-portable; need CLI args or env override before running on `adnl/` or `LPBlur/`.
- PaddleOCR `lang="ch"` is used for **Western** plates — likely suboptimal recognition; needs validation pass against ground truth.
- Federated-split plan doc was deleted in `ad5caed` (`.specstory/.../split-adnl-to-train-test-federated-map.md`). Intent vs. final approach unclear — recover from git or restate.

## Next
- See: [vibe-updates/05_13/todo/todo.md](../todo/todo.md)
