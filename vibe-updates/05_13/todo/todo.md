# Todo 05_13

## Top
- [ ] Parametrize `scripts/generate_plate_info.py` — accept `--dataroot` CLI arg; drop hard-coded `dataset/quan_lp_dataset`.
- [ ] Validate PaddleOCR `lang="ch"` vs. an English/Latin model on Western plates; benchmark accuracy on a 50-image sample.
- [ ] Run `scripts/symlink_and_split_adnl.py` on the workstation; verify `train`/`test` counts match seed-42 split.
- [ ] Decide fate of federated-split design (doc was deleted in `ad5caed`) — recover, rewrite, or formally drop.
- [ ] Add minimal pytest for `text_to_indices()` (boundary: unknown char → `#`, overflow truncate to 21, padding fill).
- [ ] Document the 33-class charset choice in README (why I,J,O,Q,W excluded from A-Z).
- [ ] Pull forward the 05-14 fix (`145c265` — remove existing symlink before re-create) — confirm idempotent re-runs work for both `adnl` and `LPBlur`.
- [ ] Wire `bd` issues for each Blocker; close 05_13 work via `bd close` after verification.

## Notes
- 4 commits dated 05-13: `40f4a26`, `ad5caed`, `5fd3d1d`, `dace65d`. Follow-up fix `145c265` landed 05-14.
- Net script churn: `util/` retired; canonical home is `scripts/`.
- PaddleOCR output files land at *dataset root*, not per-split — downstream loaders must read from `<dataroot>/plate_info.txt`.
