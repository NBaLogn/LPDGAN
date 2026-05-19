# Report: Dataset Symlinks (adnl + LPBlur)

**Commits:** `5fd3d1d`, `dace65d` (05-13); follow-up `145c265` (05-14).

## Change Summary
| Path | Action | Notes |
|------|--------|-------|
| `scripts/symlink_and_split_adnl.py` | added (75 LOC) | Seeded 90/10 split, merges andan + ninhloc into `adnl/`, separate per-variant split for LPBlur. |
| `scripts/symlink_adnl.sh` | added (22 LOC) | Bash variant; precedes Python rewrite. |
| `.specstory/.../split-adnl-to-train-test-federated-map.md` | added then removed | Design notes — removed in `ad5caed`. |

## Behavior
- `SRC_ROOT = /Users/logan/Developer/vibes/WORK/LIPLA/LPDGAN/dataset` (machine-local absolute path — not portable).
- adnl: walks `dataset/{andan,ninhloc}/*/*.jpg`, link name `{name}-{subdir}-{filename}` to deduplicate cross-source collisions.
- LPBlur: walks `dataset/LPBlur/{sharp,blur}/*/*.jpg`, link name = bare filename (assumes filenames already unique per-variant).
- Split: `random.seed(42); random.shuffle(...); 90%/10%`. Re-runs with same seed are deterministic.
- Pre-existing symlinks are unlinked before re-creation (added in fix `145c265`).

## Risks / Gaps
- `SRC_ROOT` is absolute — won't run on tnadmin or any non-Logan host without edit.
- LPBlur sharp/blur are split *independently* with the same seed — only correct if the underlying file ordering is identical across variants. Worth verifying pairs stay aligned.
- No CLI args, no logging — stdout prints only.

## Verification
- Inspect `find dataset/adnl/train/sharp -maxdepth 1 -type l | wc -l` and compare to 0.9 × total.
- Spot-check 5 random links resolve to existing source images.
