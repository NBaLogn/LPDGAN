# Report: Config + Script Cleanup + Agent Docs

**Commit:** `ad5caed` (05-13). Net +570/-248 across 16 files.

## Moves
| From | To |
|------|-----|
| `util/apply_disk_blur_mod.py` | `scripts/apply_disk_blur_mod.py` |
| `util/generate_plate_info.py` | `scripts/generate_plate_info.py` (+4 lines tweak) |

## Deletions
- `util/generate_plate_info_chunked.py` (73 LOC) — superseded by single-pass rewrite.
- `util/generate_plate_info_resume.py` (87 LOC) — superseded.
- `.specstory/.../split-adnl-to-train-test-federated-map.md` (65 LOC) — design doc removed (only added one commit earlier in `dace65d`).

## Additions
- [AGENTS.md](../../../AGENTS.md) — 136 lines of agent guidance.
- `.claude/skills/{debug-issue,explore-codebase,refactor-safely,review-changes}/skill.md` — 4 OMC skill stubs (~28 LOC each).
- `.claude/skills/omc-reference/SKILL.md` — 141-line OMC reference catalog.

## Modifications
- `CLAUDE.md` (+72) — project instructions expanded.
- `.claude/CLAUDE.md` (+65) — OMC operating principles, delegation rules, model routing.
- `.claude/settings.json` (+52/-) — hook + permission updates.
- `.claude/settings.local.json` (+4/-)
- `.gitignore` (+7) — broader dataset/image format coverage.

## Rationale Inferred
- Consolidate runnable scripts into `scripts/`, retiring `util/` as legacy.
- Onboard OMC orchestration layer (skills + agent docs).
- Stop tracking dataset binaries / image outputs.

## Risks / Gaps
- 5 deleted files lost their git history at the new location for the 2 moves — `git log --follow` still works but blame may be split.
- Federated-split design doc only existed for one commit — knowledge gap if that approach is revisited.
- Settings drift between `settings.json` (committed) vs. `settings.local.json` (developer-local) — confirm secrets/hosts only in `.local`.
