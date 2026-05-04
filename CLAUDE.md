# LPDGAN - License Plate Deblurring GAN

## Project Structure
- `main.py` — Entry point, handles train/test mode routing
- `train.py` — Training loop
- `test.py` — Test/inference loop
- `inference.py` — Standalone inference script
- `data/LPBlur_dataset.py` — Dataset class, expects `{dataroot}/{mode}/blur` and `{dataroot}/{mode}/sharp`

## Run Commands
Use `uvr` (uv run) instead of bare `python`:
```bash
uvr main.py --mode train --dataroot ./dataset
uvr main.py --mode test --dataroot ./dataset
uvr inference.py --model_path checkpoints/LPDGAN/latest.pth --input blur.jpg --output sharp.jpg
```

## Dataset Structure
```
./dataset/
  train/blur/  train/sharp/
  test/blur/   test/sharp/
```

## Key Files
- `models/LPDGAN.py` — Model architecture
- `checkpoints/LPDGAN/` — Saved checkpoints
- `results/` — Test outputs

## File Transfer to tnadmin
- `scp <file> tnadmin:/G:/nblongT04/LPDGAN/` — Transfer checkpoints/results to work server
- Remote path follows pattern `tnadmin:/G:/nblongT04/<project_name>/`

## Gotchas
- **Dataset structure is the #1 failure point** — test mode fails with "num_samples=0" when dataset is flat `dataset/blur/` instead of `dataset/train/blur/` and `dataset/test/blur/`
- `load_iter` defaults to 200, override with `--load_iter` for specific checkpoint
- `num_test` defaults to 1000 for test mode

## Tools
- `rtk gain` — Show token savings analytics
- `rtk discover` — Analyze Claude Code history for missed optimization opportunities

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:ca08a54f -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd dolt push
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->
