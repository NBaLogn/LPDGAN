# Plan: `no-plate-info2` branch — train without `plate_info.txt`

## Context

The user wants a new git worktree containing a branch named `no-plate-info2`, forked from commit `d03eeb47d2eceb713efec7ec0e289fa176d12833` ("Update README.md", 2024-05-30, by haoyGONG). The branch must let `train` mode run end-to-end on a dataset that has **no `plate_info.txt`** file.

Why this matters: the upstream baseline at `d03eeb47` requires `plate_info.txt` for the auxiliary OCR loss `PlateNum_L1`. When the file is absent, `LPBlurDataset.__init__` crashes in `pd.read_csv`. We want a graceful degradation path so the model can be trained on plate-less datasets while keeping the rest of the training pipeline (GAN + L1 + perceptual + multi-scale discriminators) intact.

A reference commit `295e15d` already solves the conditional plumbing on a different branch, but it also bundles unrelated changes (MPS device support, tensor `.contiguous()` fixes, `cuda()` → `.to(device)`). Those do not belong in a clean branch from `d03eeb47`. We hand-apply only the plate-info conditional logic.

## Branch & worktree creation

```bash
# From repo root (any existing worktree)
git fetch --all
git worktree add -b no-plate-info2 \
  ../no-plate-info2 \
  d03eeb47d2eceb713efec7ec0e289fa176d12833
cd ../no-plate-info2
```

Resulting layout (sibling to current worktree):
- Worktree path: `/Users/logan/Developer/vibes/WORK/LIPLA/LPDGAN/.claude/worktrees/no-plate-info2` (or wherever the user's `wt` config places sibling worktrees — adjust path accordingly).
- Branch: `no-plate-info2`, HEAD at `d03eeb47`.

## Critical files to modify

Only **two** source files change:

1. `data/LPBlur_dataset.py` — gate the `plate_info.txt` read and the dict assembly.
2. `models/LPDGAN.py` — gate `set_input` and `backward_G` on a `has_plate_info` flag.

No new CLI flags, no new options, no changes to `train.py`, `test.py`, `main.py`, `options/`, or `models/swin_transformer.py`. The generator already emits `plate1, plate2` heads regardless — we simply stop supervising them when `plate_info.txt` is absent.

## Change 1: `data/LPBlur_dataset.py`

**At `__init__`, replace the unconditional read (around lines 24–28 of the baseline file):**

```python
if self.opt.mode == 'train':
    plate_info_path = os.path.join(opt.dataroot, 'plate_info.txt')
    if os.path.exists(plate_info_path):
        df = pd.read_csv(plate_info_path, header=None,
                         names=['ImageName', 'PlateInfo'])
        self.txt = df.set_index('ImageName')['PlateInfo'].to_dict()
        self.has_plate_info = True
        logger.info(f'Loaded plate_info from {plate_info_path}')
    else:
        self.has_plate_info = False
        logger.info('plate_info.txt not found, skipping auxiliary plate loss')
    self.transform_fn  = aug.get_transforms(size=(112, 224))
    self.transform_fn1 = aug.get_transforms(size=(56, 112))
    self.transform_fn2 = aug.get_transforms(size=(28, 56))
    self.transform_fn3 = aug.get_transforms(size=(14, 28))
else:
    # unchanged test-mode block
    ...
```

**At `__getitem__`, the train-mode return (around lines 72–86) becomes:**

```python
if self.opt.mode == 'train':
    base = {
        'A': blur_image, 'B': sharp_image,
        'A_paths': self.blur[idx], 'B_paths': self.sharp[idx],
        'A1': blur_image1, 'B1': sharp_image1,
        'A2': blur_image2, 'B2': sharp_image2,
        'A3': blur_image3, 'B3': sharp_image3,
    }
    if self.has_plate_info:
        plate_info = self.txt[os.path.basename(self.sharp[idx])]
        try:
            plate_info = np.fromstring(plate_info, sep=' ')
        except (SyntaxError, ValueError) as e:
            print(f"Error restoring array: {e}")
        plate_info = torch.from_numpy(plate_info)
        base['plate_info'] = plate_info
    return base
```

Rationale: a single `base` dict avoids duplicating the 10 image-tensor entries across two `return` statements. (The reference 295e15d duplicated them — we improve on it slightly per DRY.)

## Change 2: `models/LPDGAN.py`

**At `set_input` (around line 81 of the baseline file):**

```python
def set_input(self, input):
    self.real_A  = input['A'].to(self.device)
    self.real_B  = input['B'].to(self.device)
    self.real_A1 = input['A1'].to(self.device)
    self.real_B1 = input['B1'].to(self.device)
    self.real_A2 = input['A2'].to(self.device)
    self.real_B2 = input['B2'].to(self.device)
    self.real_A3 = input['A3'].to(self.device)
    self.real_B3 = input['B3'].to(self.device)
    self.image_paths = input['A_paths']
    if 'plate_info' in input:
        self.plate_info = input['plate_info'].to(self.device)
        self.has_plate_info = True
    else:
        self.has_plate_info = False
```

(Keep whatever the baseline currently does for `real_A/B*` — only add the `plate_info` conditional. The exact set_input body at `d03eeb47` may differ in formatting; preserve it.)

**At `backward_G` (around lines 203–206 of the baseline file):**

```python
if self.has_plate_info:
    self.loss_PlateNum_L1 = (
        self.criterionL1(self.plate1, self.plate_info)
        + self.criterionL1(self.plate2, self.plate_info)
    ) / 2 * 0.01
    self.loss_G = (
        self.loss_G_GAN + self.loss_G_s + self.loss_G_L1
        + self.loss_P_loss + 0.1 * self.loss_PlateNum_L1
    )
else:
    self.loss_PlateNum_L1 = torch.tensor(0.0, device=self.device)
    self.loss_G = (
        self.loss_G_GAN + self.loss_G_s + self.loss_G_L1
        + self.loss_P_loss
    )
self.loss_G.backward()
```

Rationale: when plate_info is absent we record a 0.0 scalar for `loss_PlateNum_L1` so any logging / `loss_names` reporting that references it still functions (the baseline lists it in `self.loss_names`). This avoids touching `loss_names` or the visdom/print code in `train.py`.

## What we deliberately do NOT change

- **`models/swin_transformer.py`** — generator still produces `plate1, plate2`. They become unsupervised heads; that is fine.
- **`util/generate_plate_info.py`** — utility for producing `plate_info.txt`; out of scope.
- **`options/`** — no new flag. Presence/absence of the file is the switch. This is simpler than a `--no_plate` flag and matches how `295e15d` solved it.
- **`test.py` / `inference.py`** — test mode already does not read `plate_info`.
- **Device-handling, MPS, `.contiguous()` patches from 295e15d** — out of scope for this branch.

## Verification

End-to-end smoke tests, run from inside the new worktree:

1. **No `plate_info.txt` — should succeed**
   ```bash
   # ensure plate_info.txt is absent
   ls dataset/plate_info.txt 2>/dev/null && echo "REMOVE THIS FOR TEST"
   uvr main.py --mode train --dataroot ./dataset --niter 1 --niter_decay 0 --batch_size 2
   ```
   Expected:
   - Log line: `plate_info.txt not found, skipping auxiliary plate loss`
   - Training loop runs ≥1 iteration without `KeyError`, `FileNotFoundError`, or `AttributeError: ... plate_info`.
   - `loss_PlateNum_L1` printed as `0.000`.

2. **With `plate_info.txt` present — regression check**
   ```bash
   # ensure plate_info.txt exists at dataroot
   uvr main.py --mode train --dataroot ./dataset --niter 1 --niter_decay 0 --batch_size 2
   ```
   Expected:
   - Log line: `Loaded plate_info from ./dataset/plate_info.txt`
   - `loss_PlateNum_L1` is non-zero and finite.
   - Loss curve qualitatively matches the baseline `d03eeb47` for the first iteration (sanity: same RNG seed → same `loss_G_L1`).

3. **Test mode — unaffected**
   ```bash
   uvr main.py --mode test --dataroot ./dataset --load_iter 200
   ```
   Expected: no reference to `plate_info` in stack traces. Test mode never touched it at baseline.

4. **Import-level smoke**
   ```bash
   uvr -c "from data.LPBlur_dataset import LPBlurDataset; from models.LPDGAN import LPDGAN; print('ok')"
   ```
   (Per global rule, prefer a one-liner script over `-c` for nontrivial code. The line above is trivial.)

## Acceptance criteria

- [ ] Worktree exists at sibling path, branch `no-plate-info2` checked out at `d03eeb47`.
- [ ] `data/LPBlur_dataset.py` no longer raises when `plate_info.txt` is missing.
- [ ] `models/LPDGAN.py` no longer references `self.plate_info` unless `has_plate_info=True`.
- [ ] Both verification scenarios 1 and 2 above succeed.
- [ ] No changes outside the two listed files.
- [ ] Diff vs `d03eeb47` is ≤ ~50 lines.

## Rollout

1. Create the worktree + branch (see "Branch & worktree creation").
2. Apply Change 1 and Change 2.
3. Run verification scenarios 1, 2, 3, 4 in order.
4. Commit:
   ```
   feat: graceful skip of plate_info.txt in train mode

   Allow LPBlurDataset and LPDGAN to run end-to-end when
   plate_info.txt is absent from the dataroot. The auxiliary
   PlateNum_L1 OCR loss is omitted and reported as 0.0 in that
   case; all other losses (GAN, L1, perceptual, multi-scale D)
   are unchanged. Presence of plate_info.txt at the dataroot is
   the switch — no new CLI flag.
   ```
5. Push: `git push -u origin no-plate-info2`.
6. (Optional) Open PR against the appropriate base branch.

## Risk & mitigation

| Risk | Mitigation |
|------|------------|
| `loss_names` list expects `PlateNum_L1` everywhere → KeyError in visualizer | Always set `self.loss_PlateNum_L1 = torch.tensor(0.0, …)` so attribute exists. |
| Future code adds another consumer of `self.plate_info` and forgets the guard | Setting `self.has_plate_info` flag makes the intent explicit and greppable. |
| Test mode regression | We do not touch test-mode code paths; verification scenario 3 confirms. |
| Different `set_input` formatting at `d03eeb47` vs current main | Plan instructs to preserve existing body and only add the `plate_info` conditional. |
