# Plan: Fix `util/apply_disk_blur_mod.py`

## Context

`util/apply_disk_blur_mod.py` applies a disk-blur + augmentation pipeline that
simulates CCTV and dashcam footage for LPDGAN training pairs. A review found 14
issues: 5 correctness bugs, 2 unsound-physics degradation models, and 7
cosmetic/hygiene defects. The disk-blur core itself (pillbox kernel) is correct
and is not touched by this plan.

Two of the bugs (one-sided motion-blur kernel, wrap-around rolling shutter)
inject **structured non-physical artifacts** that a GAN will memorise — they are
the highest priority. All 14 are scoped into 5 steps below.

## Findings reference

| # | Location | Type | Summary |
|---|----------|------|---------|
| 1 | `_apply_motion_blur` ~L317 | bug | One-sided kernel translates the image as well as blurring it |
| 2 | `_apply_rolling_shutter` ~L333 | bug | `np.roll` wraps edge pixels into a visible seam; `borderMode` is dead code |
| 3 | `_apply_barrel_distortion` ~L57 | bug | Separable per-axis model, not radially symmetric; `k >= 0` only |
| 4 | `cctv_pipeline` ~L119, `dual_pipeline` ~L229 | bug | `int(w*scale)` can be 0 for small inputs and crashes `cv2.resize` |
| 5 | `_apply_dirt_rain` ~L388 | bug | Unseeded `import random` breaks the `--seed` reproducibility contract |
| 6 | `_apply_interlace` ~L148 | unsound | 1px shift of even rows is not interlace combing |
| 7 | `_apply_night_noise` ~L156 | unsound | Amplifies one chroma channel -> colour cast, not the IR/grayscale look claimed |
| 8 | `_apply_variable_exposure` ~L367 | cosmetic | "overbright"/"underexposed" comments swapped (gamma 0.6-0.9 darkens) |
| 9 | `dual_pipeline` ~L197, L216 | cosmetic | "50% probability each" comments do not match the actual probabilities |
| 10 | `dashcam_pipeline` docstring ~L287 | cosmetic | Says rolling shutter is "column-wise"; implementation is row-wise |
| 11 | `_apply_dirt_rain` ~L417 | perf | `cv2.blur` recomputed once per raindrop (5-12x per call) |
| 12 | `main` ~L498 | robustness | No input validation; grayscale/RGBA PIL inputs crash downstream |
| 13 | `cctv_pipeline` ~L121 | realism | `INTER_LANCZOS4` upscale; real low-end CCTV upscalers are bilinear |
| 14 | `cctv_pipeline` ~L107 | cosmetic | "Fixed viewpoint offset" comment contradicts random-per-call behaviour |

---

## Step 1 — Symmetric motion-blur PSF and non-wrapping rolling-shutter remap

**Intent**: Implement a symmetric motion-blur point-spread function and a
non-wrapping rolling-shutter remap. These two functions currently emit
structured artifacts (image translation, an edge seam) on top of the intended
degradation, which a deblur GAN would learn as signal.

**Findings**: #1, #2

**Changes**:
- `_apply_motion_blur` (~L317): the kernel line runs `for i in range(size)` from
  the center in one direction only, so the kernel centroid is offset from the
  anchor and the convolution translates the image. Rebuild the kernel symmetric
  about the center (plot points in both directions, e.g. `i` from `-size` to
  `+size`), keep it normalised.
- `_apply_rolling_shutter` (~L333): `np.roll(col_indices, offset)` wraps pixels
  from one edge to the other, producing a seam; because the rolled indices are
  always in range, the `cv2.BORDER_REFLECT` arg never triggers. Replace the
  rolled index map with an additive offset (`map_x[j, :] = arange(w) - offset`)
  so out-of-range coordinates fall through to `cv2.BORDER_REFLECT` in `remap`.

**Acceptance**:
- Motion-blur kernel centroid sits at the geometric center within rounding.
- Rolling-shutter output has no wrap-around column seam.
- `cctv_pipeline`, `dashcam_pipeline`, and `dual_pipeline` all still run.

**Out of scope**: changing probability gates or adding new degradation modes.

---

## Step 2 — True radial barrel/pincushion distortion

**Intent**: Implement a true radial lens-distortion model to replace the current
separable per-axis approximation. The existing model distorts the x and y axes
independently, so it is not rotationally symmetric and only ever produces one
distortion direction.

**Findings**: #3

**Changes**:
- `_apply_barrel_distortion` (~L57): x-displacement is computed from `xn**2`
  alone and y-displacement from `yn**2` alone. Replace with a genuine radial
  model: for each pixel compute `r2 = xn**2 + yn**2`, apply one scalar factor
  `(1 + k*r2)` to both `xn` and `yn`, and build full 2D `map_x` / `map_y`
  instead of two 1D arrays repeated.
- Widen the random `k` range to include negative values so genuine barrel
  (`k < 0`) and pincushion (`k > 0`) are both reachable, matching the docstring.

**Acceptance**:
- A centered circle maps to a centered circle, not an ellipse.
- Both barrel and pincushion distortion are reachable across random `k`.
- All three pipelines still run.

**Out of scope**: lens-specific calibration constants or chromatic aberration.

---

## Step 3 — Resize crash guard and seed reproducibility

**Intent**: Implement a minimum-size clamp on the resolution-downscale step and
route dirt/rain randomness through the seeded RNG. The pipeline currently
crashes on small inputs and silently ignores `--seed` for two of its three
modes.

**Findings**: #4, #5

**Changes**:
- `cctv_pipeline` (~L119) and `dual_pipeline` (~L229): `int(w*scale)` /
  `int(h*scale)` can evaluate to 0 for small inputs (license-plate crops),
  raising `cv2.error`. Clamp `new_w` and `new_h` to a minimum of 1.
- `_apply_dirt_rain` (~L388): uses an unseeded `import random`, so `--seed` does
  not make `dashcam` or `dual` mode reproducible. Route all dirt/rain randomness
  through the same `np.random` stream the pipelines seed (or seed a `random`
  instance from the passed seed). Update the docstring to drop the
  "system-entropy unrepeatable placement" claim.

**Acceptance**:
- Running the pipeline on a 4x4-pixel input does not raise.
- An identical `--seed` produces byte-identical output in `dashcam` and `dual`
  mode across two runs.

**Out of scope**: changing the CLI flag interface or which effects are random.

---

## Step 4 — Physically-grounded interlace and night-noise models

**Intent**: Implement physically-grounded interlace and night-noise models to
replace the current unsound approximations. Both functions claim an effect in
their docstrings that the code does not actually produce.

**Findings**: #6, #7

**Changes**:
- `_apply_interlace` (~L148): currently shifts even rows right by 1px, which is
  not interlace combing. Model field weave instead — split into even/odd fields,
  apply a small simulated-motion offset to one field relative to the other, then
  re-interleave, so comb teeth appear on moving content and static content is
  untouched. Keep it cheap.
- `_apply_night_noise` (~L156): amplifies one chroma channel, producing a colour
  cast, despite the docstring claiming an IR / grayscale shift. Replace the
  per-channel multiply with a desaturation toward luma (blend toward grayscale)
  to model the IR monochrome response, then add noise; an optional faint tint is
  acceptable.

**Acceptance**:
- Interlace output shows row-pair misalignment proportional to simulated motion
  and zero shift on fully static content.
- Night-noise output has lower saturation than the input, not higher.
- All three pipelines still run.

**Out of scope**: calibrated sensor IR response curves or temporal multi-frame
interlace.

---

## Step 5 — Cosmetic and hygiene cleanup

**Intent**: Refactor comments, docstrings, a redundant per-iteration blur call,
input validation, and an interpolation choice for cleanup. No changes to
degradation strength or pipeline structure.

**Findings**: #8, #9, #10, #11, #12, #13, #14

**Changes**:
- #8 `_apply_variable_exposure` (~L367): fix the swapped "overbright" /
  "underexposed" comments — gamma 0.6-0.9 darkens.
- #9 `dual_pipeline` (~L197, L216): correct the "50% probability each" comments
  to match the actual probabilities, or generalise the wording.
- #10 `dashcam_pipeline` docstring (~L287): rolling shutter is row-wise, not
  "column-wise"; fix the wording.
- #11 `_apply_dirt_rain` (~L417): hoist the `cv2.blur` call out of the raindrop
  loop so it runs once per call instead of 5-12 times.
- #12 `main` (~L498): `Image.open` may yield grayscale or RGBA arrays; validate
  and convert to 3-channel RGB at the boundary before the pipeline runs.
- #13 `cctv_pipeline` (~L121): switch the upscale from `INTER_LANCZOS4` to
  `INTER_LINEAR` to match real low-end CCTV upscalers; keep `INTER_AREA` on the
  downscale.
- #14 `cctv_pipeline` (~L107): fix the "Fixed viewpoint offset" comment that
  contradicts the random-per-call behaviour.

**Acceptance**:
- All listed comments and docstrings match actual code behaviour.
- `cv2.blur` runs once per `_apply_dirt_rain` invocation.
- Non-RGB inputs are handled without a crash.
- `ruff` is clean on the file.

**Out of scope**: behavioural changes to degradation strength or refactoring the
pipeline structure.
