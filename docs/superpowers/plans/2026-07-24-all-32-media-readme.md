# All-32 GIF and README Publishing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish truthful 32-stage results with one successful, normalized 2x, infinitely looping GIF per passing stage and README content generated from the accepted shared-checkpoint report.

**Architecture:** GIF frame selection and validation are pure functions tested with synthetic frames. Batch recording consumes the immutable acceptance report and refuses mismatched checkpoints or partial rollouts. README results and gallery blocks are generated from the same report, eliminating manually maintained clear-rate claims.

**Tech Stack:** Python 3.13, Pillow 12.3, ImageIO 2.37, Markdown, pytest

## Global Constraints

- Every published clear must come from the exact shared checkpoint identified by the acceptance report.
- Record 15 stochastic rollouts and publish only a genuine successful rollout.
- Effective 2x playback is achieved by deterministic frame decimation, not unsafe sub-20 ms GIF delays.
- Every GIF must contain explicit `loop=0` infinite-loop metadata.
- Preserve the terminal clear frame and use common 256x240 output dimensions.
- Do not claim 32/32 unless the immutable acceptance report passes all 32 stages.

---

## File Structure

- `src/marioai/gif.py`: frame decimation, GIF encoding, and metadata validation.
- `src/marioai/record_gif.py`: model rollout selection and recorder CLI.
- `scripts/record_all_gifs.py`: report-driven 32-stage batch recorder.
- `src/marioai/readme.py`: generated result table and gallery rendering.
- `scripts/render_readme.py`: guarded README update CLI.
- `tests/test_gif.py`: timing, loop, dimensions, and terminal-frame tests.
- `tests/test_record_gif.py`: successful-rollout selection and report identity tests.
- `tests/test_readme.py`: generated status and no-overclaim tests.
- `README.md`: generated results blocks plus updated shared-policy explanation.
- `assets/gifs/*.gif`: successful stage evidence.

### Task 1: Browser-safe 2x infinite GIF encoder

**Files:**
- Create: `src/marioai/gif.py`
- Create: `tests/test_gif.py`
- Modify: `src/marioai/record_gif.py:1-77`

**Interfaces:**
- Produces: `select_2x_frames(frames: Sequence[np.ndarray]) -> list[np.ndarray]`
- Produces: `write_looping_gif(path: Path, frames: Sequence[np.ndarray], fps: int = 30) -> GifMetadata`
- Produces: `inspect_gif(path: Path) -> GifMetadata`
- Produces immutable `GifMetadata(width, height, frames, loop, durations_ms)`.

- [ ] **Step 1: Write failing encoder tests**

```python
from PIL import Image
import numpy as np
from marioai.gif import inspect_gif, select_2x_frames, write_looping_gif


def _frames(count):
    return [np.full((240, 256, 3), i, dtype=np.uint8) for i in range(count)]


def test_2x_selection_keeps_every_second_and_terminal_frame():
    selected = select_2x_frames(_frames(6))
    assert [int(frame[0, 0, 0]) for frame in selected] == [0, 2, 4, 5]


def test_encoder_writes_infinite_loop_and_safe_common_timing(tmp_path):
    path = tmp_path / "clear.gif"
    write_looping_gif(path, _frames(10), fps=30)
    metadata = inspect_gif(path)
    assert (metadata.width, metadata.height) == (256, 240)
    assert metadata.loop == 0
    assert set(metadata.durations_ms) == {30}
```

- [ ] **Step 2: Verify GIF tests fail**

Run: `.venv/bin/pytest tests/test_gif.py -v`

Expected: FAIL because `marioai.gif` does not exist.

- [ ] **Step 3: Implement deterministic frame decimation and encoding**

```python
def select_2x_frames(frames):
    if not frames:
        raise ValueError("cannot encode an empty frame sequence")
    selected = list(frames[::2])
    if selected[-1] is not frames[-1]:
        selected.append(frames[-1])
    return selected


def write_looping_gif(path, frames, fps=30):
    selected = select_2x_frames(frames)
    duration_ms = round(1000 / fps / 10) * 10
    imageio.mimsave(
        path,
        selected,
        format="GIF",
        duration=duration_ms / 1000,
        loop=0,
    )
    return inspect_gif(path)
```

`inspect_gif` must iterate every Pillow frame, collect `duration`, and reject a
missing loop key, nonzero loop, mixed dimensions, or durations below 20 ms.

- [ ] **Step 4: Replace adaptive implicit-speed encoding**

Remove `max_gif_frames` stride calculation from `record_gif.py`. Use
`write_looping_gif`; if file size control is needed, cap rollout policy steps
before recording rather than changing speed per stage.

Run: `.venv/bin/pytest tests/test_gif.py tests/test_wrappers.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/marioai/gif.py src/marioai/record_gif.py tests/test_gif.py
git commit -m "feat: encode looping Mario GIFs at effective 2x"
```

### Task 2: Acceptance-report-driven successful recording

**Files:**
- Modify: `src/marioai/record_gif.py`
- Create: `scripts/record_all_gifs.py`
- Create: `tests/test_record_gif.py`
- Delete: `scripts/record_all_gifs.sh`

**Interfaces:**
- Consumes: `EvaluationReport`, `sha256_file`, `ALL_LEVELS`, `write_looping_gif`.
- Produces: `choose_successful_rollout(runs: Sequence[RolloutCapture]) -> RolloutCapture`
- Produces: `record_from_report(model_path: Path, report_path: Path, output_dir: Path) -> dict[str, GifMetadata]`

- [ ] **Step 1: Write failing selection and identity tests**

```python
def test_recorder_selects_shortest_genuine_clear():
    runs = [
        RolloutCapture([], False, 3000, 100, 200),
        RolloutCapture([], True, 3200, 180, 140),
        RolloutCapture([], True, 3200, 160, 120),
    ]
    assert choose_successful_rollout(runs).steps == 120


def test_recorder_refuses_partial_only_runs():
    with pytest.raises(RuntimeError, match="no successful rollout"):
        choose_successful_rollout([RolloutCapture([], False, 3100, 120, 200)])


def test_batch_refuses_checkpoint_hash_mismatch(tmp_path):
    with pytest.raises(ValueError, match="checkpoint SHA-256"):
        record_from_report(other_model, accepted_report, tmp_path)
```

- [ ] **Step 2: Verify recorder tests fail**

Run: `.venv/bin/pytest tests/test_record_gif.py -v`

Expected: FAIL because report-driven recording does not exist.

- [ ] **Step 3: Implement report identity and exact seeded reruns**

Make `_rollout` accept a seed and return a `RolloutCapture` dataclass. The batch
recorder verifies the model SHA, then reruns the report's successful seeds for
each stage using stochastic actions. It verifies `flag_get` again before
writing. If the successful seed does not reproduce, run the report's remaining
14 seeds and fail the stage if none clears.

- [ ] **Step 4: Implement all-stage CLI**

```bash
python scripts/record_all_gifs.py \
  --model models/all32-final/best.zip \
  --report reports/final-acceptance.json \
  --out-dir assets/gifs
```

Write `assets/gifs/manifest.json` containing stage, successful seed, model SHA,
frame count, dimensions, mean duration, loop value, and GIF SHA-256. Never
write a GIF for a stage the report marks failed.

- [ ] **Step 5: Run recorder tests**

Run: `.venv/bin/pytest tests/test_record_gif.py tests/test_gif.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/marioai/record_gif.py scripts/record_all_gifs.py tests/test_record_gif.py
git rm scripts/record_all_gifs.sh
git commit -m "feat: record all stage clears from acceptance evidence"
```

### Task 3: Generated README results and gallery

**Files:**
- Create: `src/marioai/readme.py`
- Create: `scripts/render_readme.py`
- Create: `tests/test_readme.py`
- Modify: `README.md:1-145`

**Interfaces:**
- Consumes: `EvaluationReport`, GIF manifest.
- Produces: `render_results(report: EvaluationReport, gif_manifest: Mapping) -> str`
- Produces: `replace_generated_block(readme: str, name: str, content: str) -> str`
- Uses markers `<!-- ALL32_RESULTS:START -->` / `<!-- ALL32_RESULTS:END -->` and `<!-- ALL32_GALLERY:START -->` / `<!-- ALL32_GALLERY:END -->`.

- [ ] **Step 1: Write failing documentation tests**

```python
def test_results_render_all_32_rows_from_one_checkpoint():
    markdown = render_results(passing_report, complete_gif_manifest)
    assert markdown.count("| 1-") == 4
    assert markdown.count("| 8-") == 4
    assert "32 of 32 stages cleared" in markdown
    assert passing_report.checkpoint_sha256[:12] in markdown


def test_incomplete_report_never_claims_32_of_32():
    markdown = render_results(report_with_31_passes, partial_gif_manifest)
    assert "31 of 32 stages cleared" in markdown
    assert "32 of 32 stages cleared" not in markdown


def test_gallery_rejects_non_looping_gif():
    manifest = {**complete_gif_manifest, "1-1": {**entry, "loop": None}}
    with pytest.raises(ValueError, match="1-1.*loop"):
        render_results(passing_report, manifest)
```

- [ ] **Step 2: Verify README tests fail**

Run: `.venv/bin/pytest tests/test_readme.py -v`

Expected: FAIL because `marioai.readme` does not exist.

- [ ] **Step 3: Implement generated blocks**

Render rows in `ALL_LEVELS` order with columns Level, Passed, Clears/15, Max X,
and GIF. For passing stages, the GIF cell is
`![World 1-1 clear](assets/gifs/1-1.gif)`; failed stages display `—`.

The gallery uses an eight-world table with four stages per row. Validate that
each referenced GIF manifest entry has the accepted model SHA, `loop == 0`,
dimensions `256x240`, and a successful seed found in the report.

- [ ] **Step 4: Rewrite stale eight-stage narrative**

Replace claims about per-level specialists as the current result with a
historical section. Update the architecture diagram to IMPALA PPO and 12
`COMPLEX_MOVEMENT` actions. Insert generated markers and retain setup,
reproduction commands, license, and the prior 1-3 diagnosis as project history.

Run:

```bash
.venv/bin/python scripts/render_readme.py \
  --readme README.md \
  --report reports/final-acceptance.json \
  --gif-manifest assets/gifs/manifest.json
```

- [ ] **Step 5: Run documentation tests**

Run: `.venv/bin/pytest tests/test_readme.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/marioai/readme.py scripts/render_readme.py tests/test_readme.py README.md
git commit -m "feat: generate truthful all-32 README results"
```

### Task 4: Record, validate, and publish final evidence

**Files:**
- Modify: `assets/gifs/1-1.gif` through `assets/gifs/8-4.gif`
- Create: `assets/gifs/manifest.json`
- Modify: `README.md`

**Interfaces:**
- Consumes: passing `reports/final-acceptance.json` and its exact shared checkpoint.
- Produces: 32 validated GIFs and final generated README.

- [ ] **Step 1: Assert acceptance before recording**

Run:

```bash
.venv/bin/python -c "from pathlib import Path; from marioai.results import EvaluationReport; r=EvaluationReport.read(Path('reports/final-acceptance.json')); assert r.passed; print(r.checkpoint_sha256)"
```

Expected: prints the exact accepted checkpoint SHA and exits zero. If it fails,
do not publish or claim 32/32.

- [ ] **Step 2: Record all successful seeds**

Run:

```bash
.venv/bin/python scripts/record_all_gifs.py \
  --model models/all32-final/best.zip \
  --report reports/final-acceptance.json \
  --out-dir assets/gifs
```

Expected: 32 successful GIFs and a 32-entry manifest.

- [ ] **Step 3: Regenerate README**

Run:

```bash
.venv/bin/python scripts/render_readme.py \
  --readme README.md \
  --report reports/final-acceptance.json \
  --gif-manifest assets/gifs/manifest.json
```

Expected: README reports exactly the report's coverage and references only
validated GIFs.

- [ ] **Step 4: Validate every asset**

Run:

```bash
.venv/bin/pytest tests/test_gif.py tests/test_record_gif.py tests/test_readme.py -v
.venv/bin/python scripts/record_all_gifs.py \
  --model models/all32-final/best.zip \
  --report reports/final-acceptance.json \
  --out-dir assets/gifs --validate-only
```

Expected: 32 files at 256x240, safe common timing, `loop=0`, successful seeds,
matching model SHA, and no metadata failures.

- [ ] **Step 5: Run full project verification**

Run: `.venv/bin/pytest -q`

Expected: all tests pass.

Run: `git diff --check`

Expected: no whitespace errors.

- [ ] **Step 6: Commit and push final evidence**

```bash
git add assets/gifs README.md reports/final-acceptance.json reports/aws-spend.json
git commit -m "docs: publish all-32 shared-policy clears"
git push origin main
```

Expected: GitHub renders every GIF at normalized effective 2x playback and
restarts it indefinitely.
