# All-32 Shared Policy Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and locally validate the configurable 32-stage, 12-action, IMPALA-PPO training and acceptance-evaluation pipeline for one shared Mario policy.

**Architecture:** A canonical stage manifest and action-set resolver feed fixed emulator workers assigned with deterministic stage weights. PPO uses a custom IMPALA residual feature extractor, while evaluation writes immutable per-rollout JSON evidence for one checkpoint. Recovery tools can add policy-cadence-compatible routes to the same shared checkpoint without introducing specialist heads.

**Tech Stack:** Python 3.13, Gymnasium 1.3, gym-super-mario-bros 9.1, Stable-Baselines3 2.9, PyTorch 2.13, PyYAML, pytest

## Global Constraints

- The acceptance artifact is one shared policy checkpoint for all 32 stages.
- A stage passes with at least one clear in exactly 15 stochastic rollouts.
- Use `COMPLEX_MOVEMENT` with 12 discrete actions throughout new training, evaluation, solving, cloning, and recording.
- Keep inference pixel-only; emulator state may be used only to generate or score evidence.
- Recurrent policies and specialist models are excluded unless separately authorized after evidence review.
- Do not start paid AWS resources in this plan.

---

## File Structure

- `src/marioai/levels.py`: canonical 32-stage manifest and validation.
- `src/marioai/actions.py`: named Mario action sets and route compatibility checks.
- `src/marioai/sampling.py`: deterministic weighted worker-to-stage assignment.
- `src/marioai/features.py`: IMPALA residual Stable-Baselines3 feature extractor.
- `src/marioai/envs.py`: environment construction using explicit action and worker assignments.
- `src/marioai/train.py`: config parsing, model construction, resume validation, and PPO execution.
- `src/marioai/evaluate.py`: seeded stochastic rollouts and acceptance-report CLI.
- `src/marioai/results.py`: evaluation report schema, checkpoint hashing, and aggregation.
- `src/marioai/demonstrations.py`: route validation and balanced multi-route datasets.
- `scripts/behavior_clone_route.py`: shared-policy behavior-cloning CLI.
- `scripts/solve_level.py`: action-set-aware ground, water, and maze route search.
- `configs/all32.yaml`: two-phase 32-stage training configuration.
- `tests/test_levels.py`: stage and action configuration tests.
- `tests/test_sampling.py`: weighted assignment tests.
- `tests/test_features.py`: IMPALA shape and PPO integration tests.
- `tests/test_train.py`: configuration and resume-compatibility tests.
- `tests/test_evaluate.py`: rollout evidence and 32-stage acceptance tests.
- `tests/test_demonstrations.py`: shared route dataset and special-mode solver tests.

### Task 1: Canonical 32-stage manifest

**Files:**
- Create: `src/marioai/levels.py`
- Create: `tests/test_levels.py`
- Modify: `configs/default.yaml:1-3`
- Create: `configs/all32.yaml`

**Interfaces:**
- Produces: `ALL_LEVELS: tuple[str, ...]`
- Produces: `WORLD_GROUPS: dict[str, tuple[str, ...]]`
- Produces: `validate_levels(levels: Sequence[str]) -> tuple[str, ...]`

- [ ] **Step 1: Write failing manifest tests**

```python
import pytest
from marioai.levels import ALL_LEVELS, WORLD_GROUPS, validate_levels


def test_manifest_contains_all_32_stages_in_world_order():
    assert ALL_LEVELS == tuple(
        f"{world}-{stage}"
        for world in range(1, 9)
        for stage in range(1, 5)
    )
    assert WORLD_GROUPS["worlds_1_4"] == ALL_LEVELS[:16]
    assert WORLD_GROUPS["worlds_5_8"] == ALL_LEVELS[16:]


def test_validate_levels_rejects_unknown_and_duplicates():
    with pytest.raises(ValueError, match="unknown Mario stage"):
        validate_levels(["9-1"])
    with pytest.raises(ValueError, match="duplicate Mario stage"):
        validate_levels(["1-1", "1-1"])
```

- [ ] **Step 2: Verify the tests fail**

Run: `.venv/bin/pytest tests/test_levels.py -v`

Expected: FAIL because `marioai.levels` does not exist.

- [ ] **Step 3: Implement the manifest**

```python
# src/marioai/levels.py
from collections.abc import Sequence

ALL_LEVELS = tuple(
    f"{world}-{stage}"
    for world in range(1, 9)
    for stage in range(1, 5)
)
WORLD_GROUPS = {
    "worlds_1_4": ALL_LEVELS[:16],
    "worlds_5_8": ALL_LEVELS[16:],
    "all": ALL_LEVELS,
}


def validate_levels(levels: Sequence[str]) -> tuple[str, ...]:
    result = tuple(levels)
    unknown = sorted(set(result) - set(ALL_LEVELS))
    if unknown:
        raise ValueError(f"unknown Mario stage(s): {', '.join(unknown)}")
    if len(set(result)) != len(result):
        raise ValueError("duplicate Mario stage in level list")
    if not result:
        raise ValueError("at least one Mario stage is required")
    return result
```

Add `levels.all` and explicit `levels.phase_1`/`levels.phase_2` lists to
`configs/all32.yaml`; keep the legacy sets in `configs/default.yaml` but add an
`all` key sourced as a literal 32-entry YAML list so configuration is
self-contained.

- [ ] **Step 4: Run manifest tests**

Run: `.venv/bin/pytest tests/test_levels.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/marioai/levels.py tests/test_levels.py configs/default.yaml configs/all32.yaml
git commit -m "feat: define canonical all-32 stage manifest"
```

### Task 2: Explicit 12-action configuration

**Files:**
- Create: `src/marioai/actions.py`
- Modify: `src/marioai/envs.py:1-69`
- Modify: `tests/test_levels.py`
- Modify: `tests/test_snapshot_wrapper.py:15-31`

**Interfaces:**
- Consumes: `validate_levels(levels) -> tuple[str, ...]`
- Produces: `resolve_action_set(name: str) -> list[list[str]]`
- Produces: `action_set_size(name: str) -> int`
- Produces: `make_mario_env(..., action_set: str = "simple")`
- Produces: `make_vec_env(..., action_set: str = "simple", level_weights: Mapping[str, float] | None = None)`

- [ ] **Step 1: Write failing action tests**

```python
from marioai.actions import action_set_size, resolve_action_set


def test_complex_action_set_has_down_and_12_actions():
    actions = resolve_action_set("complex")
    assert len(actions) == 12
    assert ["down"] in actions
    assert action_set_size("complex") == 12


def test_unknown_action_set_fails_loudly():
    with pytest.raises(ValueError, match="unknown action set"):
        resolve_action_set("wide")
```

Add an emulator smoke test that constructs
`make_mario_env("1-1", action_set="complex")`, asserts
`env.action_space.n == 12`, and closes it.

- [ ] **Step 2: Verify action tests fail**

Run: `.venv/bin/pytest tests/test_levels.py -v`

Expected: FAIL because `marioai.actions` does not exist.

- [ ] **Step 3: Implement action resolution and environment plumbing**

```python
# src/marioai/actions.py
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, SIMPLE_MOVEMENT

_ACTION_SETS = {
    "simple": SIMPLE_MOVEMENT,
    "complex": COMPLEX_MOVEMENT,
}


def resolve_action_set(name: str) -> list[list[str]]:
    try:
        return _ACTION_SETS[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown action set {name!r}; expected one of {sorted(_ACTION_SETS)}"
        ) from exc


def action_set_size(name: str) -> int:
    return len(resolve_action_set(name))
```

Replace the direct `SIMPLE_MOVEMENT` import in `envs.py` with
`resolve_action_set`, pass the selected actions to `JoypadSpace`, and thread
`action_set` through both factory functions. Keep `"simple"` as the default so
existing checkpoint tests remain valid; `configs/all32.yaml` will select
`"complex"`.

- [ ] **Step 4: Update route-based tests and run the environment suite**

Update `_raw_env` in `tests/test_snapshot_wrapper.py` to call
`resolve_action_set("simple")` instead of importing the constant directly.

Run: `.venv/bin/pytest tests/test_levels.py tests/test_snapshot_wrapper.py tests/test_wrappers.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/marioai/actions.py src/marioai/envs.py tests/test_levels.py tests/test_snapshot_wrapper.py
git commit -m "feat: configure Mario action sets explicitly"
```

### Task 3: Balanced and regression-weighted worker assignment

**Files:**
- Create: `src/marioai/sampling.py`
- Create: `tests/test_sampling.py`
- Modify: `src/marioai/envs.py:49-69`

**Interfaces:**
- Consumes: validated stage tuples.
- Produces: `assign_worker_levels(levels: Sequence[str], n_envs: int, weights: Mapping[str, float] | None = None) -> tuple[str, ...]`
- Produces: `regression_weights(levels: Sequence[str], previous: Mapping[str, bool], current: Mapping[str, bool], multiplier: float = 2.0) -> dict[str, float]`
- Produces: `make_vec_env(..., level_weights: Mapping[str, float] | None = None)`

- [ ] **Step 1: Write failing assignment tests**

```python
from collections import Counter
import pytest
from marioai.sampling import assign_worker_levels


def test_balanced_assignment_covers_32_stages_twice_with_64_workers():
    levels = tuple(f"{w}-{s}" for w in range(1, 9) for s in range(1, 5))
    assigned = assign_worker_levels(levels, 64)
    assert Counter(assigned) == {level: 2 for level in levels}


def test_regression_weight_gets_extra_fixed_workers():
    assigned = assign_worker_levels(
        ("1-1", "1-2", "1-3"), 8, {"1-1": 2.0, "1-2": 1.0, "1-3": 1.0}
    )
    assert Counter(assigned) == {"1-1": 4, "1-2": 2, "1-3": 2}


def test_weights_reject_unknown_or_nonpositive_values():
    with pytest.raises(ValueError):
        assign_worker_levels(("1-1",), 1, {"1-2": 1.0})
    with pytest.raises(ValueError):
        assign_worker_levels(("1-1",), 1, {"1-1": 0.0})


def test_regressed_stage_weight_doubles_for_next_worker_assignment():
    weights = regression_weights(
        ("1-1", "1-2"),
        previous={"1-1": True, "1-2": False},
        current={"1-1": False, "1-2": False},
    )
    assert weights == {"1-1": 2.0, "1-2": 1.0}
```

- [ ] **Step 2: Verify assignment tests fail**

Run: `.venv/bin/pytest tests/test_sampling.py -v`

Expected: FAIL because `marioai.sampling` does not exist.

- [ ] **Step 3: Implement deterministic largest-remainder assignment**

Implement validation, normalize weights, allocate each level
`floor(n_envs * weight / total_weight)` workers, and distribute remaining
workers by descending fractional remainder with manifest order as the stable
tie-breaker. Reject `n_envs < len(levels)` because every active stage must have
at least one fixed worker.

```python
def assign_worker_levels(levels, n_envs, weights=None):
    levels = tuple(levels)
    if n_envs < len(levels):
        raise ValueError("n_envs must be at least the number of active levels")
    resolved = {level: float((weights or {}).get(level, 1.0)) for level in levels}
    # Reserve one worker per level, then use largest remainders for the rest.
    counts = {level: 1 for level in levels}
    remaining = n_envs - len(levels)
    quotas = {
        level: remaining * resolved[level] / sum(resolved.values())
        for level in levels
    }
    for level in levels:
        counts[level] += int(quotas[level])
    leftovers = n_envs - sum(counts.values())
    ranked = sorted(levels, key=lambda level: (-(quotas[level] % 1), levels.index(level)))
    for level in ranked[:leftovers]:
        counts[level] += 1
    return tuple(level for level in levels for _ in range(counts[level]))
```

Thread the result into the existing fixed-worker `SubprocVecEnv` creation.
`regression_weights` starts every stage at `1.0` and applies the multiplier
only when a stage moved from passing in the previous diagnostic report to
failing in the current report.

- [ ] **Step 4: Run assignment and vector-environment tests**

Run: `.venv/bin/pytest tests/test_sampling.py tests/test_smoke.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/marioai/sampling.py src/marioai/envs.py tests/test_sampling.py
git commit -m "feat: balance fixed workers across Mario stages"
```

### Task 4: IMPALA residual feature extractor

**Files:**
- Create: `src/marioai/features.py`
- Create: `tests/test_features.py`

**Interfaces:**
- Produces: `ImpalaResidualBlock(nn.Module)`
- Produces: `ImpalaCnnFeaturesExtractor(BaseFeaturesExtractor)`
- Produces: constructor `ImpalaCnnFeaturesExtractor(observation_space, features_dim: int = 512, channels: tuple[int, ...] = (16, 32, 32))`

- [ ] **Step 1: Write failing shape and PPO smoke tests**

```python
import gymnasium as gym
import torch
from stable_baselines3 import PPO
from marioai.features import ImpalaCnnFeaturesExtractor


def test_impala_extractor_maps_stacked_frames_to_feature_width():
    space = gym.spaces.Box(0, 255, shape=(84, 84, 4), dtype="uint8")
    extractor = ImpalaCnnFeaturesExtractor(space, features_dim=256)
    output = extractor(torch.zeros(2, 84, 84, 4))
    assert output.shape == (2, 256)
    assert torch.isfinite(output).all()


def test_impala_policy_trains_one_small_update():
    venv = make_vec_env(["1-1"], n_envs=1, action_set="complex")
    try:
        model = PPO(
            "CnnPolicy",
            venv,
            n_steps=64,
            batch_size=64,
            policy_kwargs={
                "features_extractor_class": ImpalaCnnFeaturesExtractor,
                "features_extractor_kwargs": {"features_dim": 256},
                "normalize_images": True,
            },
            device="cpu",
        )
        model.learn(64)
    finally:
        venv.close()
```

- [ ] **Step 2: Verify feature tests fail**

Run: `.venv/bin/pytest tests/test_features.py -v`

Expected: FAIL because `marioai.features` does not exist.

- [ ] **Step 3: Implement the extractor**

Each IMPALA stage must apply `Conv2d`, `MaxPool2d`, then two residual blocks.
Convert channel-last observations to channel-first in `forward` because the
existing `VecFrameStack` emits `(N, 84, 84, 4)`.

```python
class ImpalaResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, inputs):
        return inputs + self.net(inputs)


def forward(self, observations):
    x = observations.float()
    if x.shape[1] != self._input_channels:
        x = x.permute(0, 3, 1, 2)
    x = self.cnn(x)
    return self.projection(torch.flatten(x, start_dim=1))
```

Determine the flattened dimension with a no-grad sample from
`observation_space.sample()` and define the projection as
`ReLU -> Linear(flattened, features_dim) -> ReLU`.

- [ ] **Step 4: Run feature and legacy smoke tests**

Run: `.venv/bin/pytest tests/test_features.py tests/test_smoke.py -v`

Expected: PASS with finite PPO loss.

- [ ] **Step 5: Commit**

```bash
git add src/marioai/features.py tests/test_features.py
git commit -m "feat: add IMPALA residual visual encoder"
```

### Task 5: All-32 training configuration and safe resume

**Files:**
- Modify: `src/marioai/train.py:1-124`
- Modify: `configs/all32.yaml`
- Create: `tests/test_train.py`
- Modify: `tests/test_smoke.py`

**Interfaces:**
- Consumes: `ImpalaCnnFeaturesExtractor`, `make_vec_env(..., action_set, level_weights)`.
- Produces: `load_training_config(path: str, phase: str, overrides: argparse.Namespace) -> dict`
- Produces: `build_policy_kwargs(cfg: Mapping) -> dict`
- Produces: `validate_resume_model(model: PPO, action_count: int, extractor_name: str) -> None`
- Produces CLI flags: `--phase {phase_1,phase_2}`, `--resume`, `--reset-timesteps`, and `--level-weights-json`.

- [ ] **Step 1: Write failing configuration tests**

```python
def test_all32_phase_two_contains_every_stage_and_complex_actions():
    cfg = load_training_config("configs/all32.yaml", "phase_2", _no_overrides())
    assert len(cfg["levels"]) == 32
    assert cfg["env"]["action_set"] == "complex"
    assert cfg["train"]["n_envs"] == 64
    assert cfg["policy"]["extractor"] == "impala"


def test_resume_rejects_seven_action_checkpoint():
    model = SimpleNamespace(
        action_space=SimpleNamespace(n=7),
        policy=SimpleNamespace(features_extractor=object()),
    )
    with pytest.raises(ValueError, match="checkpoint action count 7"):
        validate_resume_model(model, action_count=12, extractor_name="impala")
```

- [ ] **Step 2: Verify training tests fail**

Run: `.venv/bin/pytest tests/test_train.py -v`

Expected: FAIL because the configuration helpers do not exist.

- [ ] **Step 3: Refactor training into testable helpers**

Move parsing, vector-environment construction, and PPO creation out of `main`.
`build_policy_kwargs` must return:

```python
{
    "features_extractor_class": ImpalaCnnFeaturesExtractor,
    "features_extractor_kwargs": {
        "features_dim": cfg["policy"]["features_dim"],
        "channels": tuple(cfg["policy"]["channels"]),
    },
    "normalize_images": True,
}
```

Use `PPO.load(..., env=venv)` only after comparing the checkpoint action count
and extractor class. `--resume` continues `num_timesteps`; the legacy
`--init-from` path remains for compatible fine-tuning and defaults to resetting
timesteps. Save a `run-config.yaml` beside every checkpoint.

- [ ] **Step 4: Complete `configs/all32.yaml` and run focused tests**

Set:

```yaml
env:
  action_set: complex
  skip: 4
  frame_stack: 4
  shape: 84
train:
  n_envs: 64
  device: auto
  seed: 42
  checkpoint_freq: 250000
  normalize_reward: true
policy:
  extractor: impala
  features_dim: 512
  channels: [16, 32, 32]
evaluation:
  episodes: 15
  diagnostic_episodes: 3
  deterministic: false
  seed: 42000
phases:
  phase_1:
    total_timesteps: 32000000
    chunk_timesteps: 2000000
    level_weights: {}
  phase_2:
    total_timesteps: 64000000
    chunk_timesteps: 2000000
    level_weights: {}
```

Retain the current PPO hyperparameters for the first benchmark; architecture or
throughput tuning belongs to measured cloud execution, not speculative config
changes.

Run: `.venv/bin/pytest tests/test_train.py tests/test_features.py tests/test_smoke.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/marioai/train.py configs/all32.yaml tests/test_train.py tests/test_smoke.py
git commit -m "feat: configure resumable all-32 IMPALA training"
```

### Task 6: Immutable stochastic evaluation evidence

**Files:**
- Create: `src/marioai/results.py`
- Modify: `src/marioai/evaluate.py:1-55`
- Create: `tests/test_evaluate.py`

**Interfaces:**
- Produces: `sha256_file(path: Path) -> str`
- Produces: `RolloutResult` and `EvaluationReport` dataclasses with JSON conversion.
- Produces: `EvaluationReport.read(path: Path) -> EvaluationReport` and `EvaluationReport.write(path: Path) -> None`.
- Produces: `checkpoint_score(report: EvaluationReport) -> tuple[int, int, float]`.
- Produces: `is_better_checkpoint(candidate: EvaluationReport, incumbent: EvaluationReport | None) -> bool`.
- Produces: `evaluate_rollout(model, level: str, seed: int, deterministic: bool, ...) -> RolloutResult`
- Produces: `evaluate_checkpoint(model_path: Path, levels: Sequence[str], episodes: int = 15, seed: int = 42000, deterministic: bool = False) -> EvaluationReport`
- Produces CLI: `python -m marioai.evaluate --model ... --levels all --episodes 15 --stochastic --seed 42000 --out reports/evaluation.json`

- [ ] **Step 1: Write failing report tests**

```python
def test_stage_passes_with_one_of_15_clears():
    rollouts = [
        RolloutResult("1-1", i, i == 7, "flag" if i == 7 else "death", 100, 1.0, 10, 0.1)
        for i in range(15)
    ]
    stage = summarize_stage("1-1", rollouts)
    assert stage["passed"] is True
    assert stage["clears"] == 1
    assert stage["episodes"] == 15


def test_project_pass_requires_same_checkpoint_and_all_32_stages():
    report = EvaluationReport(
        checkpoint_sha256="a" * 64,
        deterministic=False,
        requested_episodes=15,
        stages={level: {"passed": True, "clears": 1, "episodes": 15}
                for level in ALL_LEVELS},
        rollouts=[],
    )
    assert report.passed is True


def test_acceptance_rejects_deterministic_or_non_15_report():
    assert not replace(report, deterministic=True).passed
    assert not replace(report, requested_episodes=14).passed


def test_checkpoint_selection_prefers_coverage_then_clears_then_progress():
    assert is_better_checkpoint(report_with_8_stages, report_with_7_stages)
    assert is_better_checkpoint(report_with_8_stages_12_clears, report_with_8_stages_9_clears)
    assert is_better_checkpoint(report_with_more_progress, report_with_less_progress)
```

- [ ] **Step 2: Verify evaluation tests fail**

Run: `.venv/bin/pytest tests/test_evaluate.py -v`

Expected: FAIL because `marioai.results` does not exist.

- [ ] **Step 3: Implement report types and rollout scoring**

Use these stable fields in each rollout:

```python
@dataclass(frozen=True)
class RolloutResult:
    level: str
    seed: int
    cleared: bool
    terminal_cause: str
    max_x: int
    reward: float
    steps: int
    wall_seconds: float
```

Seed the vector environment reset and policy sampling for each rollout. Derive
terminal cause in priority order: `flag_get`, timeout, `time == 0`, then
`death`. Save JSON atomically via a temporary file and `Path.replace`.

- [ ] **Step 4: Implement CLI and run evaluation tests**

Resolve `--levels all` to `ALL_LEVELS`. Default to stochastic predictions;
require an explicit `--deterministic` diagnostic flag. Print a 32-row summary
and a final `PASS` only when `EvaluationReport.passed` is true.

Run: `.venv/bin/pytest tests/test_evaluate.py tests/test_levels.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/marioai/results.py src/marioai/evaluate.py tests/test_evaluate.py
git commit -m "feat: emit all-32 stochastic acceptance evidence"
```

### Task 7: Shared-policy route recovery

**Files:**
- Create: `src/marioai/demonstrations.py`
- Modify: `scripts/behavior_clone_route.py:1-257`
- Modify: `scripts/solve_level.py:1-401`
- Modify: `tests/test_behavior_clone_route.py`
- Create: `tests/test_demonstrations.py`
- Modify: `tests/test_solve_level.py`

**Interfaces:**
- Consumes: `resolve_action_set`, all-32 shared PPO checkpoint.
- Produces: route metadata keys `action_set: str` and `decision_skip: int`.
- Produces: `load_demonstrations(route_dirs: Sequence[Path], action_set: str, skip: int) -> DemonstrationBatch`
- Produces: `level_mode(level: str) -> Literal["ground", "water", "maze"]`
- Produces: `swim_candidates() -> tuple[Macro, ...]`
- Produces: `maze_progress(previous_x: int, current_x: int, branch: int) -> MazeProgress`

- [ ] **Step 1: Write failing route-compatibility and mode tests**

```python
def test_route_rejects_mismatched_action_set_or_skip(tmp_path):
    save_route({
        "level": "2-2",
        "action_set": "simple",
        "decision_skip": 4,
        "actions": [1, 1, 1, 1],
        "waypoints": [],
    }, tmp_path)
    with pytest.raises(ValueError, match="action set"):
        load_demonstrations([tmp_path], action_set="complex", skip=4)


@pytest.mark.parametrize(
    ("level", "mode"),
    [("2-2", "water"), ("7-2", "water"), ("4-4", "maze"),
     ("7-4", "maze"), ("8-4", "ground")],
)
def test_level_mode(level, mode):
    assert level_mode(level) == mode


def test_swim_candidates_are_aligned_to_four_frame_decisions():
    assert all(macro.frames % 4 == 0 for macro in swim_candidates())


def test_maze_backward_snap_marks_wrong_branch():
    progress = maze_progress(previous_x=1400, current_x=900, branch=2)
    assert progress.wrong_branch is True
    assert progress.next_branch == 3
```

- [ ] **Step 2: Verify recovery tests fail**

Run: `.venv/bin/pytest tests/test_demonstrations.py tests/test_behavior_clone_route.py tests/test_solve_level.py -v`

Expected: FAIL because route metadata and special modes are absent.

- [ ] **Step 3: Implement shared demonstration loading**

Move `_decision_actions` and route observation collection into
`marioai.demonstrations`. Validate every route's level, action set, skip, action
range, and completed terminal state. Concatenate routes with equal per-level
sample weights, not raw-frame weights, so long stages do not dominate cloning.

Change the cloning CLI to accept repeated `--route-dir` arguments plus
`--action-set complex`; load one shared `--init-from` checkpoint and save one
shared `--out` checkpoint.

- [ ] **Step 4: Add solver modes without changing policy cadence**

Use `resolve_action_set(args.action_set)` in solver environment creation and
persist `action_set`/`decision_skip` in every partial and final route.

For water stages, candidates must be sequences of 4-aligned right/swim and
neutral/swim pulses, and the judge must use terminal state plus horizontal
progress rather than `grounded()`. For maze stages, treat an x-position drop of
at least 256 pixels without death as a wrong-route reset, advance the branch
candidate, restore the pre-branch snapshot, and exclude the rejected branch
from the next search expansion.

- [ ] **Step 5: Run recovery tests and regression suite**

Run: `.venv/bin/pytest tests/test_demonstrations.py tests/test_behavior_clone_route.py tests/test_solve_level.py tests/test_snapshot_wrapper.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/marioai/demonstrations.py scripts/behavior_clone_route.py scripts/solve_level.py tests/test_demonstrations.py tests/test_behavior_clone_route.py tests/test_solve_level.py
git commit -m "feat: recover shared policy with aligned demonstrations"
```

### Task 8: Local end-to-end no-spend gate

**Files:**
- Modify: `tests/test_smoke.py`
- Modify: `README.md:96-123`

**Interfaces:**
- Consumes: all preceding policy-pipeline interfaces.
- Produces: a locally saved and resumed complex-action IMPALA smoke checkpoint.
- Produces: a two-stage stochastic evaluation JSON fixture.

- [ ] **Step 1: Add the failing end-to-end smoke**

Create a temporary two-level config using 2 workers, 64 PPO steps, complex
actions, and the IMPALA extractor. Train and save, reload with
`validate_resume_model`, train 64 more steps, then evaluate one stochastic
rollout on each level and assert the JSON has the saved checkpoint SHA.

```python
assert resumed.num_timesteps >= 128
assert report.checkpoint_sha256 == sha256_file(model_path)
assert set(report.stages) == {"1-1", "1-2"}
assert all(result.seed >= 42000 for result in report.rollouts)
```

- [ ] **Step 2: Verify the end-to-end test fails**

Run: `.venv/bin/pytest tests/test_smoke.py::test_all32_pipeline_trains_resumes_and_evaluates -v`

Expected: FAIL until the smoke helper uses every new interface correctly.

- [ ] **Step 3: Complete the smoke helper and update local usage**

Add README commands:

```bash
python -m marioai.train --config configs/all32.yaml --phase phase_1 --run-name all32-phase1
python -m marioai.train --config configs/all32.yaml --phase phase_2 \
  --resume models/all32-phase1/final.zip --run-name all32-phase2
python -m marioai.evaluate --model models/all32-phase2/final.zip \
  --levels all --episodes 15 --stochastic --seed 42000 \
  --out reports/all32-phase2.json
```

Do not change the README results claim in this task.

- [ ] **Step 4: Run the complete local suite**

Run: `.venv/bin/pytest -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/test_smoke.py README.md
git commit -m "test: gate the all-32 policy pipeline locally"
```
