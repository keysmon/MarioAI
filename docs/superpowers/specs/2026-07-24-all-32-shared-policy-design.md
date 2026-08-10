# All-32 Shared Policy and Media Design

Date: 2026-07-24

Status: Approved in conversation; awaiting written-spec review

## Summary

Extend MarioAI from its current eight-stage showcase to all 32 stages in the
original Super Mario Bros. campaign. The primary deliverable is one shared
policy checkpoint that can clear every stage. A stage counts as passing when at
least one of 15 stochastic evaluation rollouts reaches the flag or otherwise
triggers the environment's completion signal.

Training will use a staged, balanced curriculum and an IMPALA-style residual
visual encoder with the full `COMPLEX_MOVEMENT` action set. Targeted
demonstrations may be added to the shared policy when reinforcement learning
stalls, but per-stage specialist policies are outside the primary design and
require a separate user decision.

The total AWS spend for this phase has a hard cap of USD $50. No cloud resource
may continue running after the cap guard is reached.

Every documented clear will have a successful GIF that plays at a normalized
effective 2x speed and contains explicit infinite-loop metadata for reliable
replay on GitHub.

## Goals

- Support all 32 stages: worlds 1 through 8, stages 1 through 4.
- Produce one shared policy checkpoint that passes the stated evaluation bar
  on all 32 stages.
- Preserve reproducible evaluation evidence, including seeds, rollout results,
  checkpoint identity, and completion statistics.
- Keep total AWS compute spend at or below $50.
- Record one successful GIF for each passing stage.
- Normalize all repository GIFs to effective 2x playback and infinite looping.
- Replace the README's eight-stage framing with a truthful 32-stage matrix.

## Non-goals

- Requiring all 15 stochastic rollouts to clear.
- Optimizing speedruns, score, or completion time after a stage has passed.
- Training specialist models without an explicit post-failure decision.
- Adding recurrent policies preemptively.
- Replacing the emulator or reinforcement-learning framework.
- Claiming a stage passes based only on a solver route or deterministic replay.

## Current State

The repository currently demonstrates eight selected stages:

- 1-1, 1-2, 1-3, and 1-4
- 2-1
- 3-1
- 4-1
- 5-1

The environment currently uses `SIMPLE_MOVEMENT`, training uses Stable
Baselines3's default CNN policy, and configuration and reporting are centered on
those eight stages. The existing GIFs use roughly consistent frame durations,
but their routes have different lengths and they lack explicit infinite-loop
metadata.

Existing stage-specific artifacts remain useful as regression evidence and
possibly as demonstration sources. They do not satisfy the new requirement
because the final acceptance checkpoint must be one shared policy for all 32
stages.

## Alternatives Considered

### Continue with the default CNN

This minimizes engineering work but retains a network intended for simpler
Atari-like tasks. It is less likely to represent the varied geometry, swimming,
platform timing, and castle routing needed across 32 stages within the budget.

### Solver-first behavior cloning

This can make a deterministic route directly learnable, but producing robust
routes for every stage would require major solver additions for water physics,
maze resets, and moving-platform timing. It shifts too much of the project from
policy learning into scripted search.

### Staged shared policy with targeted demonstrations

This is the selected approach. Reinforcement learning handles common movement
and generalization. Demonstrations are introduced only for measured failure
modes and still train the same shared checkpoint.

## Policy and Environment Design

### Action space

All training, evaluation, solver output, and replay code will use
`COMPLEX_MOVEMENT`. It contains 12 discrete actions and includes the Down input
needed by later castle mechanics. Action-set selection will be configuration,
not an implicit hard-coded choice.

Changing the action count means the existing policy output head cannot be
loaded directly. Compatible visual-trunk weights may be warm-started only when
loading is explicit and shape-checked. Otherwise the new shared policy starts
cleanly.

### Observation encoder

Replace the default NatureCNN feature extractor with an IMPALA-style residual
CNN:

- three convolutional stages with progressive channel growth;
- residual blocks within each stage;
- spatial downsampling between stages;
- a compact feature projection consumed by the PPO actor and critic.

The exact channel counts and feature width will be selected by a local smoke
benchmark. The architecture must fit the chosen 64-environment worker layout
without swapping or starving emulator workers.

### Shared policy identity

The final artifact is a single policy checkpoint with one observation encoder,
one actor head, and one critic head. Level identity may be made available only
through an explicit, stable conditioning feature if experiments show that
balanced sampling alone is insufficient. Conditioning must not select separate
weights or separate actor heads.

### Level sampling

Training uses a level manifest containing all 32 stage identifiers and their
curriculum state. Sampling is balanced by stage rather than by episode count so
short or repeatedly failing stages cannot dominate updates.

During the second curriculum phase, earlier stages remain in the sample mix.
Recent evaluation regressions increase a stage's sampling weight until its pass
status recovers. This replay mix is the main defense against catastrophic
forgetting.

## Training Phases

### Phase 0: local pipeline validation

- Make action space, architecture, stage manifest, and evaluation settings
  configurable.
- Run short local smoke tests on representative ground, water, and castle
  stages.
- Verify checkpoint save/resume and a no-spend budget dry run.
- Benchmark environment throughput before selecting the AWS instance size.

No paid training starts until this phase passes.

### Phase 1: worlds 1-4

Train the shared policy across the first 16 stages. Evaluation runs periodically
on all 16 stages using fixed recorded seed sets plus stochastic actions. A
checkpoint advances only when aggregate coverage improves without a material
regression on previously passing stages.

### Phase 2: worlds 5-8

Add the remaining 16 stages while retaining worlds 1-4 in the replay mix.
Evaluation expands to all 32 stages. Sampling weights are adjusted from measured
per-stage results, not manually from apparent difficulty alone.

### Phase 3: shared-policy recovery

For remaining failures:

1. Categorize the failure from rollout traces and emulator replay.
2. Prefer reward, curriculum, or sampling corrections when the policy reaches
   the relevant obstacle but behaves inconsistently.
3. Generate policy-cadence-compatible solver demonstrations when exploration
   never discovers the required behavior.
4. Behavior-clone demonstrations into the same shared policy, then run a short
   balanced PPO polish to restore stochastic robustness.

Water stages may require swim-specific solver macros. Castle stages 4-4 and 7-4
may require route-state handling for backward position resets after a wrong
maze branch.

### Evidence-gated recurrence

An LSTM policy is considered only if rollout evidence from maze stages shows
that visually indistinguishable states require memory. Poor exploration,
incorrect rewards, or inadequate sampling are not sufficient reasons to add
recurrence.

### Specialist fallback gate

If the shared policy does not pass all stages before the recovery allocation is
exhausted:

- stop paid training;
- preserve the best shared checkpoint and the complete evaluation matrix;
- report failed stages, failure categories, spend, and estimated specialist
  scope;
- ask the user whether to authorize specialists.

No specialist training begins implicitly.

## Evaluation and Evidence

Each candidate checkpoint is evaluated with 15 stochastic rollouts per stage.
The evaluator records:

- checkpoint hash or immutable artifact identifier;
- stage and rollout seed;
- completion status;
- terminal cause;
- maximum horizontal progress;
- episode reward and length;
- wall-clock evaluation duration.

A stage passes with at least one completion in its 15-rollout batch. The shared
checkpoint passes the project only when all 32 stages pass in evaluation using
that exact checkpoint.

Greedy evaluation and solver replay are diagnostic tools. They do not replace
the stochastic acceptance batch.

The results writer produces a machine-readable report and the README table from
the same source data to prevent documentation drift.

## AWS Execution and Budget Safety

Mario emulator stepping is expected to be CPU-bound, so the initial target is a
spot compute-optimized instance rather than a large GPU instance. The local
benchmark determines whether `c7i.8xlarge`, `c7i.16xlarge`, or a smaller
equivalent provides the best cost per environment step.

Budget allocation:

| Purpose | Maximum |
| --- | ---: |
| Pipeline validation and cloud benchmark | $4 |
| Worlds 1-4 training | $16 |
| Worlds 5-8 training | $16 |
| Shared-policy recovery and cloning | $10 |
| Evaluation and interruption reserve | $4 |
| **Total** | **$50** |

The runner will:

- calculate accrued cost from instance price and runtime;
- persist its ledger alongside checkpoints;
- refuse a launch whose projected minimum run exceeds the remaining allocation;
- begin graceful checkpoint and artifact synchronization before the hard cap;
- terminate the instance at the configured safety threshold;
- tolerate spot interruption by resuming from the latest durable checkpoint.

Instance state and spend are checked independently after every cloud run. A
failed training process must not leave a paid instance running.

## GIF and README Design

The recorder captures native emulator frames from a successful rollout.
Effective 2x playback is created by selecting every second captured frame while
retaining a browser-safe frame duration. This avoids sub-20-millisecond GIF
delays that browsers may quantize or clamp unpredictably.

Encoding requirements:

- explicit `loop=0` infinite-loop metadata;
- a common output size and frame timing;
- preservation of the terminal clear frame;
- deterministic frame selection;
- post-encode validation of dimensions, duration distribution, and loop value.

The existing eight GIFs will be regenerated from successful rollouts when
possible or re-encoded with the same rules when their source rollouts are the
only available evidence.

The README will show:

- the shared checkpoint and evaluation date;
- a 32-row stage status table;
- clears out of 15 rollouts for each stage;
- the successful GIF for each passing stage;
- no claim of 32/32 until one exact checkpoint passes all acceptance batches.

## Testing and Verification

Local automated tests will cover:

- all 32 stage identifiers and manifest validation;
- `COMPLEX_MOVEMENT` propagation through training, evaluation, solver, and
  recorder paths;
- custom feature-extractor output shapes;
- balanced and regression-weighted level sampling;
- checkpoint resume with budget-ledger restoration;
- hard-cap launch refusal and graceful-stop thresholds;
- stage pass/fail aggregation from 15 rollouts;
- README generation from evaluation data;
- GIF frame selection, terminal-frame preservation, 2x effective duration, and
  infinite-loop metadata.

Before cloud training, a short end-to-end local run must train, save, resume,
evaluate, and record a GIF. Before completion is claimed, the full test suite,
the 32-stage acceptance evaluator, GIF metadata checks, README consistency
checks, AWS instance-state checks, and actual spend reconciliation must pass.

## Completion Criteria

This phase is complete when:

1. One shared checkpoint records at least one clear in 15 stochastic rollouts
   for each of all 32 stages.
2. The evaluation report identifies that exact checkpoint and contains all
   rollout evidence.
3. All 32 successful GIFs use normalized effective 2x playback and explicit
   infinite looping.
4. The README accurately reflects the generated results.
5. The automated test suite passes.
6. AWS resources are stopped or terminated and reconciled spend is no more than
   $50.
7. The completed changes are committed and pushed.
