# MarioAI 🍄 - PPO Agent That Plays Super Mario Bros From Raw Pixels

A deep reinforcement learning agent that plays **Super Mario Bros** from raw
game frames. At inference time every policy sees only pixels and emits button
presses, exactly like a human looking at the screen.

Built on **PPO** (Proximal Policy Optimization). Across the eight levels
tackled here, the agent **clears all 8** - including the underground 1-2,
the castle 1-4, and the gap-heavy 1-3.

<p align="center">
  <img src="assets/gifs/1-1.gif" width="460" alt="PPO agent clearing World 1-1">
  <img src="assets/gifs/1-4.gif" width="460" alt="PPO agent clearing the castle World 1-4"><br>
  <em>Learned entirely from 84x84 grayscale pixels: clearing World 1-1 (left) and the castle World 1-4 (right).</em>
</p>

## Results - 8 of 8 levels cleared

| Level | Cleared? | Clear rate | How |
|-------|:---:|:---:|---|
| 1-1 | ✅ | 100% | multi-task model |
| 1-2 (underground) | ✅ | 100% | multi-task model |
| 4-1 | ✅ | 100% | multi-task model |
| 3-1 | ✅ | 100% | fine-tuned |
| **1-4 (castle)** | ✅ | 100% | fine-tuned |
| 2-1 | ✅ | clears* | fine-tuned |
| 5-1 | ✅ | clears* | fine-tuned |
| **1-3 (pits)** | ✅ | 100% | behavior cloning from a machine-searched route |

\* 2-1 and 5-1: the *deterministic* policy narrowly misses the final obstacle, but the agent clears them when sampling actions - the GIF is a genuine, unedited clear.

World 1-3 cleared all **15/15 sampled final-probe rollouts**. Its gallery GIF
is a genuine, unedited **290-policy-step clear**.

## Gallery

**Cleared (8):**

| 1-1 | 1-2 (underground) | 2-1 | 3-1 |
|:---:|:---:|:---:|:---:|
| ![1-1](assets/gifs/1-1.gif) | ![1-2](assets/gifs/1-2.gif) | ![2-1](assets/gifs/2-1.gif) | ![3-1](assets/gifs/3-1.gif) |
| **4-1** | **5-1** | **1-4 (castle)** | **1-3 (pits)** |
| ![4-1](assets/gifs/4-1.gif) | ![5-1](assets/gifs/5-1.gif) | ![1-4](assets/gifs/1-4.gif) | ![1-3](assets/gifs/1-3.gif) |

## How it was built

Getting to 8/8 took an escalation ladder, not a single training run:

1. **One multi-task model.** A single PPO `CnnPolicy` trained across six levels at once (8M steps) learned to clear **1-1, 1-2, and 4-1** outright, and got most of the way through the rest.
2. **Per-level fine-tuning.** For levels the shared model stalled on, we **fine-tuned that model on the single level** with a low, constant learning rate (so its transferred Mario skills aren't destabilized) plus extra exploration. This cracked **3-1, 2-1, 5-1, and the castle 1-4**.
3. **Machine-generated demonstration for 1-3.** An emulator-snapshot solver searched directly in the policy's skip-4 action space, producing a route whose action changes are executable at the policy's four-frame decision cadence. Behavior cloning that route through the exact preprocessing and frame-stack pipeline produced both greedy and stochastic clears.
4. **Best-of-N recording.** For levels the greedy policy narrowly misses (2-1, 5-1), recording several stochastic rollouts and keeping the cleanest captures a real clear.

Every policy sees only the 84x84 grayscale image. The offline route search for
1-3 uses emulator snapshots and position state to generate its demonstration;
none of that state is available to the cloned policy at inference time.

## The last one to fall: World 1-3

**1-3 is a level of gaps and moving lifts**, and it remained the classic wall
for vanilla PPO: true-start training repeatedly converged on a profitable
sprint that died at the lift ferry, while snapshot curricula learned brittle
timings tied to one platform phase.

The successful route was generated without a human demonstration. A
snapshot-search solver tried jump/wait macros, rewound dead ends, and searched
at one action per four native frames - the same cadence the PPO policy uses.
That detail matters: the first native-frame route cleared in the emulator but
died when quantized to skip 4. The aligned route clears by construction, and
behavior cloning it with the exact `84x84 x 4` observation stack yielded a
greedy clear plus **15/15 sampled clears** in the final probe.

## How the legacy eight-level checkpoint works

This diagram describes the historical v0.3.0 models, not the all-32
acceptance pipeline. The shared all-32 policy replaces NatureCNN with the
IMPALA residual encoder and emits one of 12 `COMPLEX_MOVEMENT` actions.

```
raw NES frame (240x256x3)
  -> grayscale + resize to 84x84
  -> frame-skip 4 (repeat action, sum reward)
  -> stack 4 frames  (so the CNN can perceive velocity from a still image)
  -> PPO CnnPolicy (NatureCNN)  ->  one of 7 SIMPLE_MOVEMENT actions
```

## Setup

Requires **Python 3.13** (the modern `gym-super-mario-bros` / `nes-py` are Gymnasium-native and need 3.13+).

```bash
git clone https://github.com/keysmon/MarioAI.git
cd MarioAI
uv venv --python 3.13 .venv && source .venv/bin/activate   # or: python3.13 -m venv .venv
pip install -r requirements.txt
pip install -e .
```

<details>
<summary>Legacy fallback (Python 3.10)</summary>

An older battle-tested stack is pinned in `requirements-legacy.txt` (needs a one-time `pip install setuptools==65.5.0 wheel==0.38.4 && pip install gym==0.21.0` first, then `import gymnasium as gym` -> `import gym` and 5-tuple -> 4-tuple `step`/`reset`).
</details>

## Usage

### All-32 shared policy pipeline (acceptance-capable)

```bash
python -m marioai.train --config configs/all32.yaml --phase phase_1 --run-name all32-phase1
python -m marioai.train --config configs/all32.yaml --phase phase_2 \
  --resume models/all32-phase1/final.zip --run-name all32-phase2
python -m marioai.evaluate --model models/all32-phase2/final.zip \
  --levels all --episodes 15 --stochastic --seed 42000 \
  --out reports/all32-phase2.json

# Record a normalized 2x, infinite-loop GIF from the shared checkpoint:
python -m marioai.record_gif --model models/all32-phase2/final.zip \
  --level 2-1 --out assets/gifs/2-1.gif --rollouts 15
```

Only a `shared_complex_impala` report containing exactly 15 stochastic
rollouts for each of all 32 stages can print acceptance `PASS`.

### Legacy eight-level reproduction (diagnostic only)

These commands reproduce the historical seven-action NatureCNN workflow.
`--legacy-diagnostic` labels its JSON as
`legacy_simple_nature_diagnostic`, which can never satisfy all-32 acceptance.

```bash
# Train and fine-tune the historical model:
python -m marioai.train --config configs/default.yaml --run-name mario_multitask
python -m marioai.train --config configs/default.yaml \
  --init-from models/mario_multitask/final.zip --levels 2-1 \
  --lr 0.00005 --ent-coef 0.03 --timesteps 2000000 --run-name ft_2-1

# Emit explicitly non-acceptance legacy diagnostics:
python -m marioai.evaluate --model models/ft_2-1/final.zip --levels 2-1 \
  --episodes 15 --stochastic --legacy-diagnostic \
  --out reports/legacy-ft-2-1-diagnostic.json

# Reproduce the 1-3 machine demonstration and clone it into a PPO policy:
python scripts/solve_level.py --level 1-3 --skip 4 --action-set simple
python scripts/behavior_clone_route.py \
  --route-dir models/ft_1-3/waypoints \
  --action-set simple \
  --init-from models/mario_multitask/final.zip \
  --out models/ft_1-3/final.zip
```

The original PPO checkpoints are on the [v0.3.0 Release](https://github.com/keysmon/MarioAI/releases/tag/v0.3.0). Training logs to TensorBoard (`tensorboard --logdir runs`).

## Notes on compute

Most models were trained on an AWS `c7i.4xlarge` (16 vCPU, 32 GB). Mario RL is
**CPU-bound** - the bottleneck is stepping the NES emulators, not the small CNN.
The final 1-3 behavior-cloning pass used a GPU, where supervised CNN updates
benefit from acceleration.

## Project layout

```
src/marioai/
  wrappers.py     frame-skip + grayscale/resize (with smooth-capture mode for GIFs)
  envs.py         env factory + multi-task SubprocVecEnv assembly
  train.py        config-driven PPO (--levels, --n-envs, --timesteps, --init-from, --lr, --ent-coef)
  evaluate.py     per-level clear-rate + mean reward
  record_gif.py   N-rollouts-keep-cleanest native-RGB GIF recorder
  curriculum.py   reverse-curriculum schedule + route persistence
configs/          hyperparameters + level sets
scripts/          route solver, behavior cloning, train/record helpers, AWS runbook
tests/            wrapper unit tests + PPO smoke test
```

## License

MIT - see [LICENSE](LICENSE).
