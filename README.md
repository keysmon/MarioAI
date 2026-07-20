# MarioAI 🍄 - PPO Agent That Plays Super Mario Bros From Raw Pixels

A deep reinforcement learning agent that learns to play **Super Mario Bros** directly from raw game frames - no access to the game's internal state, just pixels in and button presses out, exactly like a human looking at the screen.

Trained with **PPO** (Proximal Policy Optimization) on the `gym-super-mario-bros` NES environment. A single **multi-task** model is trained across several levels, then tested **zero-shot** on levels it has never seen.

<p align="center">
  <img src="assets/gifs/1-1.gif" width="480" alt="PPO agent clearing World 1-1"><br>
  <em>The trained agent clearing World 1-1 - learned entirely from raw pixels (100% clear rate over 20 greedy episodes).</em>
</p>

## What makes this interesting

- **Learns from pixels alone.** The agent never sees Mario's x-position or enemy locations as numbers - it perceives them from an 84x84 grayscale image, like the classic DeepMind Atari work. A convolutional network does the seeing.
- **One model, many levels.** Rather than a separate agent per level, a single network is trained multi-task across a set of levels (a random level per parallel worker), so it learns general Mario skills.
- **An honest generalization test.** A few levels are held out of training entirely and used to measure **zero-shot** transfer - the RL equivalent of a train/test split.

## How it works

```
raw NES frame (240x256x3)
  -> grayscale + resize to 84x84
  -> frame-skip 4 (repeat action, sum reward)
  -> stack 4 frames  (so the CNN can perceive velocity from a still image)
  -> PPO CnnPolicy (NatureCNN)  ->  one of 7 SIMPLE_MOVEMENT actions
```

Eight Mario emulators run in parallel (`SubprocVecEnv`); PPO pools their experience into one shared policy. Reward is the environment's shaped signal (rightward progress, minus a time penalty, minus a death penalty).

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

If you cannot use Python 3.13, an older battle-tested stack is pinned in `requirements-legacy.txt`. It needs a one-time pre-step (the `gym==0.21` setuptools quirk):

```bash
pip install setuptools==65.5.0 wheel==0.38.4
pip install gym==0.21.0
pip install -r requirements-legacy.txt
```
Then switch `import gymnasium as gym` -> `import gym` and the 5-tuple `step`/`reset` to old-gym 4-tuples.
</details>

## Usage

```bash
# Train the multi-task model on the default level set:
python -m marioai.train --config configs/default.yaml --run-name mario_multitask

# Or train a single level:
python -m marioai.train --config configs/default.yaml --levels 1-1 --timesteps 1000000 --run-name mario_1_1

# Evaluate clear-rate + mean reward:
python -m marioai.evaluate --model models/mario_multitask/final.zip --levels 1-1 1-2 1-3 2-1 3-1 4-1

# Record a smooth GIF (records N rollouts, keeps the cleanest):
python -m marioai.record_gif --model models/mario_multitask/final.zip --level 1-1 --out assets/gifs/1-1.gif

# Regenerate every training-level GIF:
scripts/record_all_gifs.sh models/mario_multitask/final.zip
```

Training logs to TensorBoard (`tensorboard --logdir runs`). Config (levels, hyperparameters, timesteps) lives in `configs/default.yaml`.

## Results

Greedy evaluation (deterministic policy, 20 episodes per level):

| Level | Clear rate | Mean reward |
|-------|-----------:|------------:|
| 1-1   |     100%   |      3106   |

> The single **multi-task** model (trained across `1-1, 1-2, 1-3, 2-1, 3-1, 4-1`) and its **zero-shot** results on the held-out levels `1-4` and `5-1` are training now and will be added here with a GIF per level.

<!-- GIF_GALLERY -->

## Project layout

```
src/marioai/
  wrappers.py     frame-skip + grayscale/resize (with smooth-capture mode for GIFs)
  envs.py         env factory + multi-task SubprocVecEnv assembly
  train.py        config-driven PPO training
  evaluate.py     per-level clear-rate + mean reward
  record_gif.py   N-rollouts-keep-cleanest native-RGB GIF recorder
configs/          hyperparameters + level sets
scripts/          spike, train, record, AWS runbook
tests/            wrapper unit tests + PPO smoke test
```

## Notes on training

The pipeline runs on CPU or GPU (`device: auto` in the config). Emulators are CPU-bound, so throughput scales with CPU cores; a GPU mainly speeds the policy update. The included `scripts/aws_provision.md` documents an on-demand AWS GPU workflow (quota preflight -> train -> download -> terminate) for scaling to more levels or longer runs.

## License

MIT - see [LICENSE](LICENSE).
