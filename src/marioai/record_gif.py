"""Record N rollouts of a policy on one level; keep the cleanest as a GIF."""
import argparse
import imageio
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from marioai.envs import make_mario_env


def _rollout(model, level, max_steps, skip, shape, frame_stack):
    """One rollout. Returns (frames, cleared, max_x, steps)."""
    venv = VecFrameStack(
        DummyVecEnv([lambda: make_mario_env(level=level, skip=skip, shape=shape)]),
        n_stack=frame_stack, channels_order="last",
    )
    obs = venv.reset()
    frames, cleared, max_x = [], False, 0
    step = 0
    for step in range(max_steps):
        frames.append(venv.render())   # native 240x256 RGB (render_mode set at construction)
        # deterministic=False: the NES emulator is deterministic, so greedy rollouts
        # would be identical every time. Sampling gives the variation that makes
        # "keep the cleanest of N" meaningful.
        action, _ = model.predict(obs, deterministic=False)
        obs, _, dones, infos = venv.step(action)
        cleared = cleared or bool(infos[0].get("flag_get", False))
        max_x = max(max_x, int(infos[0].get("x_pos", 0)))
        if bool(dones[0]):
            break
    venv.close()
    return frames, cleared, max_x, step + 1


def record(model, level, out, rollouts=5, fps=30, max_steps=3000,
           skip=4, shape=84, frame_stack=4):
    runs = [_rollout(model, level, max_steps, skip, shape, frame_stack)
            for _ in range(rollouts)]
    clears = [r for r in runs if r[1]]
    if clears:
        best = min(clears, key=lambda r: r[3])      # cleanest clear = fewest steps
        outcome = f"clear ({best[3]} steps)"
    else:
        best = max(runs, key=lambda r: r[2])        # best partial = furthest x
        outcome = f"partial (x_pos={best[2]})"
    imageio.mimsave(out, best[0], fps=fps)
    print(f"{level}: {outcome} -> {out}")
    return outcome


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--level", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--rollouts", type=int, default=5)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--max-steps", type=int, default=3000)
    args = p.parse_args()
    model = PPO.load(args.model, device="cpu")
    record(model, args.level, args.out, rollouts=args.rollouts,
           fps=args.fps, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
