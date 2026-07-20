"""Evaluate a trained model per level: clear-rate + mean reward.

Uses a plain (un-normalized) env: VecNormalize only normalized the *reward*
during training (norm_obs=False), so the policy's observations are identical
here and greedy action selection is unaffected.
"""
import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from marioai.envs import make_mario_env


def evaluate_level(model, level, episodes=20, frame_stack=4, skip=4, shape=84):
    venv = VecFrameStack(
        DummyVecEnv([lambda: make_mario_env(level=level, skip=skip, shape=shape)]),
        n_stack=frame_stack, channels_order="last",
    )
    clears, rewards = 0, []
    for _ in range(episodes):
        obs = venv.reset()
        done = False
        ep_reward = 0.0
        cleared = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = venv.step(action)
            ep_reward += float(reward[0])
            cleared = cleared or bool(infos[0].get("flag_get", False))
            done = bool(dones[0])
        clears += int(cleared)
        rewards.append(ep_reward)
    venv.close()
    return {"clear_rate": clears / episodes, "mean_reward": float(np.mean(rewards))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--levels", nargs="+", required=True)
    p.add_argument("--episodes", type=int, default=20)
    args = p.parse_args()

    model = PPO.load(args.model, device="cpu")
    print(f"{'level':<6} {'clear_rate':>10} {'mean_reward':>12}")
    results = {}
    for lvl in args.levels:
        r = evaluate_level(model, lvl, episodes=args.episodes)
        results[lvl] = r
        print(f"{lvl:<6} {r['clear_rate']:>10.2f} {r['mean_reward']:>12.1f}")
    return results


if __name__ == "__main__":
    main()
