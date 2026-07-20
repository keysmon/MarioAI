"""Phase 0 spike: prove the whole loop runs before building anything.

Success = this script prints "SPIKE PASSED" with a finite loss.
If it fails on the native stack, switch to requirements-legacy.txt on Python 3.10
and adjust imports (gymnasium -> gym) before continuing.
"""
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

import cv2
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total = 0.0
        terminated = truncated = False
        obs, info = None, {}
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total += reward
            if terminated or truncated:
                break
        return obs, total, terminated, truncated, info


class GrayScaleResize(gym.ObservationWrapper):
    def __init__(self, env, shape=84):
        super().__init__(env)
        self.shape = (shape, shape)
        self.observation_space = spaces.Box(0, 255, (shape, shape, 1), np.uint8)

    def observation(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, self.shape, interpolation=cv2.INTER_AREA)
        return resized[:, :, None].astype(np.uint8)


def make_env():
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", render_mode="rgb_array")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = SkipFrame(env, skip=4)
    env = GrayScaleResize(env, shape=84)
    return env


def main():
    venv = DummyVecEnv([make_env])
    venv = VecFrameStack(venv, n_stack=4, channels_order="last")
    model = PPO("CnnPolicy", venv, n_steps=128, batch_size=64, device="cpu", verbose=1)
    model.learn(total_timesteps=2000)
    loss = model.logger.name_to_value.get("train/loss", 0.0)
    assert np.isfinite(loss), f"loss not finite: {loss}"
    print(f"SPIKE PASSED (train/loss={loss})")


if __name__ == "__main__":
    main()
