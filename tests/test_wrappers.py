import numpy as np
import gymnasium as gym
from gymnasium import spaces
from marioai.wrappers import SkipFrame, GrayScaleResize


class _FakeEnv(gym.Env):
    """Emulator-free stand-in: emits frames, terminates at step `term_at`."""
    def __init__(self, term_at=10):
        self.observation_space = spaces.Box(0, 255, (240, 256, 3), np.uint8)
        self.action_space = spaces.Discrete(7)
        self.term_at = term_at
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        self.steps = 0
        return np.zeros((240, 256, 3), np.uint8), {}

    def step(self, action):
        self.steps += 1
        obs = np.full((240, 256, 3), self.steps, np.uint8)
        return obs, 1.0, self.steps >= self.term_at, False, {"steps": self.steps}


def test_skipframe_sums_reward_over_skip():
    env = SkipFrame(_FakeEnv(), skip=4)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(0)
    assert reward == 4.0
    assert info["steps"] == 4
    assert not terminated


def test_skipframe_breaks_on_early_termination():
    env = SkipFrame(_FakeEnv(term_at=2), skip=4)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(0)
    assert reward == 2.0
    assert terminated


def test_grayscale_resize_shape_and_dtype():
    env = GrayScaleResize(_FakeEnv(), shape=84)
    obs, _ = env.reset()
    assert obs.shape == (84, 84, 1)
    assert obs.dtype == np.uint8
    obs2, *_ = env.step(0)
    assert obs2.shape == (84, 84, 1)
