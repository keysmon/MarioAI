"""Frame preprocessing wrappers for the Mario env (Gymnasium API)."""
import cv2
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SkipFrame(gym.Wrapper):
    """Repeat one action for `skip` frames, summing reward.

    Fewer decisions per second speeds up learning and lets the stacked frames
    span more real time (so the CNN can perceive velocity).
    """

    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        obs, info = None, {}
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class GrayScaleResize(gym.ObservationWrapper):
    """RGB frame -> single-channel 84x84 uint8 (channels-last)."""

    def __init__(self, env, shape=84):
        super().__init__(env)
        self.shape = (shape, shape)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(shape, shape, 1), dtype=np.uint8
        )

    def observation(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, self.shape, interpolation=cv2.INTER_AREA)
        return resized[:, :, None].astype(np.uint8)
