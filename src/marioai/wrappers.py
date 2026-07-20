"""Frame preprocessing wrappers for the Mario env (Gymnasium API)."""
import cv2
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SkipFrame(gym.Wrapper):
    """Repeat one action for `skip` frames, summing reward.

    Fewer decisions per second speeds up learning and lets the stacked frames
    span more real time (so the CNN can perceive velocity).

    When `capture_frames=True`, every intra-skip rendered frame is stored in
    `last_frames` after each step. GIF recording uses this to keep all game
    frames (not just 1-of-`skip`) so playback is smooth, not choppy. It is off
    during training (rendering every frame would waste time).
    """

    def __init__(self, env, skip=4, capture_frames=False):
        super().__init__(env)
        self._skip = skip
        self._capture = capture_frames
        self.last_frames = []

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        obs, info = None, {}
        self.last_frames = []
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if self._capture:
                # copy(): nes-py render() returns a view into one reused screen
                # buffer, so without a copy every stored frame would alias the last.
                frame = self.env.render()
                if frame is not None:
                    self.last_frames.append(frame.copy())
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
