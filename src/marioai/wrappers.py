"""Frame preprocessing wrappers for the Mario env (Gymnasium API)."""
import random

import cv2
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .curriculum import CurriculumSchedule


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


class SnapshotStartWrapper(gym.Wrapper):
    """Start episodes from emulator snapshots along a solved route.

    Reverse-curriculum start states (Salimans & Chen 2018): reset() restores
    a snapshot sampled near the flag first, sliding earlier as the policy
    masters each segment (per-worker schedule, no IPC).

    nes-py snapshots are same-process-only, so the route ships as an action
    sequence (see scripts/solve_level.py); on first reset the wrapper replays
    it in its own emulator and captures in-process snapshots at the waypoint
    frames, cross-checking x_pos so a route/emulator mismatch fails loud.

    Wrap directly around JoypadSpace, under SkipFrame, so step() sees every
    native frame and reset() returns a raw RGB frame for GrayScaleResize.
    """

    def __init__(self, env, route, schedule=None, seed=0):
        super().__init__(env)
        if not route.get("waypoints"):
            raise ValueError("route has no waypoints")
        self._route = route
        self._snapshots = None  # replay-captured lazily on first reset
        self._schedule = schedule or CurriculumSchedule(
            len(route["waypoints"]), rng=random.Random(seed)
        )

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        if self._snapshots is None:
            self._capture_snapshots()
        raw = self.env.unwrapped
        start = self._schedule.sample_start()
        raw.load_state(self._snapshots[start])
        # one NOOP frame so the C++ screen buffer redraws the restored state
        raw._frame_advance(0)
        self._resync_reward_caches(raw)
        obs = raw.screen.copy()
        info = raw._get_info()
        info["curriculum_frontier"] = self._schedule.frontier
        info["curriculum_start"] = start
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            self._schedule.record(bool(info.get("flag_get", False)))
        info["curriculum_frontier"] = self._schedule.frontier
        return obs, reward, terminated, truncated, info

    def _capture_snapshots(self):
        """Replay the route once, snapshotting at each waypoint frame.

        Uses self.env.step (not self.step) so replay outcomes never touch
        the curriculum schedule.
        """
        raw = self.env.unwrapped
        marks = {wp["frame"]: wp for wp in self._route["waypoints"]}
        last_frame = max(marks)
        snaps = {}
        if 0 in marks:
            snaps[0] = raw.dump_state()
        for frame, action in enumerate(
                self._route["actions"][:last_frame], start=1):
            _, _, terminated, truncated, info = self.env.step(action)
            if (terminated or truncated) and frame < last_frame:
                raise RuntimeError(
                    f"route replay died at frame {frame}/{last_frame}; "
                    "route and emulator disagree - re-run "
                    "scripts/solve_level.py"
                )
            wp = marks.get(frame)
            if wp is not None:
                if abs(int(info["x_pos"]) - wp["x_pos"]) > 4:
                    raise RuntimeError(
                        f"route replay diverged at frame {frame}: "
                        f"x={info['x_pos']} expected {wp['x_pos']}"
                    )
                snaps[frame] = raw.dump_state()
        missing = sorted(f for f in marks if f not in snaps)
        if missing:
            raise RuntimeError(
                f"route waypoint frames beyond the action list: {missing}"
            )
        self._snapshots = [snaps[wp["frame"]]
                           for wp in self._route["waypoints"]]

    @staticmethod
    def _resync_reward_caches(raw):
        # Mirror SuperMarioBrosEnv._did_reset: the env's reward properties
        # diff against cached RAM values, and after load_state those caches
        # still describe the pre-restore state (first-step reward spike).
        raw._time_last = raw._time
        raw._x_position_max = raw._x_position
        raw._score_last = raw._score
        raw._coins_last = raw._coins
        raw._status_last = raw._powerup_level
        raw._completion_rewarded = False
        raw._reset_reward_components()
