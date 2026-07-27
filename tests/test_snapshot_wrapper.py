"""Emulator + fake-env tests for route replay and SnapshotStartWrapper.

These are the spec's pre-cloud smoke test: they prove a route (action
sequence + waypoint frames) round-trips through disk as JSON, rebuilds
snapshots by deterministic replay in a FRESH env instance (the subprocess
worker scenario), and that a restored start yields a valid observation and
a sane first-step reward (no stale-cache spike).
"""
import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace

from marioai.actions import resolve_action_set
from marioai.curriculum import CurriculumSchedule, load_route, save_route
from marioai.envs import make_mario_env, make_vec_env
from marioai.wrappers import SnapshotStartWrapper

RIGHT_B = 3
RUN_FRAMES = 80  # Mario dies to 1-1's first goomba near frame 105; stay clear


def _raw_env(level="1-1"):
    return JoypadSpace(
        gym_super_mario_bros.make(f"SuperMarioBros-{level}-v0",
                                  render_mode="rgb_array"),
        resolve_action_set("simple"),
    )


def _make_route(level="1-1", frames=RUN_FRAMES):
    """Probe-run `frames` of run-right and return a two-waypoint route."""
    env = _raw_env(level)
    _, info = env.reset(seed=0)
    start_x = int(info["x_pos"])
    for _ in range(frames):
        _, _, term, trunc, info = env.step(RIGHT_B)
        assert not (term or trunc), "died while building the probe route"
    end_x = int(info["x_pos"])
    env.close()
    return {
        "level": level,
        "action_set": "simple",
        "decision_skip": 4,
        "actions": [RIGHT_B] * frames,
        "waypoints": [
            {"index": 0, "frame": 0, "x_pos": start_x},
            {"index": 1, "frame": frames, "x_pos": end_x},
        ],
    }


def test_route_replay_restores_in_fresh_env(tmp_path):
    route = _make_route()
    save_route(route, tmp_path)

    # brand-new env instance + wrapper, as a vec worker would build it
    wrapped = SnapshotStartWrapper(_raw_env(), load_route(tmp_path), seed=0)
    obs, info = wrapped.reset(seed=0)
    # frontier starts at the last waypoint and the window extends toward the
    # end of the list, so the only sampleable start is waypoint 1
    assert info["curriculum_start"] == 1
    assert abs(info["x_pos"] - route["waypoints"][1]["x_pos"]) <= 4
    assert obs.shape == (240, 256, 3)
    assert obs.dtype == np.uint8
    # no stale-cache reward spike on the first step (progress is capped +5,
    # an unsynced clock would show up as a large negative time penalty)
    _, reward, _, _, _ = wrapped.step(0)
    assert -2.0 <= reward <= 6.0
    wrapped.close()


def test_corrupt_route_fails_loud(tmp_path):
    route = _make_route(frames=10)
    route["waypoints"][1]["frame"] = 999  # beyond the action list
    save_route(route, tmp_path)
    wrapped = SnapshotStartWrapper(_raw_env(), load_route(tmp_path), seed=0)
    with pytest.raises(RuntimeError):
        wrapped.reset(seed=0)
    wrapped.close()


def test_restore_earlier_waypoint_resyncs_caches(tmp_path):
    # three-waypoint route so a NON-final index is restored for real:
    # capture leaves the emulator at frame 80, then reset jumps BACK to
    # frame 40 - a genuine position-jump restore that only passes if the
    # reward caches were re-based on the restored state
    env = _raw_env()
    _, info = env.reset(seed=0)
    xs = {0: int(info["x_pos"])}
    actions = []
    for f in range(1, 81):
        _, _, term, trunc, info = env.step(RIGHT_B)
        assert not (term or trunc), "died while building the probe route"
        actions.append(RIGHT_B)
        if f in (40, 80):
            xs[f] = int(info["x_pos"])
    env.close()
    route = {
        "level": "1-1",
        "actions": actions,
        "waypoints": [
            {"index": 0, "frame": 0, "x_pos": xs[0]},
            {"index": 1, "frame": 40, "x_pos": xs[40]},
            {"index": 2, "frame": 80, "x_pos": xs[80]},
        ],
    }
    save_route(route, tmp_path)

    sched = CurriculumSchedule(3, window=1, history=1)
    sched.frontier = 1  # force restores of the MIDDLE waypoint only
    wrapped = SnapshotStartWrapper(_raw_env(), load_route(tmp_path),
                                   schedule=sched)
    _, info = wrapped.reset(seed=0)
    assert info["curriculum_start"] == 1
    assert abs(info["x_pos"] - xs[40]) <= 4
    raw = wrapped.env.unwrapped
    # resync must have re-based every reward cache on the RESTORED state
    assert raw._time_last == raw._time
    assert raw._x_position_max == raw._x_position
    assert raw._score_last == raw._score
    _, reward, _, _, _ = wrapped.step(0)
    assert -2.0 <= reward <= 6.0
    wrapped.close()


class _FakeSnapEnv(gym.Env):
    """Emulator-free stand-in exposing the attrs SnapshotStartWrapper touches.

    Every episode terminates on the first step with flag_get=True, so
    curriculum bookkeeping can be tested quickly without the emulator.
    """
    observation_space = spaces.Box(0, 255, (240, 256, 3), np.uint8)
    action_space = spaces.Discrete(7)

    def __init__(self):
        self.screen = np.zeros((240, 256, 3), np.uint8)
        self._time = self._time_last = 400
        self._x_position = self._x_position_max = 40
        self._score = self._score_last = 0
        self._coins = self._coins_last = 0
        self._powerup_level = self._status_last = 0
        self._completion_rewarded = False

    def reset(self, *, seed=None, options=None):
        return np.zeros((240, 256, 3), np.uint8), {}

    def step(self, action):
        obs = np.zeros((240, 256, 3), np.uint8)
        return obs, 0.0, True, False, {"flag_get": True}

    def dump_state(self):
        return object()

    def load_state(self, snapshot):
        pass

    def _frame_advance(self, action):
        pass

    def _get_info(self):
        return {"x_pos": self._x_position}

    def _reset_reward_components(self):
        pass


def test_wrapper_advances_frontier_on_clears():
    route = {
        "level": "test",
        "actions": [],
        "waypoints": [{"index": i, "frame": 0, "x_pos": 40}
                      for i in range(3)],
    }
    sched = CurriculumSchedule(3, history=2, advance_threshold=1.0)
    w = SnapshotStartWrapper(_FakeSnapEnv(), route, schedule=sched)
    for _ in range(4):
        w.reset()
        _, _, _, _, info = w.step(0)
        assert "curriculum_frontier" in info
    # 2 clears -> frontier 1, history reset; 2 more clears -> frontier 0
    assert sched.frontier == 0


def test_vec_env_with_snapshot_starts_smoke(tmp_path):
    """Full stack: SubprocVecEnv workers replay the route from disk."""
    save_route(_make_route(), tmp_path)
    venv = make_vec_env(["1-1"], n_envs=2, snapshot_dir=str(tmp_path))
    try:
        obs = venv.reset()
        assert obs.shape == (2, 84, 84, 4)
        for _ in range(20):
            obs, rewards, dones, infos = venv.step(
                np.array([RIGHT_B, RIGHT_B]))
        assert obs.shape == (2, 84, 84, 4)
        assert len(infos) == 2
        assert all("curriculum_frontier" in i for i in infos)
    finally:
        venv.close()


def test_route_level_mismatch_rejected(tmp_path):
    save_route(_make_route(level="1-1"), tmp_path)
    with pytest.raises(ValueError, match="route is for level"):
        make_mario_env(level="1-2", snapshot_dir=str(tmp_path))


@pytest.mark.parametrize(
    ("metadata", "match"),
    [
        ({"action_set": "complex"}, "action set"),
        ({"decision_skip": 1}, "decision skip"),
    ],
)
def test_snapshot_route_compatibility_rejected_before_emulator_creation(
    tmp_path, monkeypatch, metadata, match
):
    route = {**_make_route(), **metadata}
    save_route(route, tmp_path)
    monkeypatch.setattr(
        gym_super_mario_bros,
        "make",
        lambda *_args, **_kwargs: pytest.fail(
            "incompatible route reached emulator creation"
        ),
    )

    with pytest.raises(ValueError, match=match):
        make_mario_env(
            level="1-1",
            action_set="simple",
            skip=4,
            snapshot_dir=str(tmp_path),
        )


def test_advance_threshold_reaches_schedule(tmp_path):
    route = {
        "level": "test", "actions": [],
        "waypoints": [{"index": i, "frame": 0, "x_pos": 40}
                      for i in range(3)],
    }
    w = SnapshotStartWrapper(_FakeSnapEnv(), route, advance_threshold=0.25)
    assert w._schedule.advance_threshold == 0.25
