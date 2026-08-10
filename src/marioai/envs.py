"""Mario env factory + multi-task vectorized env assembly."""
from collections.abc import Mapping

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecFrameStack,
    VecMonitor,
    VecNormalize,
)
from .curriculum import load_route
from .actions import resolve_action_set
from .sampling import assign_worker_levels
from .wrappers import SkipFrame, GrayScaleResize, SnapshotStartWrapper


def make_mario_env(level="1-1", skip=4, shape=84, render_mode="rgb_array",
                   capture_frames=False, snapshot_dir=None, snapshot_seed=0,
                   curriculum_threshold=0.5, action_set: str = "simple"):
    """Build a single fully-wrapped Mario env for one level (e.g. '1-1').

    capture_frames=True makes SkipFrame buffer every intra-skip native frame in
    `last_frames` (for smooth GIF recording); leave False for training.

    snapshot_dir activates reverse-curriculum starts: episodes begin from
    emulator snapshots rebuilt by replaying the route emitted by
    scripts/solve_level.py, sampled near the flag first and sliding back
    toward the level start as the policy improves. snapshot_seed
    decorrelates the sampling streams of parallel workers. curriculum_threshold
    is the clear-rate needed to advance the frontier.
    """
    route = None
    if snapshot_dir:
        route = load_route(snapshot_dir)
        if route["level"] != level:
            raise ValueError(
                f"route is for level {route['level']!r}, env is {level!r}"
            )
        if route.get("action_set") != action_set:
            raise ValueError(
                f"snapshot route action set {route.get('action_set')!r} does "
                f"not match environment action set {action_set!r}"
            )
        stored_skip = route.get("decision_skip")
        if (
            isinstance(stored_skip, bool)
            or not isinstance(stored_skip, int)
            or stored_skip != skip
        ):
            raise ValueError(
                f"snapshot route decision skip {stored_skip!r} does not "
                f"match environment skip {skip}"
            )
    env = gym_super_mario_bros.make(
        f"SuperMarioBros-{level}-v0", render_mode=render_mode
    )
    env = JoypadSpace(env, resolve_action_set(action_set))
    if route is not None:
        env = SnapshotStartWrapper(env, route, seed=snapshot_seed,
                                   advance_threshold=curriculum_threshold)
    env = SkipFrame(env, skip=skip, capture_frames=capture_frames)
    env = GrayScaleResize(env, shape=shape)
    return env


def make_vec_env(levels, n_envs, frame_stack=4, skip=4, shape=84,
                 normalize_reward=False, monitor=True, snapshot_dir=None,
                 curriculum_threshold=0.5, action_set: str = "simple",
                 level_weights: Mapping[str, float] | None = None,
                 vecnormalize_path=None):
    """SubprocVecEnv of n_envs Marios with each worker fixed to one stage.

    Fixing one level per worker (rather than recreating a random level on each
    reset) avoids nes-py's known memory leak on repeated env creation, while the
    shared PPO update still pools experience across all levels (multi-task).
    Extra workers are allocated by deterministic weighted sampling.
    """
    def _thunk(level, worker_idx):
        def _init():
            return make_mario_env(level=level, skip=skip, shape=shape,
                                  snapshot_dir=snapshot_dir,
                                  snapshot_seed=worker_idx,
                                  curriculum_threshold=curriculum_threshold,
                                  action_set=action_set)
        return _init

    assigned = assign_worker_levels(levels, n_envs, level_weights)
    venv = SubprocVecEnv([_thunk(lvl, i) for i, lvl in enumerate(assigned)])
    if monitor:
        venv = VecMonitor(venv)
    venv = VecFrameStack(venv, n_stack=frame_stack, channels_order="last")
    if vecnormalize_path is not None and not normalize_reward:
        venv.close()
        raise ValueError(
            "cannot restore VecNormalize when reward normalization is disabled"
        )
    if normalize_reward:
        if vecnormalize_path is None:
            venv = VecNormalize(
                venv,
                norm_obs=False,
                norm_reward=True,
                clip_reward=10.0,
            )
        else:
            try:
                venv = VecNormalize.load(str(vecnormalize_path), venv)
            except Exception:
                venv.close()
                raise
            if venv.norm_obs or not venv.norm_reward:
                venv.close()
                raise ValueError(
                    "VecNormalize state does not match reward-only "
                    "normalization"
                )
            venv.training = True
    return venv
