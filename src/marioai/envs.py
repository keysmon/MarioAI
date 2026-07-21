"""Mario env factory + multi-task vectorized env assembly."""
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecFrameStack,
    VecMonitor,
    VecNormalize,
)
from .curriculum import load_route
from .wrappers import SkipFrame, GrayScaleResize, SnapshotStartWrapper


def make_mario_env(level="1-1", skip=4, shape=84, render_mode="rgb_array",
                   capture_frames=False, snapshot_dir=None, snapshot_seed=0):
    """Build a single fully-wrapped Mario env for one level (e.g. '1-1').

    capture_frames=True makes SkipFrame buffer every intra-skip native frame in
    `last_frames` (for smooth GIF recording); leave False for training.

    snapshot_dir activates reverse-curriculum starts: episodes begin from
    emulator snapshots rebuilt by replaying the route emitted by
    scripts/solve_level.py, sampled near the flag first and sliding back
    toward the level start as the policy improves. snapshot_seed
    decorrelates the sampling streams of parallel workers.
    """
    env = gym_super_mario_bros.make(
        f"SuperMarioBros-{level}-v0", render_mode=render_mode
    )
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    if snapshot_dir:
        env = SnapshotStartWrapper(env, load_route(snapshot_dir),
                                   seed=snapshot_seed)
    env = SkipFrame(env, skip=skip, capture_frames=capture_frames)
    env = GrayScaleResize(env, shape=shape)
    return env


def make_vec_env(levels, n_envs, frame_stack=4, skip=4, shape=84,
                 normalize_reward=False, monitor=True, snapshot_dir=None):
    """SubprocVecEnv of n_envs Marios; worker i is fixed to levels[i % len(levels)].

    Fixing one level per worker (rather than recreating a random level on each
    reset) avoids nes-py's known memory leak on repeated env creation, while the
    shared PPO update still pools experience across all levels (multi-task).
    """
    def _thunk(level, worker_idx):
        def _init():
            return make_mario_env(level=level, skip=skip, shape=shape,
                                  snapshot_dir=snapshot_dir,
                                  snapshot_seed=worker_idx)
        return _init

    assigned = [levels[i % len(levels)] for i in range(n_envs)]
    venv = SubprocVecEnv([_thunk(lvl, i) for i, lvl in enumerate(assigned)])
    if monitor:
        venv = VecMonitor(venv)
    venv = VecFrameStack(venv, n_stack=frame_stack, channels_order="last")
    if normalize_reward:
        venv = VecNormalize(venv, norm_obs=False, norm_reward=True, clip_reward=10.0)
    return venv
