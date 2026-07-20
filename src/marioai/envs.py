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
from .wrappers import SkipFrame, GrayScaleResize


def make_mario_env(level="1-1", skip=4, shape=84, render_mode="rgb_array",
                   capture_frames=False):
    """Build a single fully-wrapped Mario env for one level (e.g. '1-1').

    capture_frames=True makes SkipFrame buffer every intra-skip native frame in
    `last_frames` (for smooth GIF recording); leave False for training.
    """
    env = gym_super_mario_bros.make(
        f"SuperMarioBros-{level}-v0", render_mode=render_mode
    )
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = SkipFrame(env, skip=skip, capture_frames=capture_frames)
    env = GrayScaleResize(env, shape=shape)
    return env


def make_vec_env(levels, n_envs, frame_stack=4, skip=4, shape=84,
                 normalize_reward=False, monitor=True):
    """SubprocVecEnv of n_envs Marios; worker i is fixed to levels[i % len(levels)].

    Fixing one level per worker (rather than recreating a random level on each
    reset) avoids nes-py's known memory leak on repeated env creation, while the
    shared PPO update still pools experience across all levels (multi-task).
    """
    def _thunk(level):
        def _init():
            return make_mario_env(level=level, skip=skip, shape=shape)
        return _init

    assigned = [levels[i % len(levels)] for i in range(n_envs)]
    venv = SubprocVecEnv([_thunk(lvl) for lvl in assigned])
    if monitor:
        venv = VecMonitor(venv)
    venv = VecFrameStack(venv, n_stack=frame_stack, channels_order="last")
    if normalize_reward:
        venv = VecNormalize(venv, norm_obs=False, norm_reward=True, clip_reward=10.0)
    return venv
