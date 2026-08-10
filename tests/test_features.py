import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from marioai.envs import make_mario_env
from marioai.features import ImpalaCnnFeaturesExtractor


def test_impala_extractor_maps_stacked_frames_to_feature_width():
    """Catches a broken channel-last conversion or projection width."""
    space = gym.spaces.Box(0, 255, shape=(84, 84, 4), dtype="uint8")
    extractor = ImpalaCnnFeaturesExtractor(space, features_dim=256)

    output = extractor(torch.zeros(2, 84, 84, 4))

    assert output.shape == (2, 256)
    assert torch.isfinite(output).all()


def test_ppo_wrapped_impala_uses_four_input_channels_and_trains():
    """Catches treating SB3's transposed (4, 84, 84) space as 84 channels."""
    venv = VecFrameStack(
        DummyVecEnv(
            [lambda: make_mario_env("1-1", action_set="complex")]
        ),
        n_stack=4,
        channels_order="last",
    )
    try:
        model = PPO(
            "CnnPolicy",
            venv,
            n_steps=64,
            batch_size=64,
            policy_kwargs={
                "features_extractor_class": ImpalaCnnFeaturesExtractor,
                "features_extractor_kwargs": {"features_dim": 256},
                "normalize_images": True,
            },
            device="cpu",
        )

        first_convolution = model.policy.features_extractor.cnn[0]
        assert model.observation_space.shape == (4, 84, 84)
        assert first_convolution.in_channels == 4

        model.learn(64)

        assert np.isfinite(model.logger.name_to_value["train/loss"])
    finally:
        venv.close()
