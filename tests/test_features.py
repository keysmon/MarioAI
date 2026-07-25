import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO

from marioai.envs import make_vec_env
from marioai.features import ImpalaCnnFeaturesExtractor


def test_impala_extractor_maps_stacked_frames_to_feature_width():
    """Catches a broken channel-last conversion or projection width."""
    space = gym.spaces.Box(0, 255, shape=(84, 84, 4), dtype="uint8")
    extractor = ImpalaCnnFeaturesExtractor(space, features_dim=256)

    output = extractor(torch.zeros(2, 84, 84, 4))

    assert output.shape == (2, 256)
    assert torch.isfinite(output).all()


def test_impala_policy_trains_one_small_update():
    """Catches an extractor that cannot consume PPO-preprocessed frames."""
    venv = make_vec_env(["1-1"], n_envs=1, action_set="complex")
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

        model.learn(64)

        assert np.isfinite(model.logger.name_to_value["train/loss"])
    finally:
        venv.close()
