from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
from torch import nn
from stable_baselines3.common.torch_layers import NatureCNN

import marioai.train as training
from marioai.features import ImpalaCnnFeaturesExtractor
from marioai.train import (
    build_policy_kwargs,
    load_training_config,
    matching_vecnormalize_path,
    validate_resume_model,
)


def _no_overrides(**changes):
    values = {
        "levels": None,
        "timesteps": None,
        "n_envs": None,
        "lr": None,
        "ent_coef": None,
        "level_weights_json": None,
    }
    values.update(changes)
    return Namespace(**values)


def test_all32_phase_two_contains_every_stage_and_complex_actions():
    cfg = load_training_config("configs/all32.yaml", "phase_2", _no_overrides())

    assert len(cfg["levels"]) == 32
    assert cfg["env"]["action_set"] == "complex"
    assert cfg["train"]["n_envs"] == 64
    assert cfg["policy"]["extractor"] == "impala"


def test_phase_and_cli_overrides_are_composed_without_dropping_zero():
    cfg = load_training_config(
        "configs/all32.yaml",
        "phase_1",
        _no_overrides(
            levels=["1-1"],
            timesteps=0,
            n_envs=0,
            lr=0.0,
            ent_coef=0.0,
            level_weights_json='{"1-1": 2.0}',
        ),
    )

    assert cfg["levels"] == ["1-1"]
    assert cfg["train"]["total_timesteps"] == 0
    assert cfg["train"]["n_envs"] == 0
    assert cfg["ppo"]["learning_rate"] == 0.0
    assert cfg["ppo"]["ent_coef"] == 0.0
    assert cfg["train"]["level_weights"] == {"1-1": 2.0}


def test_build_policy_kwargs_selects_configured_impala_architecture():
    cfg = load_training_config("configs/all32.yaml", "phase_2", _no_overrides())

    assert build_policy_kwargs(cfg) == {
        "features_extractor_class": ImpalaCnnFeaturesExtractor,
        "features_extractor_kwargs": {
            "features_dim": 512,
            "channels": (16, 32, 32),
        },
        "normalize_images": True,
    }


def _compatibility_kwargs(**changes):
    values = {
        "action_count": 12,
        "observation_shape": (84, 84),
        "frame_stack": 4,
        "extractor_name": "impala",
        "features_dim": 32,
        "channels": (2, 3, 4),
    }
    values.update(changes)
    return values


def _impala_extractor(
    *,
    input_channels=4,
    features_dim=32,
    channels=(2, 3, 4),
):
    space = gym.spaces.Box(
        0,
        255,
        shape=(input_channels, 84, 84),
        dtype=np.uint8,
    )
    return ImpalaCnnFeaturesExtractor(
        space,
        features_dim=features_dim,
        channels=channels,
    )


def _compatible_checkpoint(
    *,
    observation_shape=(4, 84, 84),
    input_channels=4,
    features_dim=32,
    channels=(2, 3, 4),
):
    return SimpleNamespace(
        action_space=gym.spaces.Discrete(12),
        observation_space=gym.spaces.Box(
            0,
            255,
            shape=observation_shape,
            dtype=np.uint8,
        ),
        policy=SimpleNamespace(
            features_extractor=_impala_extractor(
                input_channels=input_channels,
                features_dim=features_dim,
                channels=channels,
            )
        ),
    )


def test_resume_rejects_seven_action_checkpoint():
    model = SimpleNamespace(
        action_space=SimpleNamespace(n=7),
        observation_space=gym.spaces.Box(
            0, 255, shape=(4, 84, 84), dtype=np.uint8
        ),
        policy=SimpleNamespace(features_extractor=object()),
    )

    with pytest.raises(ValueError, match="checkpoint action count 7"):
        validate_resume_model(model, **_compatibility_kwargs())


def test_resume_rejects_incompatible_extractor():
    model = SimpleNamespace(
        action_space=SimpleNamespace(n=12),
        observation_space=gym.spaces.Box(
            0, 255, shape=(4, 84, 84), dtype=np.uint8
        ),
        policy=SimpleNamespace(features_extractor=object()),
    )

    with pytest.raises(ValueError, match="checkpoint extractor object"):
        validate_resume_model(model, **_compatibility_kwargs())


def test_resume_accepts_complete_matching_environment_and_policy_signature():
    model = _compatible_checkpoint()

    validate_resume_model(model, **_compatibility_kwargs())


@pytest.mark.parametrize(
    ("model", "match"),
    [
        (
            _compatible_checkpoint(observation_shape=(4, 96, 96)),
            "observation shape",
        ),
        (
            _compatible_checkpoint(
                observation_shape=(3, 84, 84),
                input_channels=3,
            ),
            "frame stack",
        ),
        (
            _compatible_checkpoint(features_dim=64),
            "feature width",
        ),
        (
            _compatible_checkpoint(channels=(2, 4, 4)),
            "IMPALA channels",
        ),
    ],
)
def test_resume_rejects_incompatible_geometry_or_impala_configuration(
    model, match
):
    with pytest.raises(ValueError, match=match):
        validate_resume_model(model, **_compatibility_kwargs())


def test_resume_rejects_pre_fix_impala_channel_geometry():
    model = _compatible_checkpoint()
    model.policy.features_extractor.cnn[0] = nn.Conv2d(
        84, 2, kernel_size=3, padding=1
    )

    with pytest.raises(ValueError, match="input channels 84"):
        validate_resume_model(model, **_compatibility_kwargs())


def test_legacy_init_compatibility_accepts_nature_cnn():
    observation_space = gym.spaces.Box(
        0, 255, shape=(4, 84, 84), dtype=np.uint8
    )
    model = SimpleNamespace(
        action_space=SimpleNamespace(n=7),
        observation_space=observation_space,
        policy=SimpleNamespace(
            features_extractor=NatureCNN(observation_space)
        ),
    )

    validate_resume_model(
        model,
        action_count=7,
        observation_shape=(84, 84),
        frame_stack=4,
        extractor_name="nature",
        features_dim=None,
        channels=None,
    )


@pytest.mark.parametrize(
    ("checkpoint", "expected"),
    [
        (
            Path("models/run/final.zip"),
            Path("models/run/vecnormalize.pkl"),
        ),
        (
            Path("models/run/ckpt_250000_steps.zip"),
            Path("models/run/ckpt_vecnormalize_250000_steps.pkl"),
        ),
    ],
)
def test_matching_vecnormalize_path_uses_model_checkpoint_identity(
    checkpoint, expected
):
    assert matching_vecnormalize_path(checkpoint) == expected


def _orchestration_config(*, normalize_reward=False):
    return {
        "levels": ["1-1"],
        "env": {
            "action_set": "complex",
            "frame_stack": 4,
            "skip": 4,
            "shape": 84,
        },
        "train": {
            "n_envs": 1,
            "total_timesteps": 64,
            "device": "cpu",
            "seed": 42,
            "checkpoint_freq": 64,
            "normalize_reward": normalize_reward,
            "level_weights": {},
        },
        "policy": {
            "extractor": "impala",
            "features_dim": 512,
            "channels": [16, 32, 32],
        },
        "ppo": {
            "n_steps": 64,
            "batch_size": 64,
            "n_epochs": 1,
            "gamma": 0.9,
            "learning_rate": 0.00025,
            "clip_range": 0.1,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
        },
    }


def _run_orchestration(
    monkeypatch,
    tmp_path,
    source_flag=None,
    incompatible=False,
    events=None,
    reset_timesteps=False,
    normalize_reward=False,
    create_normalization=False,
):
    events = [] if events is None else events
    extractor = (
        object()
        if incompatible
        else _impala_extractor(
            features_dim=512,
            channels=(16, 32, 32),
        )
    )
    checkpoint = SimpleNamespace(
        action_space=SimpleNamespace(n=12),
        observation_space=gym.spaces.Box(
            0, 255, shape=(4, 84, 84), dtype=np.uint8
        ),
        policy=SimpleNamespace(features_extractor=extractor),
    )

    class FakeModel:
        def learn(self, *, reset_num_timesteps, callback, **_kwargs):
            if callback[0].save_vecnormalize:
                events.append(("checkpoint-vecnormalize", True))
            events.append(("learn", reset_num_timesteps))

        def save(self, _path):
            events.append(("save",))

    class FakePPO:
        def __new__(cls, *_args, **_kwargs):
            events.append(("construct",))
            return FakeModel()

        @staticmethod
        def load(path, *, env=None, **_kwargs):
            events.append(("load", path, "env" if env is not None else "no-env"))
            return checkpoint if env is None else FakeModel()

    class FakeVecEnv:
        def save(self, path):
            events.append(("vecnormalize-save", path))

        def close(self):
            events.append(("close",))

    def fake_build_training_env(_cfg, _args, vecnormalize_path=None):
        if vecnormalize_path is None:
            events.append(("env",))
        else:
            events.append(("env", Path(vecnormalize_path)))
        return FakeVecEnv()

    monkeypatch.chdir(tmp_path)
    if create_normalization:
        (tmp_path / "checkpoint.vecnormalize.pkl").write_bytes(
            b"normalization state"
        )
    monkeypatch.setattr(training, "PPO", FakePPO)
    monkeypatch.setattr(
        training,
        "load_training_config",
        lambda _path, _phase, _overrides: _orchestration_config(
            normalize_reward=normalize_reward
        ),
    )
    monkeypatch.setattr(training, "build_training_env", fake_build_training_env)

    argv = ["--run-name", "orchestration"]
    if source_flag is not None:
        argv.extend([source_flag, "checkpoint.zip"])
    if reset_timesteps:
        argv.append("--reset-timesteps")
    training.main(argv)
    return events


@pytest.mark.parametrize(
    ("source_flag", "reset_timesteps", "expected"),
    [
        (
            None,
            False,
            [
                ("env",),
                ("construct",),
                ("learn", True),
                ("save",),
                ("close",),
            ],
        ),
        (
            "--init-from",
            False,
            [
                ("load", "checkpoint.zip", "no-env"),
                ("env",),
                ("load", "checkpoint.zip", "env"),
                ("learn", True),
                ("save",),
                ("close",),
            ],
        ),
        (
            "--resume",
            False,
            [
                ("load", "checkpoint.zip", "no-env"),
                ("env",),
                ("load", "checkpoint.zip", "env"),
                ("learn", False),
                ("save",),
                ("close",),
            ],
        ),
        (
            "--resume",
            True,
            [
                ("load", "checkpoint.zip", "no-env"),
                ("env",),
                ("load", "checkpoint.zip", "env"),
                ("learn", True),
                ("save",),
                ("close",),
            ],
        ),
    ],
)
def test_checkpoint_orchestration_preflights_before_env_and_resets_by_mode(
    monkeypatch, tmp_path, source_flag, reset_timesteps, expected
):
    events = _run_orchestration(
        monkeypatch,
        tmp_path,
        source_flag,
        reset_timesteps=reset_timesteps,
    )

    assert events == expected


def test_incompatible_init_from_fails_before_vector_workers(
    monkeypatch, tmp_path
):
    events = []
    with pytest.raises(ValueError, match="checkpoint extractor object"):
        _run_orchestration(
            monkeypatch,
            tmp_path,
            "--init-from",
            incompatible=True,
            events=events,
        )
    assert events == [("load", "checkpoint.zip", "no-env")]


def test_normalized_resume_requires_matching_state_before_vector_workers(
    monkeypatch, tmp_path
):
    events = []

    with pytest.raises(FileNotFoundError, match="VecNormalize"):
        _run_orchestration(
            monkeypatch,
            tmp_path,
            "--resume",
            normalize_reward=True,
            events=events,
        )

    assert events == [("load", "checkpoint.zip", "no-env")]


def test_normalized_resume_restores_and_saves_matching_state(
    monkeypatch, tmp_path
):
    events = _run_orchestration(
        monkeypatch,
        tmp_path,
        "--resume",
        normalize_reward=True,
        create_normalization=True,
    )

    assert events == [
        ("load", "checkpoint.zip", "no-env"),
        ("env", Path("checkpoint.vecnormalize.pkl")),
        ("load", "checkpoint.zip", "env"),
        ("checkpoint-vecnormalize", True),
        ("learn", False),
        ("save",),
        (
            "vecnormalize-save",
            "models/orchestration/vecnormalize.pkl",
        ),
        ("close",),
    ]
