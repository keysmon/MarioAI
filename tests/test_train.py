from argparse import Namespace
from types import SimpleNamespace

import pytest

from marioai.features import ImpalaCnnFeaturesExtractor
from marioai.train import (
    build_policy_kwargs,
    load_training_config,
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


def test_resume_rejects_seven_action_checkpoint():
    model = SimpleNamespace(
        action_space=SimpleNamespace(n=7),
        policy=SimpleNamespace(features_extractor=object()),
    )

    with pytest.raises(ValueError, match="checkpoint action count 7"):
        validate_resume_model(model, action_count=12, extractor_name="impala")


def test_resume_rejects_incompatible_extractor():
    model = SimpleNamespace(
        action_space=SimpleNamespace(n=12),
        policy=SimpleNamespace(features_extractor=object()),
    )

    with pytest.raises(ValueError, match="checkpoint extractor object"):
        validate_resume_model(model, action_count=12, extractor_name="impala")


def test_resume_accepts_matching_action_count_and_extractor():
    extractor = object.__new__(ImpalaCnnFeaturesExtractor)
    model = SimpleNamespace(
        action_space=SimpleNamespace(n=12),
        policy=SimpleNamespace(features_extractor=extractor),
    )

    validate_resume_model(model, action_count=12, extractor_name="impala")
