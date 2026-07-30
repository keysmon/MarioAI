from argparse import Namespace
import hashlib
import json
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
    inspect_callback=False,
    inspect_timesteps=False,
    checkpoint_timesteps=0,
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
        num_timesteps=checkpoint_timesteps,
        action_space=SimpleNamespace(n=12),
        observation_space=gym.spaces.Box(
            0, 255, shape=(4, 84, 84), dtype=np.uint8
        ),
        policy=SimpleNamespace(features_extractor=extractor),
    )

    class FakeModel:
        num_timesteps = checkpoint_timesteps
        action_space = gym.spaces.Discrete(12)
        observation_space = gym.spaces.Box(
            0, 255, shape=(4, 84, 84), dtype=np.uint8
        )
        policy = SimpleNamespace(features_extractor=extractor)

        def learn(
            self,
            *,
            total_timesteps,
            reset_num_timesteps,
            callback,
            **_kwargs,
        ):
            if inspect_callback:
                events.append(
                    ("callback", type(callback[0]).__name__)
                )
            if inspect_timesteps:
                events.append(("timesteps", total_timesteps))
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
    ledger_path = tmp_path / "aws-spend.json"
    ledger_path.write_text("{}", encoding="utf-8")
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
    if inspect_callback:
        argv.extend(["--budget-ledger-snapshot", str(ledger_path)])
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


def test_training_installs_durable_checkpoint_callback(monkeypatch, tmp_path):
    """Catches wiring the old independently-written SB3 checkpoint callback."""
    events = _run_orchestration(
        monkeypatch,
        tmp_path,
        normalize_reward=True,
        inspect_callback=True,
    )

    assert ("callback", "DurableCheckpointCallback") in events


def test_resume_trains_only_remaining_phase_timesteps(monkeypatch, tmp_path):
    """Catches SB3 adding a second full phase budget after interruption."""
    events = _run_orchestration(
        monkeypatch,
        tmp_path,
        source_flag="--resume",
        checkpoint_timesteps=16,
        inspect_timesteps=True,
    )

    assert ("timesteps", 48) in events


def test_durable_checkpoint_writes_complete_resume_manifest(tmp_path):
    """Catches publishing a checkpoint without every artifact needed to resume."""
    cfg = _orchestration_config(normalize_reward=True)
    ledger_path = tmp_path / "aws-spend.json"
    ledger_path.write_text(
        json.dumps(
            {
                "allocations": {"phase_1": "16.00"},
                "cap_usd": "50.00",
                "runs": [],
                "spent_usd": "0",
            }
        ),
        encoding="utf-8",
    )

    class FakeVecNormalize:
        def save(self, path):
            Path(path).write_bytes(b"paired normalization")

    class FakeModel:
        num_timesteps = 250000
        action_space = gym.spaces.Discrete(12)
        observation_space = gym.spaces.Box(
            0, 255, shape=(4, 84, 84), dtype=np.uint8
        )
        policy = SimpleNamespace(
            features_extractor=_impala_extractor(
                features_dim=512,
                channels=(16, 32, 32),
            )
        )

        def save(self, path):
            Path(path).write_bytes(b"durable model")

        def get_vec_normalize_env(self):
            return FakeVecNormalize()

    callback = training.DurableCheckpointCallback(
        save_path=tmp_path,
        save_freq=1,
        run_config=cfg,
        budget_ledger_path=ledger_path,
        phase="phase_1",
        run_name="all32-phase_1",
        save_vecnormalize=True,
    )

    callback.save_checkpoint(FakeModel())

    manifest = json.loads(
        (tmp_path / "latest.json").read_text(encoding="utf-8")
    )
    assert manifest["num_timesteps"] == 250000
    assert manifest["model"] == "ckpt_250000_steps.zip"
    assert manifest["sha256"] == hashlib.sha256(
        b"durable model"
    ).hexdigest()
    assert manifest["phase"] == "phase_1"
    assert manifest["action_set"] == "complex"
    assert manifest["extractor"] == "impala"
    assert manifest["vecnormalize"] == (
        "ckpt_vecnormalize_250000_steps.pkl"
    )
    assert manifest["signature"] == "ckpt_signature_250000_steps.json"
    assert manifest["run_config"] == "ckpt_run_config_250000_steps.yaml"
    assert manifest["budget_ledger"] == (
        "ckpt_budget_ledger_250000_steps.json"
    )
    for field in (
        "model",
        "vecnormalize",
        "signature",
        "run_config",
        "budget_ledger",
    ):
        artifact = tmp_path / manifest[field]
        assert artifact.is_file()
        hash_field = "sha256" if field == "model" else f"{field}_sha256"
        assert manifest[hash_field] == hashlib.sha256(
            artifact.read_bytes()
        ).hexdigest()
    signature = json.loads(
        (tmp_path / manifest["signature"]).read_text(encoding="utf-8")
    )
    assert signature == {
        "environment": {
            "action_set_sha256": (
                "84fc5c090e80b377473270d02377b5dcb"
                "94c8c269d6144f3d868ce897f824e68"
            ),
            "actions": [
                ["NOOP"],
                ["right"],
                ["right", "A"],
                ["right", "B"],
                ["right", "A", "B"],
                ["A"],
                ["left"],
                ["left", "A"],
                ["left", "B"],
                ["left", "A", "B"],
                ["down"],
                ["up"],
            ],
            "action_count": 12,
            "action_set": "complex",
            "channels_order": "last",
            "curriculum_threshold": 0.5,
            "frame_stack": 4,
            "level_weights": {},
            "levels": ["1-1"],
            "observation_shape": [4, 84, 84],
            "shape": 84,
            "skip": 4,
            "start_snapshots": None,
        },
        "normalization": {
            "clip_obs": 10.0,
            "clip_reward": 10.0,
            "epsilon": 1e-08,
            "gamma": 0.99,
            "norm_obs": False,
            "norm_reward": True,
            "normalize_reward": True,
            "vecnormalize_required": True,
        },
        "phase": "phase_1",
        "policy": {
            "channels": [16, 32, 32],
            "extractor": "impala",
            "extractor_class": (
                "marioai.features.ImpalaCnnFeaturesExtractor"
            ),
            "features_dim": 512,
            "normalize_images": True,
        },
        "schema_version": 1,
    }


def test_durable_checkpoint_publishes_immutable_generation_manifest(
    tmp_path,
):
    """Every phase candidate must be addressable without a mutable head."""
    cfg = _orchestration_config(normalize_reward=False)
    ledger_path = tmp_path / "aws-spend.json"
    ledger_path.write_text('{"schema_version":1}\n', encoding="utf-8")

    class FakeModel:
        num_timesteps = 250000
        action_space = gym.spaces.Discrete(12)
        observation_space = gym.spaces.Box(
            0, 255, shape=(4, 84, 84), dtype=np.uint8
        )
        policy = SimpleNamespace(
            features_extractor=_impala_extractor(
                features_dim=512,
                channels=(16, 32, 32),
            )
        )

        def save(self, path):
            Path(path).write_bytes(b"candidate model")

    callback = training.DurableCheckpointCallback(
        save_path=tmp_path,
        save_freq=1,
        run_config=cfg,
        budget_ledger_path=ledger_path,
        phase="phase_1",
        run_name="all32-phase_1",
    )

    generation_manifest = callback.save_checkpoint(FakeModel())

    assert generation_manifest == (
        tmp_path / "ckpt_manifest_250000_steps.json"
    )
    assert generation_manifest.is_file()
    assert (tmp_path / "latest.json").read_bytes() == (
        generation_manifest.read_bytes()
    )


def test_phase_training_returns_complete_final_generation_manifest(
    monkeypatch, tmp_path
):
    """The candidate handed to diagnostics is the final durable bundle."""
    cfg = _orchestration_config(normalize_reward=False)
    ledger = tmp_path / "ledger.json"
    ledger.write_text('{"schema_version":1}\n', encoding="utf-8")
    callback_types = []

    class FakeModel:
        num_timesteps = 0
        action_space = gym.spaces.Discrete(12)
        observation_space = gym.spaces.Box(
            0, 255, shape=(4, 84, 84), dtype=np.uint8
        )
        policy = SimpleNamespace(
            features_extractor=_impala_extractor(
                features_dim=512,
                channels=(16, 32, 32),
            )
        )

        def learn(self, *, total_timesteps, callback, **_kwargs):
            callback_types.extend(
                type(item).__name__ for item in callback
            )
            self.num_timesteps += total_timesteps

        def save(self, path):
            Path(path).with_suffix(".zip").write_bytes(b"final candidate")

    class FakeEnvironment:
        def close(self):
            pass

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        training,
        "load_training_config",
        lambda *_args, **_kwargs: cfg,
    )
    monkeypatch.setattr(
        training, "build_training_env", lambda *_args, **_kwargs: FakeEnvironment()
    )
    monkeypatch.setattr(
        training, "create_model", lambda *_args, **_kwargs: FakeModel()
    )

    manifest_path = training.main(
        [
            "--phase",
            "phase_1",
            "--run-name",
            "phase-chunk",
            "--budget-ledger-snapshot",
            str(ledger),
            "--publish-final-checkpoint",
        ]
    )

    assert manifest_path == (
        tmp_path
        / "models"
        / "phase-chunk"
        / "ckpt_manifest_64_steps.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["num_timesteps"] == 64
    assert callback_types == [
        "DurableCheckpointCallback",
        "StopAtTimestepsCallback",
    ]
    for field, hash_field in (
        ("model", "sha256"),
        ("run_config", "run_config_sha256"),
        ("signature", "signature_sha256"),
        ("budget_ledger", "budget_ledger_sha256"),
    ):
        artifact = manifest_path.parent / manifest[field]
        assert artifact.is_file()
        assert training.sha256_file(artifact) == manifest[hash_field]


def test_deadline_callback_stops_training_before_outer_timeout():
    callback = training.StopAtDeadlineCallback(
        deadline_epoch=100,
        clock=lambda: 101,
    )
    callback.model = SimpleNamespace(num_timesteps=32)

    assert callback._on_step() is False
    assert callback.stopped_at_timesteps == 32


def test_phase_chunk_callback_stops_at_exact_absolute_target():
    callback = training.StopAtTimestepsCallback(2_000_000)
    callback.model = SimpleNamespace(num_timesteps=1_999_936)
    assert callback._on_step() is True

    callback.model.num_timesteps = 2_000_000
    assert callback._on_step() is False


def test_durable_checkpoint_same_timestep_artifacts_are_immutable(tmp_path):
    """Catches same-generation checkpoint bytes being silently replaced."""
    cfg = _orchestration_config(normalize_reward=False)
    ledger_path = tmp_path / "aws-spend.json"
    ledger_path.write_text(
        json.dumps(
            {
                "allocations": {"phase_1": "16.00"},
                "cap_usd": "50.00",
                "runs": [],
                "spent_usd": "0",
            }
        ),
        encoding="utf-8",
    )

    class FakeModel:
        num_timesteps = 250000
        action_space = gym.spaces.Discrete(12)
        observation_space = gym.spaces.Box(
            0, 255, shape=(4, 84, 84), dtype=np.uint8
        )
        policy = SimpleNamespace(
            features_extractor=_impala_extractor(
                features_dim=512,
                channels=(16, 32, 32),
            )
        )

        def __init__(self, payload):
            self.payload = payload

        def save(self, path):
            Path(path).write_bytes(self.payload)

    callback = training.DurableCheckpointCallback(
        save_path=tmp_path,
        save_freq=1,
        run_config=cfg,
        budget_ledger_path=ledger_path,
        phase="phase_1",
        run_name="all32-phase_1",
    )
    model_path = tmp_path / "ckpt_250000_steps.zip"

    callback.save_checkpoint(FakeModel(b"original model"))
    original_inode = model_path.stat().st_ino
    original_manifest = (tmp_path / "latest.json").read_bytes()

    callback.save_checkpoint(FakeModel(b"original model"))
    assert model_path.stat().st_ino == original_inode

    with pytest.raises(FileExistsError, match="immutable"):
        callback.save_checkpoint(FakeModel(b"different model"))
    assert model_path.read_bytes() == b"original model"
    assert (tmp_path / "latest.json").read_bytes() == original_manifest


def test_durable_checkpoint_does_not_publish_partial_bundle(tmp_path):
    """Catches latest.json pointing at a model whose paired state failed."""
    previous_manifest = b'{"num_timesteps": 125000}\n'
    (tmp_path / "latest.json").write_bytes(previous_manifest)
    ledger_path = tmp_path / "aws-spend.json"
    ledger_path.write_text("{}", encoding="utf-8")

    class BrokenVecNormalize:
        def save(self, _path):
            raise OSError("interrupted normalization write")

    class FakeModel:
        num_timesteps = 250000
        action_space = gym.spaces.Discrete(12)
        observation_space = gym.spaces.Box(
            0, 255, shape=(4, 84, 84), dtype=np.uint8
        )
        policy = SimpleNamespace(
            features_extractor=_impala_extractor(
                features_dim=512,
                channels=(16, 32, 32),
            )
        )

        def save(self, path):
            Path(path).write_bytes(b"durable model")

        def get_vec_normalize_env(self):
            return BrokenVecNormalize()

    callback = training.DurableCheckpointCallback(
        save_path=tmp_path,
        save_freq=1,
        run_config=_orchestration_config(normalize_reward=True),
        budget_ledger_path=ledger_path,
        phase="phase_1",
        run_name="all32-phase_1",
        save_vecnormalize=True,
    )

    with pytest.raises(OSError, match="interrupted normalization"):
        callback.save_checkpoint(FakeModel())

    assert (tmp_path / "latest.json").read_bytes() == previous_manifest
