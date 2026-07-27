from collections import Counter
import pickle

import numpy as np
import yaml
from stable_baselines3 import PPO

import marioai.record_gif as recording
import marioai.train as training
from marioai.envs import make_vec_env
from marioai.evaluate import evaluate_checkpoint
from marioai.results import EvaluationReport, sha256_file
from marioai.train import (
    build_policy_kwargs,
    load_training_config,
    validate_resume_model,
)


class _NoOverrides:
    levels = ["1-1"]
    timesteps = None
    n_envs = 1
    lr = None
    ent_coef = None
    level_weights_json = None


class _TemporaryConfigOverrides(_NoOverrides):
    levels = None
    n_envs = None


def _run_all32_pipeline_smoke(tmp_path, monkeypatch):
    config_path = tmp_path / "all32-smoke.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "levels": {"train": ["1-1", "1-2"]},
                "env": {
                    "action_set": "complex",
                    "skip": 4,
                    "frame_stack": 4,
                    "shape": 84,
                },
                "train": {
                    "n_envs": 2,
                    "total_timesteps": 64,
                    "device": "cpu",
                    "seed": 42,
                    "checkpoint_freq": 64,
                    "normalize_reward": True,
                },
                "policy": {
                    "extractor": "impala",
                    "features_dim": 512,
                    "channels": [16, 32, 32],
                },
                "ppo": {
                    "n_steps": 32,
                    "batch_size": 64,
                    "n_epochs": 1,
                    "gamma": 0.9,
                    "learning_rate": 0.00025,
                    "clip_range": 0.1,
                    "ent_coef": 0.01,
                    "vf_coef": 0.5,
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = load_training_config(str(config_path), None, _TemporaryConfigOverrides())
    assert cfg["train"]["n_envs"] == 2
    monkeypatch.chdir(tmp_path)
    training.main(
        [
            "--config",
            str(config_path),
            "--run-name",
            "all32-smoke",
        ]
    )
    first_run = tmp_path / "models" / "all32-smoke"
    first_model_path = first_run / "final.zip"
    first_normalization_path = first_run / "vecnormalize.pkl"
    checkpoint_normalization_path = (
        first_run / "ckpt_vecnormalize_64_steps.pkl"
    )
    assert first_model_path.is_file()
    assert first_normalization_path.is_file()
    assert checkpoint_normalization_path.is_file()
    with first_normalization_path.open("rb") as source:
        first_normalization = pickle.load(source)

    training.main(
        [
            "--config",
            str(config_path),
            "--resume",
            str(first_model_path),
            "--run-name",
            "all32-smoke-resumed",
        ]
    )
    resumed_run = tmp_path / "models" / "all32-smoke-resumed"
    model_path = resumed_run / "final.zip"
    resumed_normalization_path = resumed_run / "vecnormalize.pkl"
    with resumed_normalization_path.open("rb") as source:
        resumed_normalization = pickle.load(source)
    assert resumed_normalization.ret_rms.count > first_normalization.ret_rms.count

    resumed = PPO.load(model_path, device="cpu")
    validate_resume_model(resumed, **training.compatibility_kwargs(cfg))

    report_path = tmp_path / "all32-smoke.json"
    report = evaluate_checkpoint(
        model_path,
        levels=cfg["levels"],
        episodes=1,
        seed=42000,
        deterministic=False,
    )
    report.write(report_path)
    gif_path = tmp_path / "all32-smoke.gif"
    gif_metadata = recording.record(
        resumed,
        "1-1",
        gif_path,
        rollouts=1,
        max_steps=2,
    )
    return (
        resumed,
        model_path,
        EvaluationReport.read(report_path),
        gif_metadata,
    )


def test_ppo_trains_briefly_with_finite_loss():
    cfg = load_training_config("configs/all32.yaml", "phase_1", _NoOverrides())
    venv = make_vec_env(
        cfg["levels"],
        n_envs=cfg["train"]["n_envs"],
        monitor=True,
        action_set=cfg["env"]["action_set"],
        level_weights=cfg["train"]["level_weights"],
    )
    try:
        model = PPO(
            "CnnPolicy",
            venv,
            n_steps=64,
            batch_size=64,
            policy_kwargs=build_policy_kwargs(cfg),
            device="cpu",
            verbose=0,
        )
        model.learn(total_timesteps=64)
        loss = model.logger.name_to_value.get("train/loss", 0.0)
        assert np.isfinite(loss)
    finally:
        venv.close()


def test_all32_pipeline_trains_resumes_evaluates_and_records(
    tmp_path, monkeypatch
):
    resumed, model_path, report, gif_metadata = _run_all32_pipeline_smoke(
        tmp_path,
        monkeypatch,
    )

    assert resumed.num_timesteps >= 128
    assert report.checkpoint_sha256 == sha256_file(model_path)
    assert report.deterministic is False
    assert report.requested_episodes == 1
    assert set(report.stages) == {"1-1", "1-2"}
    assert len(report.rollouts) == 2
    assert Counter(result.level for result in report.rollouts) == {
        "1-1": 1,
        "1-2": 1,
    }
    assert all(result.seed >= 42000 for result in report.rollouts)
    assert gif_metadata["loop"] == 0
    assert gif_metadata["semantic_frames"] >= 2
    assert gif_metadata["total_duration_ms"] == (
        gif_metadata["semantic_frames"]
        * gif_metadata["frame_duration_ms"]
    )
