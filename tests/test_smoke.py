import numpy as np
import yaml
from stable_baselines3 import PPO

from marioai.actions import action_set_size
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


def _run_all32_pipeline_smoke(tmp_path):
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
                    "device": "cpu",
                    "seed": 42,
                    "checkpoint_freq": 64,
                    "normalize_reward": False,
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
        ),
        encoding="utf-8",
    )
    cfg = load_training_config(
        str(config_path), None, _TemporaryConfigOverrides()
    )
    assert cfg["train"]["n_envs"] == 2
    model_path = tmp_path / "all32-smoke.zip"

    venv = make_vec_env(
        cfg["levels"],
        n_envs=cfg["train"]["n_envs"],
        frame_stack=cfg["env"]["frame_stack"],
        skip=cfg["env"]["skip"],
        shape=cfg["env"]["shape"],
        action_set=cfg["env"]["action_set"],
    )
    try:
        assert venv.num_envs == 2
        model = PPO(
            "CnnPolicy",
            venv,
            n_steps=cfg["ppo"]["n_steps"],
            batch_size=cfg["ppo"]["batch_size"],
            n_epochs=cfg["ppo"]["n_epochs"],
            policy_kwargs=build_policy_kwargs(cfg),
            device="cpu",
            seed=cfg["train"]["seed"],
            verbose=0,
        )
        model.learn(total_timesteps=64)
        model.save(model_path)
    finally:
        venv.close()

    resumed_venv = make_vec_env(
        cfg["levels"],
        n_envs=cfg["train"]["n_envs"],
        frame_stack=cfg["env"]["frame_stack"],
        skip=cfg["env"]["skip"],
        shape=cfg["env"]["shape"],
        action_set=cfg["env"]["action_set"],
    )
    try:
        assert resumed_venv.num_envs == 2
        resumed = PPO.load(model_path, env=resumed_venv, device="cpu")
        validate_resume_model(
            resumed,
            action_count=action_set_size(cfg["env"]["action_set"]),
            extractor_name=cfg["policy"]["extractor"],
        )
        resumed.learn(total_timesteps=64, reset_num_timesteps=False)
        resumed.save(model_path)
    finally:
        resumed_venv.close()

    report_path = tmp_path / "all32-smoke.json"
    report = evaluate_checkpoint(
        model_path,
        levels=cfg["levels"],
        episodes=1,
        seed=42000,
        deterministic=False,
    )
    report.write(report_path)
    return resumed, model_path, EvaluationReport.read(report_path)


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


def test_all32_pipeline_trains_resumes_and_evaluates(tmp_path):
    resumed, model_path, report = _run_all32_pipeline_smoke(tmp_path)

    assert resumed.num_timesteps >= 128
    assert report.checkpoint_sha256 == sha256_file(model_path)
    assert set(report.stages) == {"1-1", "1-2"}
    assert all(result.seed >= 42000 for result in report.rollouts)
