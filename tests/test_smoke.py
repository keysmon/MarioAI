import numpy as np
from stable_baselines3 import PPO
from marioai.envs import make_vec_env
from marioai.train import build_policy_kwargs, load_training_config


class _NoOverrides:
    levels = ["1-1"]
    timesteps = None
    n_envs = 1
    lr = None
    ent_coef = None
    level_weights_json = None


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
