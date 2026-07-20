import numpy as np
from stable_baselines3 import PPO
from marioai.envs import make_vec_env


def test_ppo_trains_briefly_with_finite_loss():
    venv = make_vec_env(["1-1"], n_envs=1, monitor=True)
    try:
        model = PPO("CnnPolicy", venv, n_steps=128, batch_size=64,
                    device="cpu", verbose=0)
        model.learn(total_timesteps=512)
        loss = model.logger.name_to_value.get("train/loss", 0.0)
        assert np.isfinite(loss)
    finally:
        venv.close()
