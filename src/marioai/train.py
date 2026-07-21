"""Config-driven PPO training entrypoint."""
import argparse
import os
import yaml
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import LinearSchedule
from marioai.envs import make_vec_env


def resolve_device(name):
    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--levels", nargs="+", default=None,
                   help="Override training levels, e.g. --levels 1-1 1-2")
    p.add_argument("--timesteps", type=int, default=None)
    p.add_argument("--n-envs", type=int, default=None,
                   help="Override parallel env count (lower = less memory).")
    p.add_argument("--init-from", default=None,
                   help="Fine-tune: load policy weights from this model .zip and "
                        "continue training on --levels (transfer from a strong base).")
    p.add_argument("--lr", type=float, default=None,
                   help="Constant learning rate override (use a low value like 5e-5 "
                        "when fine-tuning, so a converged policy isn't destabilized).")
    p.add_argument("--run-name", required=True)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    levels = args.levels or cfg["levels"]["train"]
    timesteps = args.timesteps or cfg["train"]["total_timesteps"]
    n_envs = args.n_envs or cfg["train"]["n_envs"]
    device = resolve_device(cfg["train"]["device"])
    ppo = cfg["ppo"]

    venv = make_vec_env(
        levels, n_envs=n_envs,
        frame_stack=cfg["env"]["frame_stack"], skip=cfg["env"]["skip"],
        shape=cfg["env"]["shape"], normalize_reward=cfg["train"]["normalize_reward"],
    )

    if args.init_from:
        # Fine-tune: load the pretrained policy, attach the new (single-level) env,
        # and continue. reset_num_timesteps=True (below) restarts the LR schedule.
        print(f"FINE-TUNE from {args.init_from}")
        model = PPO.load(args.init_from, env=venv, device=device,
                         tensorboard_log=f"runs/{args.run_name}")
        if args.lr:
            # Override the restored (high, scheduled) LR with a low constant one so
            # the converged policy is nudged, not knocked off its solution.
            model.learning_rate = args.lr
            model.lr_schedule = lambda _progress, _lr=args.lr: _lr
    else:
        model = PPO(
            "CnnPolicy", venv, device=device, seed=cfg["train"]["seed"],
            n_steps=ppo["n_steps"], batch_size=ppo["batch_size"],
            n_epochs=ppo["n_epochs"], gamma=ppo["gamma"],
            learning_rate=LinearSchedule(ppo["learning_rate"], 0.0, 1.0),
            clip_range=ppo["clip_range"], ent_coef=ppo["ent_coef"],
            vf_coef=ppo["vf_coef"], tensorboard_log=f"runs/{args.run_name}", verbose=1,
        )

    out_dir = f"models/{args.run_name}"
    os.makedirs(out_dir, exist_ok=True)
    ckpt = CheckpointCallback(
        save_freq=max(cfg["train"]["checkpoint_freq"] // n_envs, 1),
        save_path=out_dir, name_prefix="ckpt",
    )
    # verbose=1 prints the per-rollout table (incl. ep_rew_mean); TensorBoard logs
    # the full curves. No progress_bar to avoid the extra `rich` dependency.
    model.learn(total_timesteps=timesteps, callback=ckpt, reset_num_timesteps=True)
    model.save(f"{out_dir}/final")
    if cfg["train"]["normalize_reward"]:
        venv.save(f"{out_dir}/vecnormalize.pkl")
    venv.close()
    print(f"SAVED {out_dir}/final.zip")


if __name__ == "__main__":
    main()
