"""Config-driven PPO training entrypoint."""

import argparse
import copy
import json
import os
from collections.abc import Mapping

import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.utils import LinearSchedule

from marioai.actions import action_set_size
from marioai.envs import make_vec_env
from marioai.features import ImpalaCnnFeaturesExtractor


def resolve_device(name):
    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


def _override(overrides: argparse.Namespace, name: str):
    return getattr(overrides, name, None)


def load_training_config(
    path: str, phase: str | None, overrides: argparse.Namespace
) -> dict:
    """Load a config, select a phase, and apply explicit CLI overrides."""
    with open(path) as config_file:
        cfg = copy.deepcopy(yaml.safe_load(config_file))

    phases = cfg.get("phases")
    if phases is not None:
        if phase not in phases:
            raise ValueError(f"unknown training phase {phase!r}")
        cfg["train"].update(phases[phase])
        levels = cfg["levels"][phase]
    else:
        levels = cfg["levels"]["train"]
    cfg["levels"] = list(levels)
    cfg["train"].setdefault("level_weights", {})

    if _override(overrides, "levels") is not None:
        cfg["levels"] = list(overrides.levels)
    if _override(overrides, "timesteps") is not None:
        cfg["train"]["total_timesteps"] = overrides.timesteps
    if _override(overrides, "n_envs") is not None:
        cfg["train"]["n_envs"] = overrides.n_envs
    if _override(overrides, "lr") is not None:
        cfg["ppo"]["learning_rate"] = overrides.lr
    if _override(overrides, "ent_coef") is not None:
        cfg["ppo"]["ent_coef"] = overrides.ent_coef

    weights_json = _override(overrides, "level_weights_json")
    if weights_json is not None:
        try:
            weights = json.loads(weights_json)
        except json.JSONDecodeError as exc:
            raise ValueError("--level-weights-json must be valid JSON") from exc
        if not isinstance(weights, dict):
            raise ValueError("--level-weights-json must contain a JSON object")
        cfg["train"]["level_weights"] = weights

    return cfg


def build_policy_kwargs(cfg: Mapping) -> dict:
    """Build Stable-Baselines3 policy arguments from training config."""
    policy = cfg.get("policy")
    if not policy:
        return {}
    if policy.get("extractor") != "impala":
        raise ValueError(f"unknown policy extractor {policy.get('extractor')!r}")
    return {
        "features_extractor_class": ImpalaCnnFeaturesExtractor,
        "features_extractor_kwargs": {
            "features_dim": policy["features_dim"],
            "channels": tuple(policy["channels"]),
        },
        "normalize_images": True,
    }


def validate_resume_model(
    model: PPO, action_count: int, extractor_name: str
) -> None:
    """Reject a checkpoint that cannot continue the configured run."""
    checkpoint_action_count = model.action_space.n
    if checkpoint_action_count != action_count:
        raise ValueError(
            f"checkpoint action count {checkpoint_action_count} does not match "
            f"configured action count {action_count}"
        )

    extractors = {"impala": ImpalaCnnFeaturesExtractor}
    try:
        expected_extractor = extractors[extractor_name]
    except KeyError as exc:
        raise ValueError(f"unknown policy extractor {extractor_name!r}") from exc
    actual_extractor = model.policy.features_extractor
    if not isinstance(actual_extractor, expected_extractor):
        raise ValueError(
            f"checkpoint extractor {type(actual_extractor).__name__} does not "
            f"match configured extractor {extractor_name}"
        )


class CurriculumLogCallback(BaseCallback):
    """TensorBoard curve of the reverse-curriculum frontier (0 = level start)."""

    def _on_step(self):
        fronts = [
            info["curriculum_frontier"]
            for info in self.locals["infos"]
            if "curriculum_frontier" in info
        ]
        if fronts:
            self.logger.record("curriculum/frontier_mean", sum(fronts) / len(fronts))
        return True


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--phase", choices=("phase_1", "phase_2"), default=None)
    parser.add_argument(
        "--levels",
        nargs="+",
        default=None,
        help="Override training levels, e.g. --levels 1-1 1-2",
    )
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument(
        "--n-envs",
        type=int,
        default=None,
        help="Override parallel env count (lower = less memory).",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--init-from",
        default=None,
        help=(
            "Fine-tune: load a compatible policy from this model .zip and reset "
            "the timestep schedule."
        ),
    )
    source.add_argument(
        "--resume",
        default=None,
        help="Resume a compatible checkpoint without resetting its timesteps.",
    )
    parser.add_argument(
        "--reset-timesteps",
        action="store_true",
        help="Restart the timestep and learning-rate schedules when resuming.",
    )
    parser.add_argument(
        "--level-weights-json",
        default=None,
        help='Override level weights with a JSON object, e.g. \'{"1-1": 2.0}\'.',
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help=(
            "Constant learning rate override (use a low value like 5e-5 when "
            "fine-tuning, so a converged policy is not destabilized)."
        ),
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=None,
        help=(
            "Entropy coefficient override (raise to ~0.05 for more exploration "
            "to break past a stubborn obstacle)."
        ),
    )
    parser.add_argument(
        "--start-snapshots",
        default=None,
        help=(
            "Route dir from scripts/solve_level.py: episodes start from snapshots "
            "rebuilt by replaying the route."
        ),
    )
    parser.add_argument(
        "--curriculum-threshold",
        type=float,
        default=0.5,
        help="Clear-rate needed to advance the reverse-curriculum frontier.",
    )
    parser.add_argument("--run-name", required=True)
    return parser.parse_args(argv)


def build_training_env(cfg: Mapping, args: argparse.Namespace):
    """Construct the configured vector environment."""
    return make_vec_env(
        cfg["levels"],
        n_envs=cfg["train"]["n_envs"],
        frame_stack=cfg["env"]["frame_stack"],
        skip=cfg["env"]["skip"],
        shape=cfg["env"]["shape"],
        normalize_reward=cfg["train"]["normalize_reward"],
        snapshot_dir=args.start_snapshots,
        curriculum_threshold=args.curriculum_threshold,
        action_set=cfg["env"].get("action_set", "simple"),
        level_weights=cfg["train"].get("level_weights"),
    )


def create_model(cfg: Mapping, args: argparse.Namespace, venv, device: str):
    """Create, resume, or initialize a PPO model for the configured run."""
    tensorboard_log = f"runs/{args.run_name}"
    if args.resume:
        return PPO.load(
            args.resume,
            env=venv,
            device=device,
            tensorboard_log=tensorboard_log,
        )
    if args.init_from:
        print(f"FINE-TUNE from {args.init_from}")
        return PPO.load(
            args.init_from,
            env=venv,
            device=device,
            tensorboard_log=tensorboard_log,
        )

    ppo = cfg["ppo"]
    return PPO(
        "CnnPolicy",
        venv,
        device=device,
        seed=cfg["train"]["seed"],
        n_steps=ppo["n_steps"],
        batch_size=ppo["batch_size"],
        n_epochs=ppo["n_epochs"],
        gamma=ppo["gamma"],
        learning_rate=LinearSchedule(ppo["learning_rate"], 0.0, 1.0),
        clip_range=ppo["clip_range"],
        ent_coef=ppo["ent_coef"],
        vf_coef=ppo["vf_coef"],
        policy_kwargs=build_policy_kwargs(cfg),
        tensorboard_log=tensorboard_log,
        verbose=1,
    )


def main(argv=None):
    args = parse_args(argv)
    try:
        cfg = load_training_config(args.config, args.phase, args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    levels = cfg["levels"]
    n_envs = cfg["train"]["n_envs"]
    device = resolve_device(cfg["train"]["device"])
    if args.start_snapshots and len(levels) != 1:
        raise SystemExit("--start-snapshots requires exactly one level")

    # Inspect resume compatibility before constructing paid parallel workers.
    if args.resume:
        checkpoint = PPO.load(args.resume, device=device)
        validate_resume_model(
            checkpoint,
            action_count=action_set_size(cfg["env"].get("action_set", "simple")),
            extractor_name=cfg.get("policy", {}).get("extractor", "nature"),
        )
        del checkpoint

    venv = build_training_env(cfg, args)
    try:
        model = create_model(cfg, args, venv, device)
        if args.init_from and args.lr is not None:
            model.learning_rate = args.lr
            model.lr_schedule = lambda _progress, _lr=args.lr: _lr
        if args.ent_coef is not None:
            model.ent_coef = args.ent_coef

        out_dir = f"models/{args.run_name}"
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "run-config.yaml"), "w") as config_file:
            yaml.safe_dump(cfg, config_file, sort_keys=False)

        checkpoint_callback = CheckpointCallback(
            save_freq=max(cfg["train"]["checkpoint_freq"] // n_envs, 1),
            save_path=out_dir,
            name_prefix="ckpt",
        )
        callbacks = [checkpoint_callback]
        if args.start_snapshots:
            callbacks.append(CurriculumLogCallback())

        reset_timesteps = not bool(args.resume) or args.reset_timesteps
        model.learn(
            total_timesteps=cfg["train"]["total_timesteps"],
            callback=callbacks,
            reset_num_timesteps=reset_timesteps,
        )
        model.save(f"{out_dir}/final")
        if cfg["train"]["normalize_reward"]:
            venv.save(f"{out_dir}/vecnormalize.pkl")
    finally:
        venv.close()
    print(f"SAVED {out_dir}/final.zip")


if __name__ == "__main__":
    main()
