"""Config-driven PPO training entrypoint."""

import argparse
import copy
import json
import os
from pathlib import Path
import re
from collections.abc import Mapping

import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.torch_layers import NatureCNN
from stable_baselines3.common.utils import LinearSchedule

from marioai.actions import action_set_size
from marioai.envs import make_vec_env
from marioai.features import ImpalaCnnFeaturesExtractor


def matching_vecnormalize_path(checkpoint_path: str | Path) -> Path:
    """Return the normalization sidecar written with one model checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    match = re.fullmatch(r"(?P<prefix>.+)_(?P<steps>\d+)_steps", checkpoint_path.stem)
    if match is not None:
        name = (
            f"{match.group('prefix')}_vecnormalize_"
            f"{match.group('steps')}_steps.pkl"
        )
        return checkpoint_path.with_name(name)
    if checkpoint_path.stem == "final":
        return checkpoint_path.with_name("vecnormalize.pkl")
    return checkpoint_path.with_suffix(".vecnormalize.pkl")


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


def compatibility_kwargs(cfg: Mapping) -> dict:
    """Return the complete checkpoint signature required by this config."""
    policy = cfg.get("policy", {})
    extractor_name = policy.get("extractor", "nature")
    return {
        "action_count": action_set_size(
            cfg["env"].get("action_set", "simple")
        ),
        "observation_shape": (
            cfg["env"]["shape"],
            cfg["env"]["shape"],
        ),
        "frame_stack": cfg["env"]["frame_stack"],
        "extractor_name": extractor_name,
        "features_dim": (
            policy.get("features_dim")
            if extractor_name == "impala"
            else None
        ),
        "channels": (
            tuple(policy["channels"])
            if extractor_name == "impala"
            else None
        ),
    }


def validate_resume_model(
    model: PPO,
    action_count: int,
    observation_shape: tuple[int, int],
    frame_stack: int,
    extractor_name: str,
    features_dim: int | None,
    channels: tuple[int, ...] | None,
) -> None:
    """Reject a checkpoint that cannot continue the configured run."""
    checkpoint_action_count = model.action_space.n
    if checkpoint_action_count != action_count:
        raise ValueError(
            f"checkpoint action count {checkpoint_action_count} does not match "
            f"configured action count {action_count}"
        )

    checkpoint_shape = tuple(model.observation_space.shape)
    expected_spatial_shape = tuple(observation_shape)
    if len(checkpoint_shape) != 3:
        raise ValueError(
            f"checkpoint observation shape {checkpoint_shape} is not a "
            "three-dimensional frame stack"
        )
    if checkpoint_shape[1:] == expected_spatial_shape:
        checkpoint_frame_stack = checkpoint_shape[0]
    elif checkpoint_shape[:2] == expected_spatial_shape:
        checkpoint_frame_stack = checkpoint_shape[2]
    else:
        raise ValueError(
            f"checkpoint observation shape {checkpoint_shape} does not match "
            f"configured observation shape {expected_spatial_shape}"
        )
    if checkpoint_frame_stack != frame_stack:
        raise ValueError(
            f"checkpoint frame stack {checkpoint_frame_stack} does not match "
            f"configured frame stack {frame_stack}"
        )

    extractors = {
        "impala": ImpalaCnnFeaturesExtractor,
        "nature": NatureCNN,
    }
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
    if extractor_name != "impala":
        return
    if features_dim is None or channels is None:
        raise ValueError(
            "IMPALA compatibility requires feature width and channel "
            "configuration"
        )

    first_convolution = next(
        (
            module
            for module in actual_extractor.cnn
            if isinstance(module, torch.nn.Conv2d)
        ),
        None,
    )
    if first_convolution is None:
        raise ValueError("checkpoint IMPALA extractor has no input convolution")
    if first_convolution.in_channels != frame_stack:
        raise ValueError(
            f"checkpoint IMPALA input channels "
            f"{first_convolution.in_channels} do not match configured frame "
            f"stack {frame_stack}"
        )
    if actual_extractor.features_dim != features_dim:
        raise ValueError(
            f"checkpoint feature width {actual_extractor.features_dim} does "
            f"not match configured feature width {features_dim}"
        )
    checkpoint_channels = tuple(
        module.out_channels
        for module in actual_extractor.cnn
        if isinstance(module, torch.nn.Conv2d)
    )
    configured_channels = tuple(channels)
    if checkpoint_channels != configured_channels:
        raise ValueError(
            f"checkpoint IMPALA channels {checkpoint_channels} do not match "
            f"configured IMPALA channels {configured_channels}"
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
        "--resume-vecnormalize",
        type=Path,
        default=None,
        help=(
            "Matching VecNormalize state for --resume; inferred from standard "
            "final/checkpoint names when omitted."
        ),
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


def build_training_env(
    cfg: Mapping,
    args: argparse.Namespace,
    vecnormalize_path: Path | None = None,
):
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
        vecnormalize_path=vecnormalize_path,
    )


def create_model(cfg: Mapping, args: argparse.Namespace, venv, device: str):
    """Create, resume, or initialize a PPO model for the configured run."""
    tensorboard_log = f"runs/{args.run_name}"
    if args.resume is not None:
        return PPO.load(
            args.resume,
            env=venv,
            device=device,
            tensorboard_log=tensorboard_log,
        )
    if args.init_from is not None:
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

    # Inspect checkpoint compatibility before constructing paid parallel workers.
    checkpoint_source = (
        args.resume if args.resume is not None else args.init_from
    )
    resume_vecnormalize_path = None
    if checkpoint_source is not None:
        checkpoint = PPO.load(checkpoint_source, device=device)
        validate_resume_model(checkpoint, **compatibility_kwargs(cfg))
        del checkpoint
    if args.resume is not None and cfg["train"]["normalize_reward"]:
        resume_vecnormalize_path = (
            args.resume_vecnormalize
            if args.resume_vecnormalize is not None
            else matching_vecnormalize_path(args.resume)
        )
        if not resume_vecnormalize_path.is_file():
            raise FileNotFoundError(
                "matching VecNormalize state is required to resume normalized "
                f"training: {resume_vecnormalize_path}"
            )
    elif args.resume_vecnormalize is not None:
        raise ValueError(
            "--resume-vecnormalize requires --resume with reward "
            "normalization enabled"
        )

    venv = build_training_env(
        cfg,
        args,
        vecnormalize_path=resume_vecnormalize_path,
    )
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
            save_vecnormalize=cfg["train"]["normalize_reward"],
        )
        callbacks = [checkpoint_callback]
        if args.start_snapshots:
            callbacks.append(CurriculumLogCallback())

        reset_timesteps = args.resume is None or args.reset_timesteps
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
