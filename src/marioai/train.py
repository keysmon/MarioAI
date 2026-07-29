"""Config-driven PPO training entrypoint."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from collections.abc import Mapping

import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.torch_layers import NatureCNN
from stable_baselines3.common.utils import LinearSchedule

from marioai.actions import action_set_size, resolve_action_set
from marioai.envs import make_vec_env
from marioai.features import ImpalaCnnFeaturesExtractor


_SAFE_RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 of one regular checkpoint artifact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Durably replace ``path`` only after the full payload reaches disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=path.suffix,
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as temporary:
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _write_durable_artifact(path: Path, writer) -> None:
    """Publish an immutable artifact, accepting only identical prior bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=path.suffix,
        dir=path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        writer(temporary_path)
        if not temporary_path.is_file():
            raise OSError(f"checkpoint writer did not create {temporary_path}")
        with temporary_path.open("rb") as artifact:
            os.fsync(artifact.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            if (
                path.is_symlink()
                or not path.is_file()
                or path.stat().st_size != temporary_path.stat().st_size
                or sha256_file(path) != sha256_file(temporary_path)
            ):
                raise FileExistsError(
                    f"immutable checkpoint artifact differs: {path}"
                ) from None
        else:
            _fsync_directory(path.parent)
    except BaseException:
        raise
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_immutable_bytes(path: Path, payload: bytes) -> None:
    """Durably publish bytes without ever replacing an existing generation."""
    _write_durable_artifact(
        path, lambda temporary: temporary.write_bytes(payload)
    )


def checkpoint_resume_signature(
    cfg: Mapping,
    phase: str | None,
    *,
    start_snapshots: str | None = None,
    curriculum_threshold: float = 0.5,
) -> dict:
    """Return the complete policy/environment/normalization resume identity."""
    environment = cfg["env"]
    policy = cfg.get("policy", {})
    extractor = policy.get("extractor", "nature")
    extractor_classes = {
        "impala": ImpalaCnnFeaturesExtractor,
        "nature": NatureCNN,
    }
    try:
        extractor_class = extractor_classes[extractor]
    except KeyError as exc:
        raise ValueError(f"unknown policy extractor {extractor!r}") from exc
    shape = environment["shape"]
    frame_stack = environment["frame_stack"]
    normalize_reward = bool(cfg["train"]["normalize_reward"])
    action_set = environment.get("action_set", "simple")
    actions = [list(action) for action in resolve_action_set(action_set)]
    action_payload = json.dumps(
        actions, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "schema_version": 1,
        "phase": phase,
        "environment": {
            "levels": list(cfg["levels"]),
            "level_weights": copy.deepcopy(
                cfg["train"].get("level_weights", {})
            ),
            "action_set": action_set,
            "actions": actions,
            "action_set_sha256": hashlib.sha256(
                action_payload
            ).hexdigest(),
            "action_count": action_set_size(action_set),
            "skip": environment["skip"],
            "frame_stack": frame_stack,
            "channels_order": "last",
            "shape": shape,
            "observation_shape": [frame_stack, shape, shape],
            "start_snapshots": start_snapshots,
            "curriculum_threshold": curriculum_threshold,
        },
        "policy": {
            "extractor": extractor,
            "extractor_class": (
                f"{extractor_class.__module__}.{extractor_class.__name__}"
            ),
            "features_dim": (
                policy.get("features_dim") if extractor == "impala" else None
            ),
            "channels": (
                list(policy["channels"]) if extractor == "impala" else None
            ),
            "normalize_images": True,
        },
        "normalization": {
            "normalize_reward": normalize_reward,
            "vecnormalize_required": normalize_reward,
            "norm_obs": False,
            "norm_reward": normalize_reward,
            "clip_obs": 10.0,
            "clip_reward": 10.0,
            "gamma": 0.99,
            "epsilon": 1e-8,
        },
    }


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
    path: str,
    phase: str | None,
    overrides: argparse.Namespace,
) -> dict:
    """Load a config, select a phase, and apply explicit CLI overrides."""
    with open(path) as config_file:
        cfg = copy.deepcopy(yaml.safe_load(config_file))

    if bool(getattr(overrides, "phase_resolved_config", False)):
        levels = cfg.get("levels")
        if not isinstance(levels, list):
            raise ValueError(
                "phase-resolved config must contain a levels list"
            )
    else:
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


class DurableCheckpointCallback(CheckpointCallback):
    """Publish complete, content-addressed resume bundles via ``latest.json``."""

    def __init__(
        self,
        *,
        save_path: str | Path,
        save_freq: int,
        run_config: Mapping,
        phase: str | None,
        run_name: str,
        budget_ledger_path: str | Path,
        name_prefix: str = "ckpt",
        save_vecnormalize: bool = False,
        verbose: int = 0,
        start_snapshots: str | None = None,
        curriculum_threshold: float = 0.5,
    ) -> None:
        if (
            not isinstance(run_name, str)
            or _SAFE_RUN_NAME.fullmatch(run_name) is None
        ):
            raise ValueError("run_name is unsafe for checkpoint filenames")
        if phase is not None and (
            not isinstance(phase, str)
            or _SAFE_RUN_NAME.fullmatch(phase) is None
        ):
            raise ValueError("phase is unsafe for checkpoint manifests")
        self.run_config = copy.deepcopy(dict(run_config))
        self.phase = phase
        self.run_name = run_name
        self.budget_ledger_path = Path(budget_ledger_path)
        self.signature_payload = checkpoint_resume_signature(
            self.run_config,
            phase,
            start_snapshots=start_snapshots,
            curriculum_threshold=curriculum_threshold,
        )
        super().__init__(
            save_freq=save_freq,
            save_path=str(save_path),
            name_prefix=name_prefix,
            save_vecnormalize=save_vecnormalize,
            verbose=verbose,
        )

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self.save_checkpoint(self.model)
        return True

    def save_checkpoint(self, model) -> Path:
        """Durably write one bundle, publishing its manifest last."""
        num_timesteps = getattr(model, "num_timesteps", None)
        if (
            isinstance(num_timesteps, bool)
            or not isinstance(num_timesteps, int)
            or num_timesteps < 0
        ):
            raise ValueError("model num_timesteps must be a nonnegative integer")
        validate_resume_model(
            model, **compatibility_kwargs(self.run_config)
        )
        save_path = Path(self.save_path)
        model_name = (
            f"{self.name_prefix}_{num_timesteps}_steps.zip"
        )
        vecnormalize_name = (
            f"{self.name_prefix}_vecnormalize_{num_timesteps}_steps.pkl"
        )
        signature_name = (
            f"{self.name_prefix}_signature_{num_timesteps}_steps.json"
        )
        run_config_name = (
            f"{self.name_prefix}_run_config_{num_timesteps}_steps.yaml"
        )
        budget_ledger_name = (
            f"{self.name_prefix}_budget_ledger_{num_timesteps}_steps.json"
        )

        model_path = save_path / model_name
        _write_durable_artifact(
            model_path, lambda temporary: model.save(str(temporary))
        )

        vecnormalize_path: Path | None = None
        if self.save_vecnormalize:
            vecnormalize = model.get_vec_normalize_env()
            if vecnormalize is None:
                raise ValueError(
                    "reward-normalized checkpoint has no VecNormalize state"
                )
            vecnormalize_path = save_path / vecnormalize_name
            _write_durable_artifact(
                vecnormalize_path,
                lambda temporary: vecnormalize.save(str(temporary)),
            )

        run_config_path = save_path / run_config_name
        _write_immutable_bytes(
            run_config_path,
            yaml.safe_dump(
                self.run_config, sort_keys=False
            ).encode("utf-8"),
        )
        if not self.budget_ledger_path.is_file():
            raise FileNotFoundError(
                "budget-ledger snapshot is required for a durable checkpoint: "
                f"{self.budget_ledger_path}"
            )
        budget_ledger_path = save_path / budget_ledger_name
        _write_immutable_bytes(
            budget_ledger_path, self.budget_ledger_path.read_bytes()
        )
        signature_path = save_path / signature_name
        _write_immutable_bytes(
            signature_path,
            (
                json.dumps(
                    self.signature_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8"),
        )

        manifest = {
            "schema_version": 1,
            "run_name": self.run_name,
            "phase": self.phase,
            "num_timesteps": num_timesteps,
            "model": model_name,
            "sha256": sha256_file(model_path),
            "action_set": self.signature_payload["environment"][
                "action_set"
            ],
            "action_count": self.signature_payload["environment"][
                "action_count"
            ],
            "extractor": self.signature_payload["policy"]["extractor"],
            "extractor_class": self.signature_payload["policy"][
                "extractor_class"
            ],
            "normalize_reward": self.signature_payload["normalization"][
                "normalize_reward"
            ],
            "vecnormalize": (
                vecnormalize_name if vecnormalize_path is not None else None
            ),
            "vecnormalize_sha256": (
                sha256_file(vecnormalize_path)
                if vecnormalize_path is not None
                else None
            ),
            "signature": signature_name,
            "signature_sha256": sha256_file(signature_path),
            "run_config": run_config_name,
            "run_config_sha256": sha256_file(run_config_path),
            "budget_ledger": budget_ledger_name,
            "budget_ledger_sha256": sha256_file(budget_ledger_path),
        }
        manifest_path = save_path / "latest.json"
        _atomic_write_bytes(
            manifest_path,
            (
                json.dumps(manifest, sort_keys=True, separators=(",", ":"))
                + "\n"
            ).encode("utf-8"),
        )
        return manifest_path


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
    parser.add_argument(
        "--phase-resolved-config",
        action="store_true",
        help=(
            "Treat --config as an already phase-resolved checkpoint config; "
            "--phase remains run identity metadata."
        ),
    )
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
    parser.add_argument(
        "--budget-ledger-snapshot",
        type=Path,
        default=None,
        help=(
            "Durable AWS budget-ledger provenance to pair with every "
            "checkpoint bundle."
        ),
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

        out_dir = Path("models") / args.run_name
        out_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_bytes(
            out_dir / "run-config.yaml",
            yaml.safe_dump(cfg, sort_keys=False).encode("utf-8"),
        )
        budget_ledger_path = args.budget_ledger_snapshot
        if budget_ledger_path is None:
            budget_ledger_path = out_dir / "training-ledger.json"
            _atomic_write_bytes(
                budget_ledger_path,
                b'{"kind":"training-only","schema_version":1}\n',
            )

        checkpoint_callback = DurableCheckpointCallback(
            save_freq=max(cfg["train"]["checkpoint_freq"] // n_envs, 1),
            save_path=out_dir,
            name_prefix="ckpt",
            run_config=cfg,
            phase=args.phase,
            run_name=args.run_name,
            budget_ledger_path=budget_ledger_path,
            save_vecnormalize=cfg["train"]["normalize_reward"],
            start_snapshots=args.start_snapshots,
            curriculum_threshold=args.curriculum_threshold,
        )
        callbacks = [checkpoint_callback]
        if args.start_snapshots:
            callbacks.append(CurriculumLogCallback())

        reset_timesteps = args.resume is None or args.reset_timesteps
        learn_timesteps = cfg["train"]["total_timesteps"]
        if args.resume is not None and not reset_timesteps:
            checkpoint_timesteps = getattr(model, "num_timesteps", None)
            if (
                isinstance(checkpoint_timesteps, bool)
                or not isinstance(checkpoint_timesteps, int)
                or checkpoint_timesteps < 0
            ):
                raise ValueError(
                    "resumed model has invalid num_timesteps"
                )
            learn_timesteps = max(
                learn_timesteps - checkpoint_timesteps, 0
            )
        if learn_timesteps > 0:
            model.learn(
                total_timesteps=learn_timesteps,
                callback=callbacks,
                reset_num_timesteps=reset_timesteps,
            )
        model.save(str(out_dir / "final"))
        if cfg["train"]["normalize_reward"]:
            venv.save(str(out_dir / "vecnormalize.pkl"))
    finally:
        venv.close()
    print(f"SAVED {out_dir}/final.zip")


if __name__ == "__main__":
    main()
