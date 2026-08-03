#!/usr/bin/env python
"""Budget-guarded lifecycle orchestration for all-32 AWS training."""

from __future__ import annotations

import argparse
import base64
from collections.abc import Callable
import copy
from dataclasses import asdict, dataclass, replace
from decimal import Decimal, ROUND_CEILING
import fcntl
import ipaddress
import json
import math
import os
from pathlib import Path
import pickle
import re
import shutil
import shlex
import subprocess
import sys
import tempfile
import time
from typing import Any
from urllib.parse import urlsplit
import uuid

import yaml
from stable_baselines3.common.vec_env import VecNormalize

import marioai.train as training
if __package__:
    from scripts import train_phase as phase_training
else:
    import train_phase as phase_training
from marioai.aws import AwsCli, AwsConfig, PreflightResult, SpotOffer
from marioai.budget import BudgetExceeded, BudgetLedger, CostedRun


_SECONDS_PER_HOUR = Decimal("3600")
_HOURS_PER_BILLING_MONTH = Decimal("720")
_PREFLIGHT_AUTH_TTL_SECONDS = Decimal("300")
_AMI_ID_PATTERN = re.compile(r"ami-[0-9a-f]{8,17}")
_CLIENT_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,64}")
_INSTANCE_ID_PATTERN = re.compile(r"i-[0-9a-f]{8,17}")
_AWS_COMMAND_TIMEOUT_SECONDS = 60
_INSTANCE_READY_TIMEOUT_SECONDS = 600
_INSTANCE_READY_POLL_SECONDS = 5
_SSH_READY_TIMEOUT_SECONDS = 300
_SSH_READY_POLL_SECONDS = 5
_EC2_SETTLEMENT_RESERVE_SECONDS = Decimal("180")
_LAUNCH_STATE_SCHEMA_VERSION = 1
_CHECKPOINT_MANIFEST_SCHEMA_VERSION = 1
_CHECKPOINT_HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
_CHECKPOINT_FILENAME_PATTERN = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._-]*"
)
_BENCHMARK_ENVIRONMENT_STEPS = 250_000
_BENCHMARK_MAX_HOURS = Decimal("0.25")


class AwsLifecycleError(RuntimeError):
    """Raised when a paid-instance lifecycle cannot be handled safely."""


class AwsCapacityUnavailable(AwsLifecycleError):
    """Raised when EC2 definitively rejects a launch before creating it."""


@dataclass(frozen=True)
class BenchmarkObservation:
    """Exact remote duration and memory observation for one fixed workload."""

    environment_steps: int
    elapsed_seconds: Decimal
    peak_rss_gb: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.environment_steps, bool)
            or not isinstance(self.environment_steps, int)
            or self.environment_steps <= 0
        ):
            raise ValueError("environment_steps must be a positive integer")
        if (
            not isinstance(self.elapsed_seconds, Decimal)
            or not self.elapsed_seconds.is_finite()
            or self.elapsed_seconds <= 0
        ):
            raise ValueError(
                "elapsed_seconds must be a positive finite Decimal"
            )
        if (
            isinstance(self.peak_rss_gb, bool)
            or not isinstance(self.peak_rss_gb, (int, float))
            or not math.isfinite(self.peak_rss_gb)
            or self.peak_rss_gb < 0
        ):
            raise ValueError("peak_rss_gb must be finite and non-negative")


@dataclass(frozen=True)
class Benchmark:
    """One measured candidate's environment-step throughput and hourly rate."""

    instance_type: str
    env_steps_per_second: float
    instance_hourly_usd: Decimal
    volume_hourly_usd: Decimal = Decimal("0")
    peak_rss_gb: float = 0.0
    environment_steps: int = 250_000
    elapsed_seconds: Decimal | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.instance_type, str) or not self.instance_type:
            raise ValueError("instance_type must be a non-empty string")
        if (
            isinstance(self.env_steps_per_second, bool)
            or not isinstance(self.env_steps_per_second, (int, float))
            or not math.isfinite(self.env_steps_per_second)
            or self.env_steps_per_second <= 0
        ):
            raise ValueError(
                "env_steps_per_second must be positive and finite"
            )
        for field, value in (
            ("instance_hourly_usd", self.instance_hourly_usd),
            ("volume_hourly_usd", self.volume_hourly_usd),
        ):
            if (
                not isinstance(value, Decimal)
                or not value.is_finite()
                or value < 0
            ):
                raise ValueError(
                    f"{field} must be a finite non-negative Decimal"
                )
        if (
            isinstance(self.environment_steps, bool)
            or not isinstance(self.environment_steps, int)
            or self.environment_steps <= 0
        ):
            raise ValueError("environment_steps must be a positive integer")
        if self.elapsed_seconds is not None and (
            not isinstance(self.elapsed_seconds, Decimal)
            or not self.elapsed_seconds.is_finite()
            or self.elapsed_seconds <= 0
        ):
            raise ValueError(
                "elapsed_seconds must be a positive finite Decimal"
            )
        if (
            isinstance(self.peak_rss_gb, bool)
            or not isinstance(self.peak_rss_gb, (int, float))
            or not math.isfinite(self.peak_rss_gb)
            or self.peak_rss_gb < 0
        ):
            raise ValueError("peak_rss_gb must be finite and non-negative")

    @classmethod
    def from_observation(
        cls,
        *,
        instance_type: str,
        observation: BenchmarkObservation,
        instance_hourly_usd: Decimal,
        volume_hourly_usd: Decimal,
    ) -> Benchmark:
        if not isinstance(observation, BenchmarkObservation):
            raise ValueError(
                "observation must be a BenchmarkObservation"
            )
        return cls(
            instance_type=instance_type,
            env_steps_per_second=float(
                Decimal(observation.environment_steps)
                / observation.elapsed_seconds
            ),
            instance_hourly_usd=instance_hourly_usd,
            volume_hourly_usd=volume_hourly_usd,
            peak_rss_gb=float(observation.peak_rss_gb),
            environment_steps=observation.environment_steps,
            elapsed_seconds=observation.elapsed_seconds,
        )

    @property
    def cost_per_million_steps(self) -> Decimal:
        return (
            (self.instance_hourly_usd + self.volume_hourly_usd)
            * Decimal("1000000")
            / Decimal(str(self.env_steps_per_second))
            / _SECONDS_PER_HOUR
        )

    @property
    def observed_cost_usd(self) -> Decimal:
        elapsed_seconds = self.elapsed_seconds
        if elapsed_seconds is None:
            elapsed_seconds = (
                Decimal(self.environment_steps)
                / Decimal(str(self.env_steps_per_second))
            )
        return (
            elapsed_seconds
            / _SECONDS_PER_HOUR
            * (self.instance_hourly_usd + self.volume_hourly_usd)
        )

    def to_dict(self) -> dict[str, Any]:
        elapsed_seconds = self.elapsed_seconds
        if elapsed_seconds is None:
            elapsed_seconds = (
                Decimal(self.environment_steps)
                / Decimal(str(self.env_steps_per_second))
            )
        return {
            "instance_type": self.instance_type,
            "environment_steps": self.environment_steps,
            "elapsed_seconds": str(elapsed_seconds),
            "env_steps_per_second": float(self.env_steps_per_second),
            "instance_hourly_usd": str(self.instance_hourly_usd),
            "volume_hourly_usd": str(self.volume_hourly_usd),
            "cost_per_million_steps": str(
                self.cost_per_million_steps
            ),
            "peak_rss_gb": float(self.peak_rss_gb),
            "observed_cost_usd": format(
                self.observed_cost_usd.normalize(), "f"
            ),
        }


def select_benchmark(offers: list[Benchmark]) -> Benchmark:
    """Select the measured candidate with the lowest cost per million steps."""
    if not offers:
        raise ValueError("at least one benchmark measurement is required")
    if not all(isinstance(offer, Benchmark) for offer in offers):
        raise ValueError("benchmark measurements must be Benchmark values")
    return min(
        offers,
        key=lambda offer: (
            offer.cost_per_million_steps,
            offer.instance_type,
        ),
    )


@dataclass(frozen=True)
class ResumeBundle:
    """One locally staged and fully verified checkpoint generation."""

    root: Path
    manifest_path: Path
    model_path: Path
    run_config_path: Path
    signature_path: Path
    vecnormalize_path: Path | None
    budget_ledger_path: Path
    manifest: dict[str, Any]


@dataclass(frozen=True)
class PhaseResumeBundle:
    """Verified durable last/best phase lineage staged for remote resume."""

    root: Path
    lineage_path: Path
    lineage: phase_training.PhaseLineage


class S3CheckpointStore:
    """Exact-object, read-only AWS CLI boundary for resume artifacts."""

    def __init__(
        self,
        *,
        profile: str,
        region: str,
        runner: Callable[..., Any] = subprocess.run,
    ) -> None:
        self.profile = profile
        self.region = region
        self._runner = runner

    def download(self, uri: str, destination: Path) -> None:
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            dir=destination.parent,
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        try:
            command = [
                "aws",
                "s3",
                "cp",
                uri,
                str(temporary_path),
                "--only-show-errors",
                "--no-progress",
                "--profile",
                self.profile,
                "--region",
                self.region,
            ]
            self._runner(
                command,
                check=True,
                text=True,
                capture_output=True,
                timeout=_AWS_COMMAND_TIMEOUT_SECONDS,
            )
            if not temporary_path.is_file():
                raise AwsLifecycleError(
                    f"S3 download did not create {destination.name}"
                )
            with temporary_path.open("rb") as artifact:
                os.fsync(artifact.fileno())
            os.replace(temporary_path, destination)
            directory_fd = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except subprocess.TimeoutExpired as error:
            raise AwsLifecycleError(
                f"S3 download timed out for {uri}"
            ) from error
        except subprocess.CalledProcessError as error:
            stderr = (error.stderr or "").strip()
            detail = f": {stderr}" if stderr else ""
            raise AwsLifecycleError(
                f"S3 download failed for {uri}{detail}"
            ) from error
        except OSError as error:
            raise AwsLifecycleError(
                f"could not download checkpoint object {uri}: {error}"
            ) from error
        finally:
            temporary_path.unlink(missing_ok=True)


def _s3_location(uri: str) -> tuple[str, str]:
    if not isinstance(uri, str) or "%" in uri or "\\" in uri:
        raise AwsLifecycleError("checkpoint S3 URI is invalid")
    parsed = urlsplit(uri)
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or parsed.query
        or parsed.fragment
        or parsed.username is not None
        or parsed.port is not None
    ):
        raise AwsLifecycleError("checkpoint S3 URI is invalid")
    key = parsed.path.removeprefix("/")
    segments = key.split("/")
    if (
        not key
        or any(
            not segment or segment in {".", ".."} for segment in segments
        )
    ):
        raise AwsLifecycleError("checkpoint S3 URI is invalid")
    return parsed.netloc, key


def _manifest_object_prefix(config: AwsConfig, uri: str) -> str:
    root_bucket, root_key = _s3_location(config.s3_prefix.rstrip("/"))
    manifest_bucket, manifest_key = _s3_location(uri)
    configured_prefix = f"{root_key.rstrip('/')}/"
    if (
        manifest_bucket != root_bucket
        or not manifest_key.startswith(configured_prefix)
        or not manifest_key.endswith("/latest.json")
    ):
        raise AwsLifecycleError(
            "checkpoint manifest is outside the configured S3 prefix"
        )
    return uri.removesuffix("latest.json")


def _checkpoint_filename(
    value: Any, *, field: str, suffix: str
) -> str:
    if (
        not isinstance(value, str)
        or _CHECKPOINT_FILENAME_PATTERN.fullmatch(value) is None
        or Path(value).name != value
        or not value.endswith(suffix)
    ):
        raise AwsLifecycleError(
            f"checkpoint manifest has unsafe {field} filename"
        )
    return value


def _checkpoint_hash(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or _CHECKPOINT_HASH_PATTERN.fullmatch(value) is None
    ):
        raise AwsLifecycleError(
            f"checkpoint manifest has invalid {field} hash"
        )
    return value


def _resolved_all32_config(path: Path, phase: str) -> dict:
    overrides = argparse.Namespace(
        levels=None,
        timesteps=None,
        n_envs=None,
        lr=None,
        ent_coef=None,
        level_weights_json=None,
    )
    try:
        return training.load_training_config(
            str(path), phase, overrides
        )
    except (OSError, TypeError, ValueError) as error:
        raise AwsLifecycleError(
            f"cannot resolve trusted all-32 training config {path}"
        ) from error


def _default_model_validator(path: Path, cfg: dict) -> Any:
    device = training.resolve_device(cfg["train"]["device"])
    model = training.PPO.load(str(path), device=device)
    training.validate_resume_model(
        model, **training.compatibility_kwargs(cfg)
    )
    return model


def _validate_vecnormalize_checkpoint(
    path: Path, signature: dict[str, Any]
) -> None:
    """Deserialize and verify the actual normalization state and spaces."""
    try:
        with Path(path).open("rb") as sidecar:
            vecnormalize = pickle.load(sidecar)
    except Exception as error:
        raise AwsLifecycleError(
            "checkpoint VecNormalize state cannot be deserialized"
        ) from error
    if not isinstance(vecnormalize, VecNormalize):
        raise AwsLifecycleError(
            "checkpoint VecNormalize sidecar has the wrong type"
        )
    environment = signature["environment"]
    normalization = signature["normalization"]
    action_count = getattr(
        getattr(vecnormalize, "action_space", None), "n", None
    )
    observation_shape = tuple(
        getattr(
            getattr(vecnormalize, "observation_space", None),
            "shape",
            (),
        )
    )
    expected_settings = {
        "norm_obs": normalization["norm_obs"],
        "norm_reward": normalization["norm_reward"],
        "clip_obs": normalization["clip_obs"],
        "clip_reward": normalization["clip_reward"],
        "gamma": normalization["gamma"],
        "epsilon": normalization["epsilon"],
    }
    if (
        action_count != environment["action_count"]
        or observation_shape
        != tuple(environment["observation_shape"])
        or any(
            getattr(vecnormalize, field, object()) != expected
            for field, expected in expected_settings.items()
        )
    ):
        raise AwsLifecycleError(
            "checkpoint VecNormalize spaces/settings do not match "
            "the trusted signature"
        )


def restore_checkpoint_bundle(
    *,
    config: AwsConfig,
    phase: str,
    checkpoint_s3_uri: str,
    repo_dir: Path,
    ledger_path: Path,
    object_store: Any,
    model_validator: Callable[[Path, dict], Any] = _default_model_validator,
) -> ResumeBundle | PhaseResumeBundle:
    """Download and verify exactly one manifested generation before launch."""
    if phase not in {"phase_1", "phase_2"}:
        raise AwsLifecycleError(
            "only phase_1 and phase_2 shared-policy checkpoints can resume"
        )
    object_prefix = _manifest_object_prefix(config, checkpoint_s3_uri)
    staging_parent = Path(repo_dir) / ".resume"
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=f"{phase}-", dir=staging_parent)
    )
    try:
        manifest_path = staging_root / "latest.json"
        object_store.download(checkpoint_s3_uri, manifest_path)
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise AwsLifecycleError(
                "checkpoint manifest download is not a regular file"
            )
        try:
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise AwsLifecycleError(
                "checkpoint manifest is not valid UTF-8 JSON"
            ) from error
        if (
            isinstance(manifest, dict)
            and manifest.get("kind") == "marioai-phase-lineage"
        ):
            shutil.rmtree(staging_root)
            return restore_phase_lineage(
                config=config,
                phase=phase,
                lineage_s3_uri=checkpoint_s3_uri,
                repo_dir=repo_dir,
                ledger_path=ledger_path,
                object_store=object_store,
            )
        expected_fields = {
            "schema_version",
            "run_name",
            "phase",
            "num_timesteps",
            "model",
            "sha256",
            "action_set",
            "action_count",
            "extractor",
            "extractor_class",
            "normalize_reward",
            "vecnormalize",
            "vecnormalize_sha256",
            "signature",
            "signature_sha256",
            "run_config",
            "run_config_sha256",
            "budget_ledger",
            "budget_ledger_sha256",
        }
        if not isinstance(manifest, dict) or set(manifest) != expected_fields:
            raise AwsLifecycleError(
                "checkpoint manifest has an invalid schema"
            )
        if (
            manifest["schema_version"]
            != _CHECKPOINT_MANIFEST_SCHEMA_VERSION
            or manifest["phase"] != phase
            or isinstance(manifest["num_timesteps"], bool)
            or not isinstance(manifest["num_timesteps"], int)
            or manifest["num_timesteps"] < 0
            or not isinstance(manifest["run_name"], str)
            or _CHECKPOINT_FILENAME_PATTERN.fullmatch(
                manifest["run_name"]
            )
            is None
        ):
            raise AwsLifecycleError(
                "checkpoint manifest identity does not match requested phase"
            )
        timestep = manifest["num_timesteps"]
        model_name = _checkpoint_filename(
            manifest["model"], field="model", suffix=".zip"
        )
        if model_name != f"ckpt_{timestep}_steps.zip":
            raise AwsLifecycleError(
                "checkpoint model filename does not match its timestep"
            )
        artifact_specs = [
            ("model", "sha256", ".zip"),
            ("run_config", "run_config_sha256", ".yaml"),
            ("signature", "signature_sha256", ".json"),
        ]
        normalize_reward = manifest["normalize_reward"]
        if not isinstance(normalize_reward, bool):
            raise AwsLifecycleError(
                "checkpoint normalization identity is invalid"
            )
        if normalize_reward:
            artifact_specs.append(
                ("vecnormalize", "vecnormalize_sha256", ".pkl")
            )
        elif (
            manifest["vecnormalize"] is not None
            or manifest["vecnormalize_sha256"] is not None
        ):
            raise AwsLifecycleError(
                "unnormalized checkpoint names VecNormalize state"
            )
        artifact_specs.append(
            ("budget_ledger", "budget_ledger_sha256", ".json")
        )
        expected_generation_names = {
            "run_config": f"ckpt_run_config_{timestep}_steps.yaml",
            "signature": f"ckpt_signature_{timestep}_steps.json",
            "budget_ledger": (
                f"ckpt_budget_ledger_{timestep}_steps.json"
            ),
        }
        if normalize_reward:
            expected_generation_names["vecnormalize"] = (
                f"ckpt_vecnormalize_{timestep}_steps.pkl"
            )
        if any(
            manifest[field] != expected_name
            for field, expected_name in expected_generation_names.items()
        ):
            raise AwsLifecycleError(
                "checkpoint sidecar filenames do not match the model timestep"
            )
        artifact_paths: dict[str, Path] = {}
        seen_names: set[str] = set()
        for field, hash_field, suffix in artifact_specs:
            filename = _checkpoint_filename(
                manifest[field], field=field, suffix=suffix
            )
            expected_hash = _checkpoint_hash(
                manifest[hash_field], field=field
            )
            if filename in seen_names:
                raise AwsLifecycleError(
                    "checkpoint manifest reuses an artifact filename"
                )
            seen_names.add(filename)
            destination = staging_root / filename
            object_store.download(
                f"{object_prefix}{filename}", destination
            )
            if not destination.is_file() or destination.is_symlink():
                raise AwsLifecycleError(
                    f"checkpoint {field} is not a regular file"
                )
            if training.sha256_file(destination) != expected_hash:
                raise AwsLifecycleError(
                    f"checkpoint {field} SHA-256 mismatch"
                )
            artifact_paths[field] = destination

        try:
            downloaded_config = yaml.safe_load(
                artifact_paths["run_config"].read_text(encoding="utf-8")
            )
            signature = json.loads(
                artifact_paths["signature"].read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, yaml.YAMLError, json.JSONDecodeError) as error:
            raise AwsLifecycleError(
                "checkpoint config or signature is malformed"
            ) from error
        if not isinstance(downloaded_config, dict) or not isinstance(
            signature, dict
        ):
            raise AwsLifecycleError(
                "checkpoint config and signature must be mappings"
            )
        trusted_config_path = (
            Path(repo_dir) / "configs" / "all32.yaml"
        )
        if not trusted_config_path.is_file():
            trusted_config_path = (
                Path(__file__).resolve().parents[1]
                / "configs"
                / "all32.yaml"
            )
        trusted_config = _resolved_all32_config(
            trusted_config_path, phase
        )
        expected_signature = training.checkpoint_resume_signature(
            trusted_config, phase
        )
        if (
            downloaded_config != trusted_config
            or signature != expected_signature
            or training.checkpoint_resume_signature(
                downloaded_config, phase
            )
            != expected_signature
            or manifest["action_set"]
            != expected_signature["environment"]["action_set"]
            or manifest["action_count"]
            != expected_signature["environment"]["action_count"]
            or manifest["extractor"]
            != expected_signature["policy"]["extractor"]
            or manifest["extractor_class"]
            != expected_signature["policy"]["extractor_class"]
            or normalize_reward
            != expected_signature["normalization"]["normalize_reward"]
        ):
            raise AwsLifecycleError(
                "checkpoint policy/environment/normalization signature "
                "does not match the requested phase"
            )
        checkpoint_ledger = _configured_ledger(
            artifact_paths["budget_ledger"], config
        )
        current_ledger = _configured_ledger(Path(ledger_path), config)
        _require_authoritative_ledger_superset(
            checkpoint_ledger, current_ledger
        )
        if normalize_reward:
            _validate_vecnormalize_checkpoint(
                artifact_paths["vecnormalize"], signature
            )
        model = model_validator(
            artifact_paths["model"], trusted_config
        )
        model_timesteps = getattr(model, "num_timesteps", None)
        if (
            isinstance(model_timesteps, bool)
            or not isinstance(model_timesteps, int)
            or model_timesteps != timestep
        ):
            raise AwsLifecycleError(
                "checkpoint model timestep does not match the manifest"
            )
        return ResumeBundle(
            root=staging_root,
            manifest_path=manifest_path,
            model_path=artifact_paths["model"],
            run_config_path=artifact_paths["run_config"],
            signature_path=artifact_paths["signature"],
            vecnormalize_path=artifact_paths.get("vecnormalize"),
            budget_ledger_path=artifact_paths["budget_ledger"],
            manifest=manifest,
        )
    except BaseException:
        shutil.rmtree(staging_root)
        raise


def _configured_object_relative(config: AwsConfig, uri: str) -> str:
    root_bucket, root_key = _s3_location(config.s3_prefix.rstrip("/"))
    object_bucket, object_key = _s3_location(uri)
    prefix = f"{root_key.rstrip('/')}/"
    if object_bucket != root_bucket or not object_key.startswith(prefix):
        raise AwsLifecycleError(
            "phase lineage object is outside the configured S3 prefix"
        )
    relative = object_key.removeprefix(prefix)
    path = Path(relative)
    if (
        not relative
        or path.is_absolute()
        or "\\" in relative
        or "%" in relative
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise AwsLifecycleError("phase lineage object path is unsafe")
    return path.as_posix()


def _read_json_file(path: Path, description: str) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AwsLifecycleError(
            f"{description} is not valid UTF-8 JSON"
        ) from error


def _phase_identity_paths(payload: dict[str, Any]) -> set[str]:
    paths: set[str] = set()
    for field in ("last_candidate", "promoted_best"):
        bundle = payload.get(field)
        if bundle is None:
            continue
        if not isinstance(bundle, dict):
            raise AwsLifecycleError(
                "phase lineage bundle identity is malformed"
            )
        expected = {
            item.name for item in phase_training.BundleIdentity.__dataclass_fields__.values()
        }
        if set(bundle) != expected:
            raise AwsLifecycleError(
                "phase lineage bundle identity has an invalid schema"
            )
        for path_field in (
            "manifest_path",
            "model_path",
            "run_config_path",
            "signature_path",
            "budget_ledger_path",
        ):
            value = bundle.get(path_field)
            if not isinstance(value, str):
                raise AwsLifecycleError(
                    "phase lineage bundle path is invalid"
                )
            paths.add(value)
        vecnormalize = bundle.get("vecnormalize_path")
        if vecnormalize is not None:
            if not isinstance(vecnormalize, str):
                raise AwsLifecycleError(
                    "phase lineage VecNormalize path is invalid"
                )
            paths.add(vecnormalize)
    for field in ("last_diagnostic", "best_report"):
        report = payload.get(field)
        if report is None:
            continue
        if not isinstance(report, dict) or set(report) != {
            "path",
            "sha256",
            "checkpoint_sha256",
        }:
            raise AwsLifecycleError(
                "phase lineage report identity is malformed"
            )
        path = report.get("path")
        if not isinstance(path, str):
            raise AwsLifecycleError(
                "phase lineage report path is invalid"
            )
        paths.add(path)
    for value in paths:
        candidate = Path(value)
        if (
            candidate.is_absolute()
            or "\\" in value
            or "%" in value
            or any(part in {"", ".", ".."} for part in candidate.parts)
        ):
            raise AwsLifecycleError(
                "phase lineage identity path is unsafe"
            )
    return paths


def _default_phase_bundle_validator(
    identity: phase_training.BundleIdentity,
    repository_root: Path,
    *,
    phase: str,
    trusted_repo: Path,
) -> None:
    """Verify dynamic weights while pinning every other trusted setting."""
    manifest_path = repository_root / identity.manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_config_path = repository_root / identity.run_config_path
    signature_path = repository_root / identity.signature_path
    downloaded_config = yaml.safe_load(
        run_config_path.read_text(encoding="utf-8")
    )
    signature = json.loads(
        signature_path.read_text(encoding="utf-8")
    )
    trusted_path = trusted_repo / "configs" / "all32.yaml"
    if not trusted_path.is_file():
        trusted_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "all32.yaml"
        )
    trusted_config = _resolved_all32_config(trusted_path, phase)
    if not isinstance(downloaded_config, dict) or not isinstance(
        signature, dict
    ):
        raise AwsLifecycleError(
            "phase checkpoint config or signature is malformed"
        )
    weights = downloaded_config.get("train", {}).get("level_weights")
    levels = downloaded_config.get("levels")
    if (
        not isinstance(levels, list)
        or not isinstance(weights, dict)
        or (
            bool(weights)
            and (
                set(weights) != set(levels)
                or not all(
                    isinstance(level, str)
                    and not isinstance(weight, bool)
                    and isinstance(weight, (int, float))
                    and math.isfinite(weight)
                    and weight > 0
                    for level, weight in weights.items()
                )
            )
        )
    ):
        raise AwsLifecycleError(
            "phase checkpoint regression weights are invalid"
        )
    downloaded_static = copy.deepcopy(downloaded_config)
    trusted_static = copy.deepcopy(trusted_config)
    downloaded_target = downloaded_static["train"].get(
        "total_timesteps"
    )
    trusted_target = trusted_static["train"].get("total_timesteps")
    if (
        isinstance(downloaded_target, bool)
        or not isinstance(downloaded_target, int)
        or isinstance(trusted_target, bool)
        or not isinstance(trusted_target, int)
        or identity.num_timesteps > downloaded_target
        or downloaded_target > trusted_target
    ):
        raise AwsLifecycleError(
            "phase checkpoint chunk target is outside trusted bounds"
        )
    downloaded_static["train"]["level_weights"] = {}
    trusted_static["train"]["level_weights"] = {}
    downloaded_static["train"]["total_timesteps"] = trusted_target
    expected_signature = training.checkpoint_resume_signature(
        downloaded_config, phase
    )
    if (
        downloaded_static != trusted_static
        or signature != expected_signature
        or manifest.get("phase") != phase
        or manifest.get("action_set")
        != expected_signature["environment"]["action_set"]
        or manifest.get("action_count")
        != expected_signature["environment"]["action_count"]
        or manifest.get("extractor")
        != expected_signature["policy"]["extractor"]
        or manifest.get("extractor_class")
        != expected_signature["policy"]["extractor_class"]
        or manifest.get("normalize_reward")
        != expected_signature["normalization"]["normalize_reward"]
    ):
        raise AwsLifecycleError(
            "phase checkpoint does not match the trusted all-32 identity"
        )
    if identity.vecnormalize_path is not None:
        _validate_vecnormalize_checkpoint(
            repository_root / identity.vecnormalize_path,
            signature,
        )
    model = _default_model_validator(
        repository_root / identity.model_path,
        downloaded_config,
    )
    if getattr(model, "num_timesteps", None) != identity.num_timesteps:
        raise AwsLifecycleError(
            "phase checkpoint model timestep does not match its manifest"
        )


def _materialize_phase_file(
    source: Path, destination: Path, *, replace_existing: bool
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not replace_existing and destination.exists():
        if (
            destination.is_symlink()
            or not destination.is_file()
            or training.sha256_file(destination)
            != training.sha256_file(source)
        ):
            raise AwsLifecycleError(
                f"existing phase artifact differs: {destination}"
            )
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(source.read_bytes())
            output.flush()
            os.fsync(output.fileno())
        if replace_existing:
            os.replace(temporary, destination)
        else:
            try:
                os.link(temporary, destination)
            except FileExistsError as error:
                raise AwsLifecycleError(
                    f"phase artifact appeared during restore: {destination}"
                ) from error
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def restore_phase_lineage(
    *,
    config: AwsConfig,
    phase: str,
    lineage_s3_uri: str,
    repo_dir: Path,
    ledger_path: Path,
    object_store: Any,
    bundle_validator: Callable[..., None] | None = None,
) -> PhaseResumeBundle:
    """Download and verify both phase heads plus incumbent evidence."""
    if phase not in {"phase_1", "phase_2"}:
        raise AwsLifecycleError("phase lineage requires phase_1 or phase_2")
    head_relative = _configured_object_relative(
        config, lineage_s3_uri
    )
    if (
        head_relative
        != f"models/all32-{phase}/latest.json"
    ):
        raise AwsLifecycleError(
            "phase lineage URI does not name the canonical phase head"
        )
    repo_root = Path(repo_dir).resolve()
    staging_parent = repo_root / ".resume"
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=f"{phase}-lineage-", dir=staging_parent)
    )
    mirror = staging_root / "repository"
    downloaded: set[str] = set()

    def download(relative: str) -> Path:
        if relative in downloaded:
            return mirror / relative
        candidate = Path(relative)
        if (
            candidate.is_absolute()
            or "\\" in relative
            or "%" in relative
            or any(part in {"", ".", ".."} for part in candidate.parts)
        ):
            raise AwsLifecycleError(
                "phase lineage identity path is unsafe"
            )
        destination = mirror / candidate
        object_store.download(
            f"{config.s3_prefix}{candidate.as_posix()}",
            destination,
        )
        downloaded.add(relative)
        return destination

    try:
        head_path = download(head_relative)
        head = _read_json_file(head_path, "phase lineage head")
        state_name = head.get("phase_state") if isinstance(head, dict) else None
        if (
            not isinstance(head, dict)
            or set(head) != {
                "schema_version",
                "kind",
                "phase",
                "run_name",
                "phase_state",
                "phase_state_sha256",
            }
            or head.get("schema_version") != 1
            or head.get("kind") != "marioai-phase-lineage"
            or head.get("phase") != phase
            or head.get("run_name") != f"all32-{phase}"
            or not isinstance(state_name, str)
            or _CHECKPOINT_FILENAME_PATTERN.fullmatch(state_name) is None
            or Path(state_name).name != state_name
        ):
            raise AwsLifecycleError(
                "phase lineage head has an invalid schema"
            )
        state_relative = (
            f"{Path(head_relative).parent.as_posix()}/{state_name}"
        )
        state_path = download(state_relative)
        expected_state_hash = _checkpoint_hash(
            head.get("phase_state_sha256"), field="phase state"
        )
        if training.sha256_file(state_path) != expected_state_hash:
            raise AwsLifecycleError("phase lineage state SHA-256 mismatch")
        state_payload = _read_json_file(
            state_path, "phase lineage state"
        )
        if not isinstance(state_payload, dict):
            raise AwsLifecycleError(
                "phase lineage state must be a JSON object"
            )
        identity_paths = _phase_identity_paths(state_payload)
        for relative in sorted(identity_paths):
            download(relative)
        staged_store = phase_training.PhaseLineageStore(
            head_path, repository_root=mirror
        )
        lineage = staged_store.load()
        if (
            lineage.phase != phase
            or lineage.run_name != f"all32-{phase}"
        ):
            raise AwsLifecycleError(
                "phase lineage identity does not match requested phase"
            )
        selected_validator = (
            _default_phase_bundle_validator
            if bundle_validator is None
            else bundle_validator
        )
        unique_bundles = {
            bundle.manifest_sha256: bundle
            for bundle in (
                lineage.last_candidate,
                lineage.promoted_best,
            )
            if bundle is not None
        }
        current_ledger = _configured_ledger(ledger_path, config)
        for bundle in unique_bundles.values():
            selected_validator(
                bundle,
                mirror,
                phase=phase,
                trusted_repo=repo_root,
            )
            checkpoint_ledger = _configured_ledger(
                mirror / bundle.budget_ledger_path, config
            )
            _require_authoritative_ledger_superset(
                checkpoint_ledger, current_ledger
            )

        for relative in sorted(downloaded - {head_relative}):
            _materialize_phase_file(
                mirror / relative,
                repo_root / relative,
                replace_existing=False,
            )
        canonical_head = repo_root / head_relative
        _materialize_phase_file(
            head_path, canonical_head, replace_existing=True
        )
        canonical_store = phase_training.PhaseLineageStore(
            canonical_head, repository_root=repo_root
        )
        canonical_lineage = canonical_store.load()
        return PhaseResumeBundle(
            root=staging_root,
            lineage_path=canonical_head,
            lineage=canonical_lineage,
        )
    except BaseException:
        shutil.rmtree(staging_root)
        raise


def verify_resume_bundle(
    bundle: ResumeBundle | PhaseResumeBundle,
) -> None:
    """Re-authenticate staged bytes immediately before paid mutation."""
    if isinstance(bundle, PhaseResumeBundle):
        restored = phase_training.PhaseLineageStore(
            bundle.lineage_path,
            repository_root=bundle.lineage_path.parents[2],
        ).load()
        if restored != bundle.lineage:
            raise AwsLifecycleError(
                "verified phase lineage changed before launch"
            )
        return
    artifact_fields = [
        ("model", "sha256", bundle.model_path),
        (
            "run_config",
            "run_config_sha256",
            bundle.run_config_path,
        ),
        (
            "signature",
            "signature_sha256",
            bundle.signature_path,
        ),
        (
            "budget_ledger",
            "budget_ledger_sha256",
            bundle.budget_ledger_path,
        ),
    ]
    if bundle.vecnormalize_path is not None:
        artifact_fields.append(
            (
                "vecnormalize",
                "vecnormalize_sha256",
                bundle.vecnormalize_path,
            )
        )
    for name_field, hash_field, path in artifact_fields:
        if (
            bundle.manifest.get(name_field) != path.name
            or not path.is_file()
            or path.is_symlink()
            or training.sha256_file(path)
            != bundle.manifest.get(hash_field)
        ):
            raise AwsLifecycleError(
                f"verified resume {name_field} changed before launch"
            )


@dataclass(frozen=True)
class LaunchReservation:
    """Durable pre-mutation identity and prices for one launch attempt."""

    client_token: str
    state: str
    phase: str
    request: dict[str, Any]
    instance_hourly_usd: Decimal
    on_demand_hourly_usd: Decimal
    volume_hourly_usd: Decimal
    max_hours: Decimal
    grace_hours: Decimal
    requested_epoch_seconds: Decimal
    instance_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _LAUNCH_STATE_SCHEMA_VERSION,
            "client_token": self.client_token,
            "state": self.state,
            "phase": self.phase,
            "request": self.request,
            "instance_hourly_usd": str(self.instance_hourly_usd),
            "on_demand_hourly_usd": str(self.on_demand_hourly_usd),
            "volume_hourly_usd": str(self.volume_hourly_usd),
            "max_hours": str(self.max_hours),
            "grace_hours": str(self.grace_hours),
            "requested_epoch_seconds": str(self.requested_epoch_seconds),
            "instance_id": self.instance_id,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> LaunchReservation:
        expected = {
            "schema_version",
            "client_token",
            "state",
            "phase",
            "request",
            "instance_hourly_usd",
            "on_demand_hourly_usd",
            "volume_hourly_usd",
            "max_hours",
            "grace_hours",
            "requested_epoch_seconds",
            "instance_id",
        }
        if not isinstance(payload, dict) or set(payload) != expected:
            raise AwsLifecycleError("launch reservation has an invalid schema")
        if payload["schema_version"] != _LAUNCH_STATE_SCHEMA_VERSION:
            raise AwsLifecycleError(
                "launch reservation has an unsupported schema version"
            )
        try:
            reservation = cls(
                client_token=payload["client_token"],
                state=payload["state"],
                phase=payload["phase"],
                request=payload["request"],
                instance_hourly_usd=Decimal(
                    payload["instance_hourly_usd"]
                ),
                on_demand_hourly_usd=Decimal(
                    payload["on_demand_hourly_usd"]
                ),
                volume_hourly_usd=Decimal(payload["volume_hourly_usd"]),
                max_hours=Decimal(payload["max_hours"]),
                grace_hours=Decimal(payload["grace_hours"]),
                requested_epoch_seconds=Decimal(
                    payload["requested_epoch_seconds"]
                ),
                instance_id=payload["instance_id"],
            )
        except (ArithmeticError, TypeError, ValueError) as error:
            raise AwsLifecycleError(
                "launch reservation contains invalid decimals"
            ) from error
        reservation._validate()
        return reservation

    def _validate(self) -> None:
        if (
            not isinstance(self.client_token, str)
            or _CLIENT_TOKEN_PATTERN.fullmatch(self.client_token) is None
        ):
            raise AwsLifecycleError(
                "launch reservation has an invalid ClientToken"
            )
        if not isinstance(self.phase, str) or not self.phase:
            raise AwsLifecycleError("launch reservation has an invalid phase")
        if self.state not in {"reserved", "launched"}:
            raise AwsLifecycleError("launch reservation has an invalid state")
        if not isinstance(self.request, dict):
            raise AwsLifecycleError("launch reservation has an invalid request")
        for name, value in (
            ("instance_hourly_usd", self.instance_hourly_usd),
            ("on_demand_hourly_usd", self.on_demand_hourly_usd),
            ("volume_hourly_usd", self.volume_hourly_usd),
            ("max_hours", self.max_hours),
            ("grace_hours", self.grace_hours),
            ("requested_epoch_seconds", self.requested_epoch_seconds),
        ):
            if not value.is_finite() or value < 0:
                raise AwsLifecycleError(
                    f"launch reservation has invalid {name}"
                )
        if self.max_hours <= 0:
            raise AwsLifecycleError(
                "launch reservation max_hours must be positive"
            )
        if self.instance_id is not None and (
            not isinstance(self.instance_id, str)
            or _INSTANCE_ID_PATTERN.fullmatch(self.instance_id) is None
        ):
            raise AwsLifecycleError(
                "launch reservation has an invalid instance ID"
            )
        if (self.state == "reserved") != (self.instance_id is None):
            raise AwsLifecycleError(
                "launch reservation state and instance ID disagree"
            )


class LaunchStateStore:
    """Serialize lifecycle access and atomically persist a launch reservation."""

    def __init__(
        self,
        ledger_path: Path,
        *,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self.ledger_path = Path(ledger_path)
        self.state_path = self.ledger_path.with_name(
            f"{self.ledger_path.name}.launch.json"
        )
        self.lock_path = self.ledger_path.with_name(
            f"{self.ledger_path.name}.lock"
        )
        self.wall_clock = wall_clock
        self._lock_file: Any = None

    def __enter__(self) -> LaunchStateStore:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_file = self.lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(
                self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
            )
        except BlockingIOError as error:
            self._lock_file.close()
            self._lock_file = None
            raise AwsLifecycleError(
                f"another AWS lifecycle owns ledger {self.ledger_path}"
            ) from error
        return self

    def __exit__(self, *_args: Any) -> None:
        if self._lock_file is not None:
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
            self._lock_file.close()
            self._lock_file = None

    def now_epoch_seconds(self) -> Decimal:
        value = self.wall_clock()
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise AwsLifecycleError("wall clock returned an invalid value")
        result = Decimal(str(value))
        if not result.is_finite() or result < 0:
            raise AwsLifecycleError("wall clock returned an invalid value")
        return result

    def load(self) -> LaunchReservation | None:
        if not self.state_path.exists():
            return None
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise AwsLifecycleError(
                f"cannot read launch reservation {self.state_path}"
            ) from error
        return LaunchReservation.from_dict(payload)

    def save(self, reservation: LaunchReservation) -> None:
        if not isinstance(reservation, LaunchReservation):
            raise ValueError("reservation must be a LaunchReservation")
        reservation._validate()
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=self.state_path.parent,
            delete=False,
        ) as temporary:
            json.dump(reservation.to_dict(), temporary, sort_keys=True)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        try:
            os.replace(temporary_path, self.state_path)
            directory_fd = os.open(self.state_path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise

    def clear(self) -> None:
        self.state_path.unlink(missing_ok=True)
        directory_fd = os.open(self.state_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


@dataclass(frozen=True)
class LaunchedInstance:
    """The immutable prices and identity used to account for one instance."""

    phase: str
    instance_id: str
    instance_type: str
    availability_zone: str
    subnet_id: str
    ami_id: str
    public_ip: str | None
    spot_hourly_usd: Decimal
    volume_hourly_usd: Decimal
    max_hours: Decimal
    launched_monotonic: Decimal


class AwsCommandAdapter:
    """Separate command boundary for the three Task 3 control operations."""

    def __init__(
        self,
        *,
        config: AwsConfig,
        readonly: AwsCli,
        runner: Callable[..., Any] = subprocess.run,
    ) -> None:
        if not isinstance(config, AwsConfig):
            raise ValueError("config must be an AwsConfig")
        profile = getattr(readonly, "profile", None)
        region = getattr(readonly, "region", None)
        if not isinstance(profile, str) or not profile:
            raise ValueError("readonly adapter must expose a profile")
        if not isinstance(region, str) or not region:
            raise ValueError("readonly adapter must expose a region")
        self.readonly = readonly
        self.config = config
        self.profile = profile
        self.region = region
        self._runner = runner
        self._authorized_subnet_azs: set[tuple[str, str]] = set()
        self._resolved_ami_id: str | None = None
        self._authorized_termination_ids: set[str] = set()
        self._authorized_instance_profile: dict[str, str] | None = None

    def preflight(self, config: AwsConfig) -> PreflightResult:
        """Delegate only to the public read-only boundary."""
        self._authorized_subnet_azs.clear()
        self._authorized_instance_profile = None
        result = self.readonly.preflight(config)
        if not isinstance(result, PreflightResult):
            raise AwsLifecycleError("preflight returned an invalid result")
        if config != self.config:
            raise AwsLifecycleError(
                "preflight configuration does not match lifecycle adapter"
            )
        self._authorized_instance_profile = _preflight_profile_identity(result)
        self._authorized_subnet_azs.update(result.subnet_azs)
        return result

    def latest_spot_prices(
        self, instance_types: Any
    ) -> tuple[SpotOffer, ...]:
        """Delegate current offer lookup to the authorized read-only boundary."""
        return self.readonly.latest_spot_prices(instance_types)

    def verify_account(self, config: AwsConfig) -> None:
        """Verify the exact account before status or emergency mutation."""
        if self.profile != config.profile or self.region != config.region:
            raise AwsLifecycleError(
                "AWS CLI profile/region does not match configuration"
            )
        payload = self.readonly.run(["sts", "get-caller-identity"])
        if not isinstance(payload, dict) or payload.get(
            "Account"
        ) != config.account_id:
            raise AwsLifecycleError(
                f"expected AWS account {config.account_id} before lifecycle action"
            )

    def resolve_ami(self, parameter_name: str) -> str:
        """Resolve only the configured public SSM AMI parameter."""
        if parameter_name != self.config.ami_ssm_parameter or not (
            isinstance(parameter_name, str)
            and parameter_name.startswith("/aws/service/")
        ):
            raise AwsLifecycleError(
                "AMI parameter must be an AWS public SSM parameter"
            )
        payload = self._invoke(
            [
                "ssm",
                "get-parameter",
                "--name",
                parameter_name,
                "--query",
                "Parameter.Value",
            ]
        )
        if (
            not isinstance(payload, str)
            or _AMI_ID_PATTERN.fullmatch(payload) is None
        ):
            raise AwsLifecycleError("SSM get-parameter returned an invalid AMI")
        self._resolved_ami_id = payload
        return payload

    def run_instances(
        self, request: Any, *, max_hours: Decimal
    ) -> dict[str, Any]:
        """Perform one validated, idempotent one-time Spot request."""
        if (
            not isinstance(max_hours, Decimal)
            or not max_hours.is_finite()
            or max_hours <= 0
        ):
            raise AwsLifecycleError(
                "run-instances requires positive finite max_hours"
            )
        _validate_launch_request(
            request,
            config=self.config,
            authorized_subnet_azs=self._authorized_subnet_azs,
            resolved_ami_id=self._resolved_ami_id,
            max_hours=max_hours,
        )
        request_json = json.dumps(
            request, sort_keys=True, separators=(",", ":")
        )
        try:
            payload = self._invoke(
                [
                    "ec2",
                    "run-instances",
                    "--cli-input-json",
                    request_json,
                ]
            )
        except subprocess.TimeoutExpired as error:
            raise AwsLifecycleError(
                "AWS ec2 run-instances timed out with an ambiguous result; "
                f"reconcile using ClientToken {request['ClientToken']!r} "
                "and do not submit a new token"
            ) from error
        except AwsLifecycleError as error:
            cause = error.__cause__
            stderr = (
                (cause.stderr or "").strip()
                if isinstance(cause, subprocess.CalledProcessError)
                else ""
            )
            if re.search(
                r"An error occurred \(InsufficientInstanceCapacity\) "
                r"when calling the RunInstances operation:",
                stderr,
            ):
                raise AwsCapacityUnavailable(
                    "EC2 definitively rejected run-instances before creation: "
                    "InsufficientInstanceCapacity"
                ) from error
            raise
        if not isinstance(payload, dict):
            raise AwsLifecycleError(
                "AWS ec2 run-instances returned a non-object response"
            )
        try:
            instance = _launched_instance_payload(
                payload,
                request=request,
                expected_instance_profile=self._authorized_instance_profile,
            )
        except AwsLifecycleError as error:
            raise AwsLifecycleError(
                f"{error}; reconcile using ClientToken "
                f"{request['ClientToken']!r}"
            ) from error
        self._authorized_termination_ids.add(instance["InstanceId"])
        return payload

    def terminate_instance(self, instance_id: str) -> None:
        """Idempotently request termination of one validated EC2 instance ID."""
        if (
            not isinstance(instance_id, str)
            or _INSTANCE_ID_PATTERN.fullmatch(instance_id) is None
        ):
            raise AwsLifecycleError("invalid EC2 instance ID")
        if instance_id not in self._authorized_termination_ids:
            raise AwsLifecycleError(
                "EC2 instance is not an authorized MarioAI-All32 target"
            )
        payload = self._invoke(
            [
                "ec2",
                "terminate-instances",
                "--instance-ids",
                instance_id,
            ]
        )
        if not isinstance(payload, dict):
            raise AwsLifecycleError(
                "AWS ec2 terminate-instances returned a non-object response"
            )
        _validate_termination_response(payload, instance_id)

    def project_instances(
        self,
        *,
        instance_id: str | None = None,
        client_token: str | None = None,
        active_only: bool = False,
        timeout_seconds: int | float | None = None,
    ) -> tuple[dict[str, str | None], ...]:
        """Return strict project instance summaries through the read-only CLI."""
        if instance_id is not None and (
            not isinstance(instance_id, str)
            or _INSTANCE_ID_PATTERN.fullmatch(instance_id) is None
        ):
            raise AwsLifecycleError("invalid EC2 instance ID")
        if client_token is not None and (
            not isinstance(client_token, str)
            or _CLIENT_TOKEN_PATTERN.fullmatch(client_token) is None
        ):
            raise AwsLifecycleError("invalid EC2 ClientToken")
        if instance_id is not None and client_token is not None:
            raise AwsLifecycleError(
                "instance ID and ClientToken filters are mutually exclusive"
            )
        args = ["ec2", "describe-instances"]
        if instance_id is not None:
            args.extend(["--instance-ids", instance_id])
        args.extend(
            [
                "--filters",
                "Name=tag:Project,Values=MarioAI-All32",
            ]
        )
        if client_token is not None:
            args.append(f"Name=client-token,Values={client_token}")
        if instance_id is None:
            if active_only:
                args.append(
                    "Name=instance-state-name,"
                    "Values=pending,running,stopping,stopped,shutting-down"
                )
        payload = self.readonly.run(args, timeout_seconds=timeout_seconds)
        summaries = _instance_summaries(payload)
        self._authorized_termination_ids.update(
            summary["instance_id"]
            for summary in summaries
            if isinstance(summary["instance_id"], str)
        )
        return summaries

    def _invoke(self, args: list[str]) -> dict | list | str:
        command = [
            "aws",
            *args,
            "--profile",
            self.profile,
            "--region",
            self.region,
            "--output",
            "json",
        ]
        try:
            completed = self._runner(
                command,
                check=True,
                text=True,
                capture_output=True,
                timeout=_AWS_COMMAND_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            raise
        except subprocess.CalledProcessError as error:
            stderr = (error.stderr or "").strip()
            detail = f": {stderr}" if stderr else ""
            raise AwsLifecycleError(
                f"AWS {' '.join(args[:2])} failed{detail}"
            ) from error
        except OSError as error:
            raise AwsLifecycleError(
                f"could not execute AWS CLI for {' '.join(args[:2])}: {error}"
            ) from error
        stdout = completed.stdout
        if not isinstance(stdout, str):
            raise AwsLifecycleError(
                f"AWS {' '.join(args[:2])} returned non-text output"
            )
        if not stdout.strip():
            return ""
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError as error:
            raise AwsLifecycleError(
                f"AWS {' '.join(args[:2])} returned invalid JSON"
            ) from error
        if not isinstance(payload, (dict, list, str)):
            raise AwsLifecycleError(
                f"AWS {' '.join(args[:2])} returned unsupported JSON"
            )
        return payload


@dataclass(frozen=True)
class RemotePoll:
    """One successful remote/instance liveness observation."""

    running: bool
    exit_code: int | None = None


class SshRemoteSupervisor:
    """Upload the repository and start the independent remote shell guard."""

    def __init__(
        self,
        *,
        aws: Any,
        ssh_key: Path,
        local_repo: Path,
        runner: Callable[..., Any] = subprocess.run,
        remote_repo: str = "/home/ubuntu/MarioAI",
        user: str = "ubuntu",
        monotonic: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
        startup_confirmation_seconds: int = 10,
        process_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.aws = aws
        self.ssh_key = Path(ssh_key)
        self.local_repo = Path(local_repo)
        self._runner = runner
        self._process_factory = process_factory or subprocess.Popen
        self._poll_subprocess = (
            process_factory is not None or runner is subprocess.run
        )
        self._monotonic = monotonic
        self._sleeper = sleeper
        self._startup_confirmation_seconds = startup_confirmation_seconds
        self._hosts: dict[str, str] = {}
        if not self.ssh_key.is_file():
            raise AwsLifecycleError(f"SSH key does not exist: {self.ssh_key}")
        if not self.local_repo.is_dir():
            raise AwsLifecycleError(
                f"local repository does not exist: {self.local_repo}"
            )
        if re.fullmatch(r"/[A-Za-z0-9_./-]+", remote_repo) is None:
            raise AwsLifecycleError("remote repository path is unsafe")
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", user) is None:
            raise AwsLifecycleError("SSH user is unsafe")
        if (
            isinstance(startup_confirmation_seconds, bool)
            or not isinstance(startup_confirmation_seconds, int)
            or startup_confirmation_seconds <= 0
        ):
            raise AwsLifecycleError(
                "startup confirmation seconds must be positive"
            )
        self.remote_repo = remote_repo.rstrip("/")
        self.user = user

    def start(
        self,
        instance: LaunchedInstance,
        *,
        phase: str,
        max_seconds: int,
        s3_prefix: str,
        train_args: tuple[str, ...] = (),
        on_tick: Callable[[], None] | None = None,
        absolute_deadline: Decimal | None = None,
    ) -> None:
        """Rsync source and launch cloud_train.sh under a recorded remote PID."""
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", phase) is None:
            raise AwsLifecycleError("phase is unsafe for remote execution")
        if (
            isinstance(max_seconds, bool)
            or not isinstance(max_seconds, int)
            or max_seconds <= 0
        ):
            raise AwsLifecycleError("max_seconds must be a positive integer")
        if (
            not isinstance(s3_prefix, str)
            or re.fullmatch(r"s3://[^/]+/.+/", s3_prefix) is None
        ):
            raise AwsLifecycleError("S3 prefix is invalid")
        if (
            not isinstance(train_args, tuple)
            or not all(
                isinstance(argument, str)
                and argument
                and "\0" not in argument
                for argument in train_args
            )
        ):
            raise AwsLifecycleError(
                "training arguments must be a tuple of non-empty strings"
            )
        target = self._prepare_repository(
            instance,
            on_tick=on_tick,
            absolute_deadline=absolute_deadline,
        )
        elapsed_seconds = int(
            max(
                Decimal("0"),
                _monotonic_decimal(self._monotonic())
                - instance.launched_monotonic,
            ).to_integral_value(rounding=ROUND_CEILING)
        )
        remaining_seconds = max_seconds - elapsed_seconds
        if remaining_seconds <= 0:
            raise AwsLifecycleError(
                "maximum paid runtime expired during remote bootstrap"
            )
        startup_script = (
            "set -eu\n"
            "repo=$1\n"
            "shift\n"
            'cd "$repo"\n'
            "pidfile=/tmp/marioai-cloud-train.pid\n"
            "logfile=/tmp/marioai-cloud-train.log\n"
            'rm -f "$pidfile"\n'
            "nohup ./scripts/cloud_train.sh \"$@\" "
            "</dev/null >\"$logfile\" 2>&1 &\n"
            "pid=$!\n"
            'tmp_pidfile="${pidfile}.$$"\n'
            "printf '%s\\n' \"$pid\" >\"$tmp_pidfile\"\n"
            'mv "$tmp_pidfile" "$pidfile"\n'
            f"remaining={self._startup_confirmation_seconds}\n"
            'while [ "$remaining" -gt 0 ]; do\n'
            "  sleep 1\n"
            '  if ! kill -0 "$pid" 2>/dev/null; then\n'
            "    set +e\n"
            '    wait "$pid"\n'
            "    status=$?\n"
            "    set -e\n"
            '    tail -n 40 "$logfile" >&2 || true\n'
            '    if [ "$status" -eq 0 ]; then exit 70; fi\n'
            '    exit "$status"\n'
            "  fi\n"
            '  remaining=$((remaining - 1))\n'
            "done\n"
        )
        remote_args = [
            self.remote_repo,
            phase,
            str(remaining_seconds),
            self.remote_repo,
            s3_prefix,
            *train_args,
        ]
        self._run_remote_command(
            [
                *self._ssh_command(target),
                "bash",
                "-s",
                "--",
                *[shlex.quote(argument) for argument in remote_args],
            ],
            timeout=max(30, self._startup_confirmation_seconds + 20),
            input_text=startup_script,
            on_tick=on_tick,
            absolute_deadline=absolute_deadline,
        )

    def benchmark(
        self,
        instance: LaunchedInstance,
        *,
        environment_steps: int,
        max_seconds: int,
        on_tick: Callable[[], None] | None = None,
        absolute_deadline: Decimal | None = None,
    ) -> BenchmarkObservation:
        """Prepare a fresh host and parse one exact fixed-step measurement."""
        if (
            isinstance(environment_steps, bool)
            or not isinstance(environment_steps, int)
            or environment_steps <= 0
        ):
            raise AwsLifecycleError(
                "benchmark environment_steps must be a positive integer"
            )
        if (
            isinstance(max_seconds, bool)
            or not isinstance(max_seconds, int)
            or max_seconds <= 0
        ):
            raise AwsLifecycleError(
                "benchmark max_seconds must be a positive integer"
            )
        target = self._prepare_repository(
            instance,
            on_tick=on_tick,
            absolute_deadline=absolute_deadline,
        )
        elapsed_seconds = int(
            max(
                Decimal("0"),
                _monotonic_decimal(self._monotonic())
                - instance.launched_monotonic,
            ).to_integral_value(rounding=ROUND_CEILING)
        )
        remaining_seconds = max_seconds - elapsed_seconds
        if remaining_seconds <= 0:
            raise AwsLifecycleError(
                "maximum paid runtime expired during benchmark bootstrap"
            )
        benchmark_script = """\
set -eu
repo=$1
steps=$2
cd "$repo"
exec .venv/bin/python scripts/train_phase.py benchmark \
  --environment-steps "$steps"
"""
        stdout = self._run_remote_command(
            [
                *self._ssh_command(target),
                "bash",
                "-s",
                "--",
                shlex.quote(self.remote_repo),
                str(environment_steps),
            ],
            timeout=remaining_seconds,
            input_text=benchmark_script,
            on_tick=on_tick,
            absolute_deadline=absolute_deadline,
        )
        try:
            payload = json.loads(stdout)
            if not isinstance(payload, dict) or set(payload) != {
                "environment_steps",
                "elapsed_seconds",
                "peak_rss_gb",
            }:
                raise ValueError("invalid benchmark schema")
            elapsed = payload["elapsed_seconds"]
            if not isinstance(elapsed, str):
                raise ValueError(
                    "benchmark elapsed_seconds must be a JSON string"
                )
            observation = BenchmarkObservation(
                environment_steps=payload["environment_steps"],
                elapsed_seconds=Decimal(elapsed),
                peak_rss_gb=payload["peak_rss_gb"],
            )
        except (
            ArithmeticError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            raise AwsLifecycleError(
                "remote benchmark returned invalid JSON evidence"
            ) from error
        if observation.environment_steps != environment_steps:
            raise AwsLifecycleError(
                "remote benchmark did not run the requested environment steps"
            )
        return observation

    def _prepare_repository(
        self,
        instance: LaunchedInstance,
        *,
        on_tick: Callable[[], None] | None,
        absolute_deadline: Decimal | None,
    ) -> str:
        host = _validated_public_ip(instance.public_ip)
        self._hosts[instance.instance_id] = host
        target = f"{self.user}@{host}"
        self._wait_for_ssh(
            target,
            on_tick=on_tick,
            absolute_deadline=absolute_deadline,
        )
        os_bootstrap = """\
set -eu
. /etc/os-release
[ "$ID" = ubuntu ]
[ "$VERSION_ID" = 24.04 ]
sudo env DEBIAN_FRONTEND=noninteractive apt-get update
sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y \
  --no-install-recommends ca-certificates curl libgl1 unzip rsync python3 \
  python3-venv
if ! command -v aws >/dev/null 2>&1; then
  work_dir="$(mktemp -d)"
  trap 'rm -rf "$work_dir"' EXIT
  curl --fail --location --silent --show-error \
    https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip \
    --output "$work_dir/awscliv2.zip"
  unzip -q "$work_dir/awscliv2.zip" -d "$work_dir"
  sudo "$work_dir/aws/install" --install-dir /usr/local/aws-cli \
    --bin-dir /usr/local/bin
fi
command -v rsync
aws --version
"""
        self._run_remote_command(
            [*self._ssh_command(target), "bash", "-s"],
            timeout=900,
            input_text=os_bootstrap,
            on_tick=on_tick,
            absolute_deadline=absolute_deadline,
        )
        ssh_transport = (
            f"ssh -i {shlex.quote(str(self.ssh_key))} "
            "-o BatchMode=yes -o StrictHostKeyChecking=accept-new "
            "-o ConnectTimeout=10 -o ServerAliveInterval=15 "
            "-o ServerAliveCountMax=2"
        )
        self._run_remote_command(
            [
                "rsync",
                "--archive",
                "--compress",
                "--exclude",
                ".git",
                "--exclude",
                ".venv",
                "--exclude",
                "__pycache__",
                "--rsh",
                ssh_transport,
                f"{self.local_repo}/",
                f"{target}:{self.remote_repo}/",
            ],
            timeout=300,
            on_tick=on_tick,
            absolute_deadline=absolute_deadline,
        )
        dependency_bootstrap = """\
set -eu
repo=$1
bootstrap="$HOME/.marioai-bootstrap"
python3 -m venv "$bootstrap"
"$bootstrap/bin/python" -m pip install \
  --disable-pip-version-check --upgrade pip uv
"$bootstrap/bin/uv" python install 3.13
"$bootstrap/bin/uv" venv --clear --python 3.13 "$repo/.venv"
"$bootstrap/bin/uv" pip install --python "$repo/.venv/bin/python" \
  -r "$repo/requirements.txt" --editable "$repo"
mkdir -p "$repo/models" "$repo/reports"
cd "$repo"
"$repo/.venv/bin/python" -c \
  'import marioai, torch; import scripts.train_phase'
aws sts get-caller-identity --output json >/dev/null
"""
        self._run_remote_command(
            [
                *self._ssh_command(target),
                "bash",
                "-s",
                "--",
                shlex.quote(self.remote_repo),
            ],
            timeout=1800,
            input_text=dependency_bootstrap,
            on_tick=on_tick,
            absolute_deadline=absolute_deadline,
        )
        return target

    def poll(self, instance: LaunchedInstance) -> RemotePoll:
        summaries = self.aws.project_instances(
            instance_id=instance.instance_id
        )
        if not summaries:
            return RemotePoll(running=False, exit_code=0)
        state = summaries[0]["state"]
        return RemotePoll(
            running=state in {"pending", "running"},
            exit_code=None if state in {"pending", "running"} else 0,
        )

    def request_shutdown(
        self,
        instance: LaunchedInstance,
        *,
        on_tick: Callable[[], None] | None = None,
        absolute_deadline: Decimal | None = None,
    ) -> None:
        host = self._hosts.get(instance.instance_id)
        if host is None:
            host = _validated_public_ip(instance.public_ip)
        target = f"{self.user}@{host}"
        shutdown_script = (
            "set -eu\n"
            "if [ -r /tmp/marioai-cloud-train.pid ]; then\n"
            "  pid=\"$(cat /tmp/marioai-cloud-train.pid)\"\n"
            "  kill -TERM \"$pid\"\n"
            "  remaining=180\n"
            "  while kill -0 \"$pid\" 2>/dev/null; do\n"
            "    if [ \"$remaining\" -le 0 ]; then exit 124; fi\n"
            "    sleep 5\n"
            "    remaining=$((remaining - 1))\n"
            "  done\n"
            "else\n"
            "  sudo shutdown -h now\n"
            "fi\n"
        )
        self._run_remote_command(
            [
                *self._ssh_command(target),
                "bash",
                "-s",
            ],
            timeout=900,
            input_text=shutdown_script,
            on_tick=on_tick,
            absolute_deadline=absolute_deadline,
        )

    def _ssh_command(self, target: str) -> list[str]:
        return [
            "ssh",
            "-i",
            str(self.ssh_key),
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=2",
            target,
        ]

    def _wait_for_ssh(
        self,
        target: str,
        *,
        on_tick: Callable[[], None] | None = None,
        absolute_deadline: Decimal | None = None,
    ) -> None:
        deadline = (
            _monotonic_decimal(self._monotonic())
            + Decimal(_SSH_READY_TIMEOUT_SECONDS)
        )
        if absolute_deadline is not None:
            deadline = min(deadline, absolute_deadline)
        last_detail = ""
        while True:
            try:
                self._runner(
                    [*self._ssh_command(target), "bash", "-s"],
                    check=True,
                    text=True,
                    capture_output=True,
                    timeout=15,
                    input="exit 0\n",
                )
                if on_tick is not None:
                    on_tick()
                return
            except subprocess.CalledProcessError as error:
                last_detail = (error.stderr or "").strip()
            except subprocess.TimeoutExpired:
                last_detail = "SSH attempt timed out"
            except OSError as error:
                last_detail = str(error)
            if on_tick is not None:
                on_tick()
            remaining = deadline - _monotonic_decimal(self._monotonic())
            if remaining <= 0:
                suffix = f": {last_detail}" if last_detail else ""
                raise AwsLifecycleError(
                    "SSH did not become ready within 300 seconds" + suffix
                )
            self._sleeper(
                float(min(Decimal(_SSH_READY_POLL_SECONDS), remaining))
            )

    def _run_remote_command(
        self,
        command: list[str],
        *,
        timeout: int,
        input_text: str | None = None,
        on_tick: Callable[[], None] | None = None,
        absolute_deadline: Decimal | None = None,
    ) -> str:
        if self._poll_subprocess:
            return self._run_polled_process(
                command,
                timeout=timeout,
                input_text=input_text,
                on_tick=on_tick,
                absolute_deadline=absolute_deadline,
            )
        kwargs: dict[str, Any] = {
            "check": True,
            "text": True,
            "capture_output": True,
            "timeout": timeout,
        }
        if input_text is not None:
            kwargs["input"] = input_text
        try:
            completed = self._runner(command, **kwargs)
        except subprocess.TimeoutExpired as error:
            raise AwsLifecycleError(
                f"{command[0]} timed out after {timeout} seconds"
            ) from error
        except subprocess.CalledProcessError as error:
            stderr = (error.stderr or "").strip()
            detail = f": {stderr}" if stderr else ""
            raise AwsLifecycleError(f"{command[0]} failed{detail}") from error
        except OSError as error:
            raise AwsLifecycleError(
                f"could not execute {command[0]}: {error}"
            ) from error
        if on_tick is not None:
            on_tick()
        stdout = completed.stdout
        if not isinstance(stdout, str):
            raise AwsLifecycleError(
                f"{command[0]} returned non-text output"
            )
        return stdout

    def _run_polled_process(
        self,
        command: list[str],
        *,
        timeout: int,
        input_text: str | None,
        on_tick: Callable[[], None] | None,
        absolute_deadline: Decimal | None,
    ) -> str:
        started = _monotonic_decimal(self._monotonic())
        deadline = started + Decimal(timeout)
        if absolute_deadline is not None:
            deadline = min(deadline, absolute_deadline)
        process = self._process_factory(
            command,
            stdin=subprocess.PIPE if input_text is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        pending_input = input_text
        try:
            while True:
                remaining = deadline - _monotonic_decimal(self._monotonic())
                if remaining <= 0:
                    if on_tick is not None:
                        on_tick()
                    raise AwsLifecycleError(
                        f"{command[0]} exceeded the absolute paid deadline"
                    )
                poll_timeout = float(min(Decimal("60"), remaining))
                try:
                    stdout, stderr = process.communicate(
                        input=pending_input,
                        timeout=poll_timeout,
                    )
                except subprocess.TimeoutExpired:
                    pending_input = None
                    if on_tick is not None:
                        on_tick()
                    continue
                if on_tick is not None:
                    on_tick()
                if process.returncode != 0:
                    detail_text = (stderr or "").strip()
                    detail = f": {detail_text}" if detail_text else ""
                    raise AwsLifecycleError(f"{command[0]} failed{detail}")
                if not isinstance(stdout, str):
                    raise AwsLifecycleError(
                        f"{command[0]} returned non-text output"
                    )
                return stdout
        finally:
            if process.returncode is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


class AwsOrchestrator:
    """Coordinate guarded launch, accounting, and unconditional termination."""

    def __init__(
        self,
        *,
        config: AwsConfig,
        aws: Any,
        ledger: BudgetLedger,
        remote: Any,
        monotonic: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
        client_token_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
        reservation_store: LaunchStateStore | None = None,
    ) -> None:
        if not isinstance(config, AwsConfig):
            raise ValueError("config must be an AwsConfig")
        if not isinstance(ledger, BudgetLedger):
            raise ValueError("ledger must be a BudgetLedger")
        self.config = config
        self.aws = aws
        self.ledger = ledger
        self.remote = remote
        self._monotonic = monotonic
        self._sleeper = sleeper
        self._client_token_factory = client_token_factory
        self.reservation_store = reservation_store
        self._preflight_result: PreflightResult | None = None
        self._preflight_authorized_at: Decimal | None = None

    def preflight(self) -> PreflightResult:
        """Run the exact read-only preflight before authorizing this process."""
        self._preflight_result = None
        self._preflight_authorized_at = None
        result = self.aws.preflight(self.config)
        if not isinstance(result, PreflightResult):
            raise AwsLifecycleError("preflight returned an invalid result")
        self._preflight_result = result
        self._preflight_authorized_at = _monotonic_decimal(self._monotonic())
        return result

    @property
    def volume_hourly_usd(self) -> Decimal:
        """Return a conservative 30-day-month gp3 hourly rate."""
        return (
            self.config.gp3_monthly_usd_per_gb
            * Decimal(self.config.root_volume_gb)
            / _HOURS_PER_BILLING_MONTH
        )

    def launch_guarded_instance(
        self,
        phase: str,
        max_hours: Decimal,
        *,
        before_mutation: Callable[[], None] | None = None,
        instance_type: str | None = None,
    ) -> LaunchedInstance:
        """Launch one allowed Spot instance after the conservative budget gate."""
        if (
            isinstance(self.aws, AwsCommandAdapter)
            and self.reservation_store is None
        ):
            raise AwsLifecycleError(
                "real AWS launch requires a locked durable reservation store"
            )
        self._require_fresh_preflight()
        if phase not in self.config.allocations:
            raise ValueError(f"phase {phase!r} is not configured")
        if (
            instance_type is not None
            and instance_type not in self.config.instance_types
        ):
            raise AwsLifecycleError(
                "benchmark instance type is not a configured candidate"
            )
        if (
            not isinstance(max_hours, Decimal)
            or not max_hours.is_finite()
            or max_hours <= 0
        ):
            raise ValueError("max_hours must be a positive finite Decimal")

        offers = self.aws.latest_spot_prices(self.config.instance_types)
        if (
            not isinstance(offers, tuple)
            or not offers
            or not all(isinstance(offer, SpotOffer) for offer in offers)
        ):
            raise AwsLifecycleError("no valid allowed Spot offer is available")
        candidate_offers = tuple(
            offer
            for offer in offers
            if instance_type is None
            or offer.instance_type == instance_type
        )
        if not candidate_offers:
            raise AwsLifecycleError(
                "no valid allowed Spot offer is available for the "
                "configured candidate"
            )
        ordered_offers = tuple(
            sorted(
                candidate_offers,
                key=lambda item: (
                    item.hourly_usd,
                    item.instance_type,
                    item.availability_zone,
                    item.subnet_id,
                ),
            )
        )
        selected_instance_type = ordered_offers[0].instance_type
        ordered_offers = tuple(
            offer
            for offer in ordered_offers
            if offer.instance_type == selected_instance_type
        )
        if any(
            offer.instance_type not in self.config.instance_types
            for offer in ordered_offers
        ):
            raise AwsLifecycleError(
                "Spot offer is outside the preflight-authorized configuration"
            )
        on_demand_hourly = self.config.on_demand_ceiling_usd[
            selected_instance_type
        ]
        grace_hours = Decimal(self.config.grace_minutes) / Decimal("60")
        reserve_usd = (
            on_demand_hourly + self.volume_hourly_usd
        ) * grace_hours
        self.ledger.require_launch(
            phase,
            on_demand_hourly,
            self.volume_hourly_usd,
            max_hours,
            reserve_usd,
        )

        ami_id = self.aws.resolve_ami(self.config.ami_ssm_parameter)
        if (
            not isinstance(ami_id, str)
            or _AMI_ID_PATTERN.fullmatch(ami_id) is None
        ):
            raise AwsLifecycleError(
                "SSM returned an invalid AMI ID; launch was not attempted"
            )
        self._require_fresh_preflight()
        tags = [
            {"Key": "Project", "Value": "MarioAI-All32"},
            {"Key": "Phase", "Value": phase},
        ]
        last_capacity_error: AwsCapacityUnavailable | None = None
        for offer in ordered_offers:
            client_token = self._client_token_factory()
            if (
                not isinstance(client_token, str)
                or _CLIENT_TOKEN_PATTERN.fullmatch(client_token) is None
            ):
                raise AwsLifecycleError(
                    "client token must contain 1-64 safe characters"
                )
            request = {
                "ImageId": ami_id,
                "InstanceType": offer.instance_type,
                "MinCount": 1,
                "MaxCount": 1,
                "ClientToken": client_token,
                "UserData": _shutdown_user_data(max_hours),
                "KeyName": self.config.key_name,
                "IamInstanceProfile": {"Name": self.config.instance_profile},
                "Placement": {"AvailabilityZone": offer.availability_zone},
                "InstanceMarketOptions": {
                    "MarketType": "spot",
                    "SpotOptions": {"SpotInstanceType": "one-time"},
                },
                "InstanceInitiatedShutdownBehavior": "terminate",
                "NetworkInterfaces": [
                    {
                        "AssociatePublicIpAddress": True,
                        "DeleteOnTermination": True,
                        "DeviceIndex": 0,
                        "Groups": [self.config.security_group_id],
                        "SubnetId": offer.subnet_id,
                    }
                ],
                "BlockDeviceMappings": [
                    {
                        "DeviceName": "/dev/sda1",
                        "Ebs": {
                            "DeleteOnTermination": True,
                            "Encrypted": True,
                            "VolumeSize": self.config.root_volume_gb,
                            "VolumeType": "gp3",
                        },
                    }
                ],
                "TagSpecifications": [
                    {"ResourceType": "instance", "Tags": tags},
                    {"ResourceType": "volume", "Tags": tags},
                ],
            }
            final_preflight = self.preflight()
            if (
                offer.subnet_id,
                offer.availability_zone,
            ) not in set(final_preflight.subnet_azs):
                raise AwsLifecycleError(
                    "selected Spot offer is outside the final preflight"
                )
            if self.aws.project_instances(active_only=True):
                raise AwsLifecycleError(
                    "active MarioAI-All32 instance appeared before mutation"
                )
            self._require_fresh_preflight()
            if before_mutation is not None:
                before_mutation()
            reservation = LaunchReservation(
                client_token=client_token,
                state="reserved",
                phase=phase,
                request=request,
                instance_hourly_usd=offer.hourly_usd,
                on_demand_hourly_usd=on_demand_hourly,
                volume_hourly_usd=self.volume_hourly_usd,
                max_hours=max_hours,
                grace_hours=grace_hours,
                requested_epoch_seconds=(
                    self.reservation_store.now_epoch_seconds()
                    if self.reservation_store is not None
                    else Decimal("0")
                ),
            )
            if self.reservation_store is not None:
                if self.reservation_store.load() is not None:
                    raise AwsLifecycleError(
                        "an unresolved launch reservation exists; run reconcile"
                    )
                self.reservation_store.save(reservation)
            self._preflight_result = None
            self._preflight_authorized_at = None
            launch_requested_at = _monotonic_decimal(self._monotonic())
            try:
                payload = self.aws.run_instances(
                    request, max_hours=max_hours
                )
            except AwsCapacityUnavailable as error:
                last_capacity_error = error
                if self.reservation_store is not None:
                    durable = self.reservation_store.load()
                    if (
                        durable is None
                        or durable.client_token != client_token
                        or durable.state != "reserved"
                    ):
                        raise AwsLifecycleError(
                            "definitive capacity rejection does not match "
                            "the durable launch reservation"
                        ) from error
                    self.reservation_store.clear()
                continue
            try:
                instance = _launched_instance_payload(
                    payload,
                    request=request,
                    expected_instance_profile=_preflight_profile_identity(
                        final_preflight
                    ),
                )
            except AwsLifecycleError as error:
                raise AwsLifecycleError(
                    f"{error}; reconcile the ambiguous launch using "
                    f"ClientToken {client_token!r}"
                ) from error
            launched = LaunchedInstance(
                phase=phase,
                instance_id=instance["InstanceId"],
                instance_type=offer.instance_type,
                availability_zone=offer.availability_zone,
                subnet_id=offer.subnet_id,
                ami_id=ami_id,
                public_ip=instance.get("PublicIpAddress"),
                spot_hourly_usd=offer.hourly_usd,
                volume_hourly_usd=self.volume_hourly_usd,
                max_hours=max_hours,
                launched_monotonic=launch_requested_at,
            )
            if self.reservation_store is not None:
                try:
                    self.reservation_store.save(
                        replace(
                            reservation,
                            state="launched",
                            instance_id=instance["InstanceId"],
                        )
                    )
                except BaseException as state_error:
                    try:
                        self.terminate_and_settle(
                            launched, self.reservation_store.ledger_path
                        )
                    except BaseException as cleanup_error:
                        state_error.add_note(
                            "post-launch cleanup also failed: "
                            f"{type(cleanup_error).__name__}: {cleanup_error}"
                        )
                    raise
            return launched
        if last_capacity_error is not None:
            raise last_capacity_error
        raise AwsLifecycleError(
            "no valid allowed Spot offer is available for launch"
        )

    def _require_fresh_preflight(self) -> None:
        authorized_at = self._preflight_authorized_at
        if self._preflight_result is None or authorized_at is None:
            raise AwsLifecycleError(
                "a successful preflight is required before launch"
            )
        now = _monotonic_decimal(self._monotonic())
        age = now - authorized_at
        if age < 0 or age > _PREFLIGHT_AUTH_TTL_SECONDS:
            self._preflight_result = None
            self._preflight_authorized_at = None
            raise AwsLifecycleError(
                "preflight authorization expired; run preflight again"
            )

    def wait_for_running_public_ip(
        self,
        instance: LaunchedInstance,
        *,
        ledger_path: Path | None = None,
    ) -> LaunchedInstance:
        """Wait boundedly for the launched instance and its public IPv4."""
        deadline = (
            _monotonic_decimal(self._monotonic())
            + Decimal(_INSTANCE_READY_TIMEOUT_SECONDS)
        )
        deadline = min(deadline, self.training_deadline(instance))
        while True:
            summaries = self.aws.project_instances(
                instance_id=instance.instance_id
            )
            if ledger_path is not None:
                self._paid_tick(
                    instance,
                    ledger_path,
                    absolute_deadline=self.training_deadline(instance),
                )
            if not isinstance(summaries, tuple) or len(summaries) > 1:
                raise AwsLifecycleError(
                    "describe-instances returned duplicate launch targets"
                )
            if summaries:
                summary = summaries[0]
                if summary.get("instance_id") != instance.instance_id:
                    raise AwsLifecycleError(
                        "describe-instances returned a different launch target"
                    )
                state = summary.get("state")
                if state in {
                    "stopping",
                    "stopped",
                    "shutting-down",
                    "terminated",
                }:
                    raise AwsLifecycleError(
                        f"instance entered {state!r} before remote startup"
                    )
                if state not in {"pending", "running"}:
                    raise AwsLifecycleError(
                        f"instance returned unexpected state {state!r}"
                    )
                public_ip = summary.get("public_ip")
                if state == "running" and public_ip is not None:
                    return replace(
                        instance,
                        public_ip=_validated_public_ip(public_ip),
                    )
            now = _monotonic_decimal(self._monotonic())
            remaining = deadline - now
            if remaining <= 0:
                raise AwsLifecycleError(
                    "instance did not become running with a public IP before "
                    "the readiness or absolute training deadline"
                )
            self._sleeper(
                float(
                    min(
                        Decimal(_INSTANCE_READY_POLL_SECONDS),
                        remaining,
                    )
                )
            )

    def persist_elapsed(
        self, instance: LaunchedInstance, ledger_path: Path
    ) -> CostedRun:
        """Reload and durably persist monotonic observed cost."""
        ledger_path = Path(ledger_path)
        if ledger_path.exists():
            self.ledger = _configured_ledger(ledger_path, self.config)
        elapsed_seconds = max(
            Decimal("0"),
            _monotonic_decimal(self._monotonic())
            - instance.launched_monotonic,
        )
        prior_run = next(
            (
                prior
                for prior in self.ledger.runs
                if prior.phase == instance.phase
                and prior.instance_id == instance.instance_id
            ),
            None,
        )
        elapsed_hours = elapsed_seconds / _SECONDS_PER_HOUR
        if prior_run is not None:
            elapsed_hours = max(elapsed_hours, prior_run.hours)
        run = CostedRun(
            phase=instance.phase,
            instance_id=instance.instance_id,
            hours=elapsed_hours,
            instance_hourly_usd=self.config.on_demand_ceiling_usd[
                instance.instance_type
            ],
            volume_hourly_usd=instance.volume_hourly_usd,
        )
        self.ledger = self.ledger.update_run(run)
        self.ledger.save(ledger_path)
        return run

    def training_deadline(self, instance: LaunchedInstance) -> Decimal:
        return (
            instance.launched_monotonic
            + instance.max_hours * _SECONDS_PER_HOUR
        )

    def final_deadline(self, instance: LaunchedInstance) -> Decimal:
        return self.training_deadline(instance) + Decimal(
            self.config.grace_minutes * 60
        )

    def remote_cleanup_deadline(
        self, instance: LaunchedInstance
    ) -> Decimal:
        return (
            self.final_deadline(instance)
            - _EC2_SETTLEMENT_RESERVE_SECONDS
        )

    def _paid_tick(
        self,
        instance: LaunchedInstance,
        ledger_path: Path,
        *,
        absolute_deadline: Decimal,
    ) -> CostedRun:
        run = self.persist_elapsed(instance, ledger_path)
        if self.ledger.spent_usd >= self.config.shutdown_threshold_usd:
            raise AwsLifecycleError(
                "durable spend reached the configured shutdown threshold"
            )
        if _monotonic_decimal(self._monotonic()) >= absolute_deadline:
            raise AwsLifecycleError("absolute paid deadline reached")
        return run

    def terminate_and_settle(
        self, instance: LaunchedInstance, ledger_path: Path
    ) -> CostedRun:
        """Terminate, wait boundedly for terminal state, and persist final cost."""
        termination_error: BaseException | None = None
        terminal_confirmed = False
        try:
            self.aws.terminate_instance(instance.instance_id)
            deadline = self.final_deadline(instance)
            empty_confirmations = 0
            while True:
                remaining = deadline - _monotonic_decimal(self._monotonic())
                if remaining <= 0:
                    raise AwsLifecycleError(
                        "absolute paid deadline reached before EC2 terminal "
                        "confirmation"
                    )
                summaries = self.aws.project_instances(
                    instance_id=instance.instance_id,
                    timeout_seconds=float(min(Decimal("60"), remaining)),
                )
                self.persist_elapsed(instance, ledger_path)
                if not summaries:
                    empty_confirmations += 1
                    if empty_confirmations < 2:
                        continue
                    terminal_confirmed = True
                    break
                empty_confirmations = 0
                if summaries[0].get("state") == "terminated":
                    terminal_confirmed = True
                    break
                if len(summaries) != 1 or summaries[0].get(
                    "instance_id"
                ) != instance.instance_id:
                    raise AwsLifecycleError(
                        "termination status returned an unexpected target"
                    )
                remaining = deadline - _monotonic_decimal(self._monotonic())
                if remaining <= 0:
                    raise AwsLifecycleError(
                        "absolute paid deadline reached before EC2 terminal "
                        "confirmation"
                    )
                self._sleeper(float(min(Decimal("60"), remaining)))
        except BaseException as error:
            termination_error = error

        accounting_error: BaseException | None = None
        try:
            final_run = self.persist_elapsed(instance, ledger_path)
        except BaseException as error:
            accounting_error = error
            final_run = None

        if (
            terminal_confirmed
            and accounting_error is None
            and self.reservation_store is not None
        ):
            self.reservation_store.clear()
        if termination_error is not None:
            if accounting_error is not None:
                termination_error.add_note(
                    "final cost persistence also failed: "
                    f"{type(accounting_error).__name__}: {accounting_error}"
                )
            raise termination_error
        if accounting_error is not None:
            raise accounting_error
        assert final_run is not None
        return final_run

    def monitor_and_terminate(
        self, instance: LaunchedInstance, ledger_path: Path
    ) -> CostedRun:
        """Persist every successful poll and always terminate the EC2 instance."""
        if not isinstance(instance, LaunchedInstance):
            raise ValueError("instance must be a LaunchedInstance")
        body_error: BaseException | None = None
        final_run: CostedRun | None = None
        try:
            while True:
                poll = self.remote.poll(instance)
                running = getattr(poll, "running", None)
                if not isinstance(running, bool):
                    raise AwsLifecycleError(
                        "remote poll returned an invalid running state"
                    )
                run = self.persist_elapsed(instance, ledger_path)
                if not running:
                    break
                if (
                    self.ledger.spent_usd
                    >= self.config.shutdown_threshold_usd
                    or run.hours >= instance.max_hours
                ):
                    self.remote.request_shutdown(
                        instance,
                        on_tick=lambda: self._paid_tick(
                            instance,
                            ledger_path,
                            absolute_deadline=self.remote_cleanup_deadline(
                                instance
                            ),
                        ),
                        absolute_deadline=self.remote_cleanup_deadline(
                            instance
                        ),
                    )
                    break
                self._sleeper(60)
        except BaseException as error:
            body_error = error
        try:
            final_run = self.terminate_and_settle(instance, ledger_path)
        except BaseException as termination_error:
            if body_error is None:
                raise
            body_error.add_note(
                "EC2 termination also failed (including final accounting): "
                f"{type(termination_error).__name__}: {termination_error}"
            )
        if body_error is not None:
            raise body_error.with_traceback(body_error.__traceback__)
        assert final_run is not None
        return final_run


def _monotonic_decimal(value: Any) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AwsLifecycleError("monotonic clock returned an invalid value")
    result = Decimal(str(value))
    if not result.is_finite() or result < 0:
        raise AwsLifecycleError("monotonic clock returned an invalid value")
    return result


def _validated_public_ip(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise AwsLifecycleError(
            "instance has no public IP address for remote supervision"
        )
    try:
        address = ipaddress.ip_address(value)
    except ValueError as error:
        raise AwsLifecycleError(
            "instance returned an invalid public IP address"
        ) from error
    if address.version != 4:
        raise AwsLifecycleError("remote supervision requires a public IPv4 address")
    return value


def _preflight_profile_identity(
    preflight: PreflightResult,
) -> dict[str, str]:
    arn = preflight.instance_profile_arn
    profile_id = preflight.instance_profile_id
    if (
        not isinstance(arn, str)
        or not arn
        or not isinstance(profile_id, str)
        or not profile_id
    ):
        raise AwsLifecycleError(
            "preflight returned an invalid instance-profile identity"
        )
    return {"Arn": arn, "Id": profile_id}


def _launched_instance_payload(
    payload: Any,
    *,
    request: dict[str, Any],
    expected_instance_profile: dict[str, str] | None,
) -> dict[str, str]:
    if not isinstance(payload, dict):
        raise AwsLifecycleError("run-instances response must be a JSON object")
    instances = payload.get("Instances")
    if not isinstance(instances, list) or len(instances) != 1:
        raise AwsLifecycleError(
            "run-instances response must contain exactly one instance"
        )
    instance = instances[0]
    if not isinstance(instance, dict):
        raise AwsLifecycleError("run-instances returned a malformed instance")
    instance_id = instance.get("InstanceId")
    if (
        not isinstance(instance_id, str)
        or _INSTANCE_ID_PATTERN.fullmatch(instance_id) is None
    ):
        raise AwsLifecycleError(
            "run-instances response has no valid instance ID"
        )
    public_ip = instance.get("PublicIpAddress")
    if public_ip is not None and (
        not isinstance(public_ip, str) or not public_ip
    ):
        raise AwsLifecycleError(
            "run-instances response has an invalid public IP address"
        )
    placement = instance.get("Placement")
    security_groups = instance.get("SecurityGroups")
    actual_group_ids = (
        {
            group.get("GroupId")
            for group in security_groups
            if isinstance(group, dict)
        }
        if isinstance(security_groups, list)
        else set()
    )
    expected_group_ids = set(request["NetworkInterfaces"][0]["Groups"])
    actual_tags_payload = instance.get("Tags")
    actual_tags = (
        {
            (tag.get("Key"), tag.get("Value"))
            for tag in actual_tags_payload
            if isinstance(tag, dict)
        }
        if isinstance(actual_tags_payload, list)
        else set()
    )
    expected_tags = {
        (tag["Key"], tag["Value"])
        for tag in request["TagSpecifications"][0]["Tags"]
    }
    if (
        instance.get("ClientToken") != request["ClientToken"]
        or instance.get("ImageId") != request["ImageId"]
        or instance.get("InstanceType") != request["InstanceType"]
        or instance.get("InstanceLifecycle") != "spot"
        or instance.get("KeyName") != request["KeyName"]
        or instance.get("IamInstanceProfile")
        != expected_instance_profile
        or not isinstance(placement, dict)
        or placement.get("AvailabilityZone")
        != request["Placement"]["AvailabilityZone"]
        or instance.get("SubnetId")
        != request["NetworkInterfaces"][0]["SubnetId"]
        or actual_group_ids != expected_group_ids
        or actual_tags != expected_tags
    ):
        raise AwsLifecycleError(
            "run-instances response does not match the exact request"
        )
    return instance


def _shutdown_user_data(max_hours: Decimal) -> str:
    seconds = int(
        (max_hours * _SECONDS_PER_HOUR).to_integral_value(
            rounding=ROUND_CEILING
        )
    )
    minutes = max(1, (seconds + 59) // 60)
    script = (
        "#!/bin/sh\n"
        f"shutdown -h +{minutes} 'MarioAI maximum paid runtime reached'\n"
    )
    return base64.b64encode(script.encode("utf-8")).decode("ascii")


def _validate_termination_response(
    payload: dict[str, Any], instance_id: str
) -> None:
    transitions = payload.get("TerminatingInstances")
    if not isinstance(transitions, list) or len(transitions) != 1:
        raise AwsLifecycleError(
            "termination response is ambiguous; run reconcile"
        )
    transition = transitions[0]
    current = (
        transition.get("CurrentState")
        if isinstance(transition, dict)
        else None
    )
    if (
        not isinstance(transition, dict)
        or transition.get("InstanceId") != instance_id
        or not isinstance(current, dict)
        or current.get("Name") not in {"shutting-down", "terminated"}
    ):
        raise AwsLifecycleError(
            "termination response is ambiguous; run reconcile"
        )


def _validate_launch_request(
    request: Any,
    *,
    config: AwsConfig,
    authorized_subnet_azs: set[tuple[str, str]],
    resolved_ami_id: str | None,
    max_hours: Decimal,
) -> None:
    if not isinstance(request, dict):
        raise AwsLifecycleError("run-instances request must be a mapping")
    expected_keys = {
        "ImageId",
        "InstanceType",
        "MinCount",
        "MaxCount",
        "ClientToken",
        "UserData",
        "KeyName",
        "IamInstanceProfile",
        "Placement",
        "InstanceMarketOptions",
        "InstanceInitiatedShutdownBehavior",
        "NetworkInterfaces",
        "BlockDeviceMappings",
        "TagSpecifications",
    }
    if set(request) != expected_keys:
        raise AwsLifecycleError(
            "run-instances request fields do not match the exact allowlist"
        )
    if (
        resolved_ami_id is None
        or request.get("ImageId") != resolved_ami_id
        or _AMI_ID_PATTERN.fullmatch(resolved_ami_id) is None
    ):
        raise AwsLifecycleError(
            "run-instances AMI does not match the resolved public parameter"
        )
    if request.get("InstanceType") not in config.instance_types:
        raise AwsLifecycleError(
            "run-instances instance type is outside configuration"
        )
    if request.get("KeyName") != config.key_name:
        raise AwsLifecycleError("run-instances key does not match configuration")
    if request.get("IamInstanceProfile") != {
        "Name": config.instance_profile
    }:
        raise AwsLifecycleError(
            "run-instances profile does not match configuration"
        )
    placement = request.get("Placement")
    if (
        not isinstance(placement, dict)
        or set(placement) != {"AvailabilityZone"}
        or not isinstance(placement["AvailabilityZone"], str)
    ):
        raise AwsLifecycleError("run-instances placement is invalid")
    market_options = request.get("InstanceMarketOptions")
    spot_options = (
        market_options.get("SpotOptions")
        if isinstance(market_options, dict)
        else None
    )
    if (
        not isinstance(market_options, dict)
        or set(market_options) != {"MarketType", "SpotOptions"}
        or market_options.get("MarketType") != "spot"
        or not isinstance(spot_options, dict)
        or set(spot_options) != {"SpotInstanceType"}
        or spot_options.get("SpotInstanceType") != "one-time"
    ):
        raise AwsLifecycleError(
            "run-instances requires a one-time Spot market request"
        )
    if (
        isinstance(request.get("MinCount"), bool)
        or isinstance(request.get("MaxCount"), bool)
        or request.get("MinCount") != 1
        or request.get("MaxCount") != 1
    ):
        raise AwsLifecycleError("run-instances must request exactly one instance")
    if request.get("InstanceInitiatedShutdownBehavior") != "terminate":
        raise AwsLifecycleError(
            "instance-initiated shutdown behavior must be terminate"
        )
    client_token = request.get("ClientToken")
    if (
        not isinstance(client_token, str)
        or _CLIENT_TOKEN_PATTERN.fullmatch(client_token) is None
    ):
        raise AwsLifecycleError(
            "run-instances requires a valid idempotency ClientToken"
        )
    network_interfaces = request.get("NetworkInterfaces")
    network = (
        network_interfaces[0]
        if isinstance(network_interfaces, list)
        and len(network_interfaces) == 1
        and isinstance(network_interfaces[0], dict)
        else None
    )
    if (
        not isinstance(network, dict)
        or network
        != {
            "AssociatePublicIpAddress": True,
            "DeleteOnTermination": True,
            "DeviceIndex": 0,
            "Groups": [config.security_group_id],
            "SubnetId": network.get("SubnetId"),
        }
        or network.get("SubnetId") not in config.subnet_ids
        or (
            network["SubnetId"],
            placement["AvailabilityZone"],
        )
        not in authorized_subnet_azs
    ):
        raise AwsLifecycleError(
            "run-instances requires one public network interface"
        )
    block_devices = request.get("BlockDeviceMappings")
    ebs = (
        block_devices[0].get("Ebs")
        if isinstance(block_devices, list)
        and len(block_devices) == 1
        and isinstance(block_devices[0], dict)
        else None
    )
    if (
        not isinstance(ebs, dict)
        or block_devices[0].get("DeviceName") != "/dev/sda1"
        or ebs
        != {
            "DeleteOnTermination": True,
            "Encrypted": True,
            "VolumeSize": config.root_volume_gb,
            "VolumeType": "gp3",
        }
    ):
        raise AwsLifecycleError(
            "run-instances requires encrypted delete-on-termination gp3"
        )
    tags = [
        {"Key": "Project", "Value": "MarioAI-All32"},
        {
            "Key": "Phase",
            "Value": next(
                (
                    item.get("Value")
                    for specification in request.get(
                        "TagSpecifications", []
                    )
                    if isinstance(specification, dict)
                    for item in specification.get("Tags", [])
                    if isinstance(item, dict)
                    and item.get("Key") == "Phase"
                ),
                None,
            ),
        },
    ]
    phase = tags[1]["Value"]
    if phase not in config.allocations or request.get(
        "TagSpecifications"
    ) != [
        {"ResourceType": "instance", "Tags": tags},
        {"ResourceType": "volume", "Tags": tags},
    ]:
        raise AwsLifecycleError(
            "run-instances tags do not match the configured project phase"
        )
    if request.get("UserData") != _shutdown_user_data(max_hours):
        raise AwsLifecycleError(
            "run-instances shutdown delay does not match max_hours"
        )


def _positive_decimal_argument(value: str) -> Decimal:
    try:
        result = Decimal(value)
    except (ValueError, ArithmeticError) as error:
        raise argparse.ArgumentTypeError(
            "must be a positive decimal value"
        ) from error
    if not result.is_finite() or result <= 0:
        raise argparse.ArgumentTypeError("must be a positive decimal value")
    return result


def build_parser() -> argparse.ArgumentParser:
    """Build the explicit guarded lifecycle subcommands."""
    parser = argparse.ArgumentParser(
        description="Guarded AWS lifecycle for MarioAI all-32 training"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser(
        "preflight", help="run exact read-only prerequisite checks"
    )
    preflight.add_argument("--config", required=True, type=Path)

    benchmark = subparsers.add_parser(
        "benchmark",
        help="measure configured Spot candidates within the benchmark cap",
    )
    benchmark.add_argument("--config", required=True, type=Path)
    benchmark.add_argument("--ledger", required=True, type=Path)
    benchmark.add_argument(
        "--max-spend",
        required=True,
        type=_positive_decimal_argument,
    )
    benchmark.add_argument("--ssh-key", type=Path)
    benchmark.add_argument(
        "--repo-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )

    launch = subparsers.add_parser(
        "launch", help="preflight and launch one guarded Spot instance"
    )
    launch.add_argument("--config", required=True, type=Path)
    launch.add_argument("--ledger", required=True, type=Path)
    launch.add_argument("--phase", required=True)
    launch.add_argument(
        "--max-hours", required=True, type=_positive_decimal_argument
    )
    launch.add_argument("--instance-type")
    launch.add_argument("--ssh-key", type=Path)
    launch.add_argument(
        "--repo-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    resume = subparsers.add_parser(
        "resume",
        help="verify one exact manifested checkpoint, then launch guarded Spot",
    )
    resume.add_argument("--config", required=True, type=Path)
    resume.add_argument("--ledger", required=True, type=Path)
    resume.add_argument(
        "--phase", required=True, choices=("phase_1", "phase_2")
    )
    resume.add_argument(
        "--max-hours", required=True, type=_positive_decimal_argument
    )
    resume.add_argument("--instance-type")
    resume.add_argument("--ssh-key", type=Path)
    resume.add_argument(
        "--repo-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    resume.add_argument("--checkpoint-s3-uri", required=True)

    status = subparsers.add_parser(
        "status", help="show Project=MarioAI-All32 instance state"
    )
    status.add_argument("--config", required=True, type=Path)
    status.add_argument("--instance-id")

    terminate = subparsers.add_parser(
        "terminate", help="idempotently terminate one instance"
    )
    terminate.add_argument("--config", required=True, type=Path)
    terminate.add_argument("--instance-id", required=True)

    reconcile = subparsers.add_parser(
        "reconcile", help="terminate active project instances and show ledger"
    )
    reconcile.add_argument("--config", required=True, type=Path)
    reconcile.add_argument("--ledger", required=True, type=Path)

    return parser


def _configured_ledger(path: Path, config: AwsConfig) -> BudgetLedger:
    path = Path(path)
    if not path.exists():
        return BudgetLedger(
            cap_usd=config.cap_usd, allocations=config.allocations
        )
    ledger = BudgetLedger.load(path, cap_usd=config.cap_usd)
    if ledger.cap_usd != config.cap_usd:
        raise AwsLifecycleError(
            "persisted ledger cap does not match the approved configuration"
        )
    if dict(ledger.allocations) != dict(config.allocations):
        raise AwsLifecycleError(
            "persisted ledger allocations do not match the approved configuration"
        )
    return ledger


def _require_authoritative_ledger_superset(
    checkpoint: BudgetLedger, authoritative: BudgetLedger
) -> None:
    """Reject rollback of any checkpoint accounting identity or progress."""
    if (
        checkpoint.cap_usd != authoritative.cap_usd
        or dict(checkpoint.allocations) != dict(authoritative.allocations)
        or authoritative.spent_usd < checkpoint.spent_usd
    ):
        raise AwsLifecycleError(
            "authoritative ledger is not a monotonic superset of checkpoint "
            "accounting"
        )
    authoritative_runs = {
        (run.phase, run.instance_id): run for run in authoritative.runs
    }
    for checkpoint_run in checkpoint.runs:
        current_run = authoritative_runs.get(
            (checkpoint_run.phase, checkpoint_run.instance_id)
        )
        if (
            current_run is None
            or current_run.hours < checkpoint_run.hours
            or current_run.instance_hourly_usd
            != checkpoint_run.instance_hourly_usd
            or current_run.volume_hourly_usd
            != checkpoint_run.volume_hourly_usd
        ):
            raise AwsLifecycleError(
                "authoritative ledger is not a monotonic superset of "
                "checkpoint run history"
            )


def _settle_reservation(
    ledger: BudgetLedger,
    reservation: LaunchReservation,
    *,
    instance_id: str | None,
) -> BudgetLedger:
    """Consume the conservatively gated main runtime and global grace."""
    if (
        instance_id is not None
        and reservation.instance_id is not None
        and instance_id != reservation.instance_id
    ):
        raise AwsLifecycleError(
            "settlement instance ID does not match durable reservation"
        )
    pending_id = f"pending:{reservation.client_token}"
    main_instance_id = instance_id or reservation.instance_id or pending_id
    migrated_ids = {main_instance_id}
    if main_instance_id != pending_id:
        migrated_ids.add(pending_id)
    prior_main_runs = tuple(
        run
        for run in ledger.runs
        if run.phase == reservation.phase
        and run.instance_id in migrated_ids
    )
    filtered_runs = tuple(
        run for run in ledger.runs if run not in prior_main_runs
    )
    represented_before = sum(
        (run.cost_usd for run in ledger.runs), Decimal("0")
    )
    unrepresented_spend = max(
        Decimal("0"), ledger.spent_usd - represented_before
    )
    remaining_represented_cost = sum(
        (run.cost_usd for run in filtered_runs), Decimal("0")
    )
    base = BudgetLedger(
        cap_usd=ledger.cap_usd,
        spent_usd=remaining_represented_cost + unrepresented_spend,
        runs=filtered_runs,
        allocations=ledger.allocations,
    )
    main_run = CostedRun(
        phase=reservation.phase,
        instance_id=main_instance_id,
        hours=max(
            (run.hours for run in prior_main_runs),
            default=reservation.max_hours,
        )
        if prior_main_runs
        else reservation.max_hours,
        instance_hourly_usd=reservation.on_demand_hourly_usd,
        volume_hourly_usd=reservation.volume_hourly_usd,
    )
    settled = base.update_run(main_run)
    if settled.spent_usd < ledger.spent_usd:
        settled = BudgetLedger(
            cap_usd=settled.cap_usd,
            spent_usd=ledger.spent_usd,
            runs=settled.runs,
            allocations=settled.allocations,
        )
    grace_run = CostedRun(
        phase="__launch_grace__",
        instance_id=f"pending:{reservation.client_token}",
        hours=reservation.grace_hours,
        instance_hourly_usd=reservation.on_demand_hourly_usd,
        volume_hourly_usd=reservation.volume_hourly_usd,
    )
    return settled.update_run(grace_run)


def _instance_summaries(payload: Any) -> tuple[dict[str, str | None], ...]:
    if not isinstance(payload, dict):
        raise AwsLifecycleError(
            "describe-instances response must be a JSON object"
        )
    reservations = payload.get("Reservations")
    if not isinstance(reservations, list):
        raise AwsLifecycleError(
            "describe-instances response has invalid Reservations"
        )
    summaries: list[dict[str, str | None]] = []
    for reservation in reservations:
        if not isinstance(reservation, dict):
            raise AwsLifecycleError(
                "describe-instances response has malformed reservation"
            )
        instances = reservation.get("Instances")
        if not isinstance(instances, list):
            raise AwsLifecycleError(
                "describe-instances response has invalid Instances"
            )
        for instance in instances:
            if not isinstance(instance, dict):
                raise AwsLifecycleError(
                    "describe-instances response has malformed instance"
                )
            instance_id = instance.get("InstanceId")
            instance_type = instance.get("InstanceType")
            state_payload = instance.get("State")
            state = (
                state_payload.get("Name")
                if isinstance(state_payload, dict)
                else None
            )
            public_ip = instance.get("PublicIpAddress")
            if (
                not isinstance(instance_id, str)
                or _INSTANCE_ID_PATTERN.fullmatch(instance_id) is None
                or not isinstance(instance_type, str)
                or not instance_type
                or not isinstance(state, str)
                or not state
                or (
                    public_ip is not None
                    and (not isinstance(public_ip, str) or not public_ip)
                )
            ):
                raise AwsLifecycleError(
                    "describe-instances returned an invalid instance summary"
                )
            summaries.append(
                {
                    "instance_id": instance_id,
                    "instance_type": instance_type,
                    "public_ip": public_ip,
                    "state": state,
                }
            )
    return tuple(
        sorted(summaries, key=lambda item: str(item["instance_id"]))
    )


def _write_json(stream: Any, payload: Any) -> None:
    json.dump(payload, stream, sort_keys=True)
    stream.write("\n")


def _terminal_target_confirmed(aws: Any, instance_id: str) -> bool:
    """Require an explicit terminal state or two exact empty observations."""
    first = aws.project_instances(instance_id=instance_id)
    if len(first) > 1:
        raise AwsLifecycleError(
            "terminal confirmation returned duplicate instance targets"
        )
    if first:
        if first[0].get("instance_id") != instance_id:
            raise AwsLifecycleError(
                "terminal confirmation returned a different instance"
            )
        return first[0].get("state") == "terminated"
    second = aws.project_instances(instance_id=instance_id)
    if len(second) > 1:
        raise AwsLifecycleError(
            "terminal confirmation returned duplicate instance targets"
        )
    if not second:
        return True
    if second[0].get("instance_id") != instance_id:
        raise AwsLifecycleError(
            "terminal confirmation returned a different instance"
        )
    return second[0].get("state") == "terminated"


def _resume_training_args(
    bundle: ResumeBundle | PhaseResumeBundle,
    repo_dir: Path,
    *,
    authoritative_ledger_path: Path,
) -> tuple[str, ...]:
    repo_root = Path(repo_dir).resolve()

    def relative(path: Path) -> str:
        try:
            return path.resolve(strict=True).relative_to(repo_root).as_posix()
        except (OSError, ValueError) as error:
            raise AwsLifecycleError(
                "verified resume artifact escaped the repository"
            ) from error

    if isinstance(bundle, PhaseResumeBundle):
        run_name = bundle.lineage.run_name
        lineage_path = relative(bundle.lineage_path)
        return (
            "--config",
            "configs/all32.yaml",
            "--run-name",
            run_name,
            "--lineage",
            lineage_path,
            "--budget-ledger-snapshot",
            relative(authoritative_ledger_path),
        )

    run_name = bundle.manifest.get("run_name")
    if (
        not isinstance(run_name, str)
        or _CHECKPOINT_FILENAME_PATTERN.fullmatch(run_name) is None
    ):
        raise AwsLifecycleError("verified resume bundle has unsafe run name")
    arguments = [
        "--config",
        relative(bundle.run_config_path),
        "--phase-resolved-config",
        "--run-name",
        run_name,
        "--resume",
        relative(bundle.manifest_path),
    ]
    arguments.extend(
        [
            "--budget-ledger-snapshot",
            relative(authoritative_ledger_path),
        ]
    )
    return tuple(arguments)


def _snapshot_authoritative_ledger(
    bundle: ResumeBundle | PhaseResumeBundle,
    ledger_path: Path,
    config: AwsConfig,
) -> Path:
    """Copy current locked accounting into the staged remote resume bundle."""
    snapshot_path = bundle.root / "authoritative-budget-ledger.json"
    _configured_ledger(ledger_path, config).save(snapshot_path)
    return snapshot_path


def _resume_budget_ledger_paths(
    bundle: ResumeBundle | PhaseResumeBundle,
    repo_dir: Path,
) -> tuple[Path, ...]:
    if isinstance(bundle, ResumeBundle):
        return (bundle.budget_ledger_path,)
    identities = {
        identity.budget_ledger_path
        for identity in (
            bundle.lineage.last_candidate,
            bundle.lineage.promoted_best,
        )
        if identity is not None
    }
    return tuple(
        Path(repo_dir) / relative for relative in sorted(identities)
    )


def _fresh_phase_training_args(
    *,
    phase: str,
    repo_dir: Path,
    ledger_path: Path,
    config: AwsConfig,
) -> tuple[str, ...]:
    """Stage current accounting and return exact fresh phase-worker arguments."""
    if phase not in {"phase_1", "phase_2"}:
        raise AwsLifecycleError(
            "fresh phase training requires phase_1 or phase_2"
        )
    repo_root = Path(repo_dir).resolve()
    staging_parent = repo_root / ".resume"
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=f"{phase}-fresh-", dir=staging_parent)
    )
    snapshot_path = staging_root / "authoritative-budget-ledger.json"
    _configured_ledger(ledger_path, config).save(snapshot_path)
    relative_snapshot = snapshot_path.relative_to(repo_root).as_posix()
    return (
        "--config",
        "configs/all32.yaml",
        "--run-name",
        f"all32-{phase}",
        "--budget-ledger-snapshot",
        relative_snapshot,
    )


def _phase_spent_usd(ledger: BudgetLedger, phase: str) -> Decimal:
    return sum(
        (run.cost_usd for run in ledger.runs if run.phase == phase),
        Decimal("0"),
    )


def _run_benchmark_command(
    *,
    args: argparse.Namespace,
    config: AwsConfig,
    aws: Any,
    output: Any,
    remote: Any,
    runner: Callable[..., Any],
    monotonic: Callable[[], float],
    sleeper: Callable[[float], None],
    client_token_factory: Callable[[], str],
    wall_clock: Callable[[], float],
) -> int:
    allocation = config.allocations["benchmark"]
    if args.max_spend > allocation:
        raise ValueError(
            "--max-spend exceeds the configured benchmark allocation"
        )
    with LaunchStateStore(
        args.ledger, wall_clock=wall_clock
    ) as reservation_store:
        if reservation_store.load() is not None:
            raise AwsLifecycleError(
                "an unresolved launch reservation exists; run reconcile"
            )
        ledger = _configured_ledger(args.ledger, config)
        selected_remote = remote
        if selected_remote is None:
            ssh_key = args.ssh_key
            if ssh_key is None:
                ssh_key = (
                    Path.home() / ".ssh" / f"{config.key_name}.pem"
                )
            selected_remote = SshRemoteSupervisor(
                aws=aws,
                ssh_key=ssh_key,
                local_repo=args.repo_dir,
                runner=runner,
                monotonic=monotonic,
                sleeper=sleeper,
            )
        orchestrator = AwsOrchestrator(
            config=config,
            aws=aws,
            ledger=ledger,
            remote=selected_remote,
            monotonic=monotonic,
            sleeper=sleeper,
            client_token_factory=client_token_factory,
            reservation_store=reservation_store,
        )
        orchestrator.preflight()
        if aws.project_instances(active_only=True):
            raise AwsLifecycleError(
                "active MarioAI-All32 instance exists; run reconcile"
            )

        measurements: list[Benchmark] = []
        decision = "all_candidates_measured"
        grace_hours = Decimal(config.grace_minutes) / Decimal("60")
        for index, instance_type in enumerate(config.instance_types):
            phase_spent = _phase_spent_usd(
                orchestrator.ledger, "benchmark"
            )
            conservative_rate = (
                config.on_demand_ceiling_usd[instance_type]
                + orchestrator.volume_hourly_usd
            )
            projected_candidate = conservative_rate * (
                _BENCHMARK_MAX_HOURS + grace_hours
            )
            if phase_spent + projected_candidate > args.max_spend:
                decision = "allocation_exhausted"
                break
            if index:
                orchestrator.preflight()
            instance = orchestrator.launch_guarded_instance(
                "benchmark",
                _BENCHMARK_MAX_HOURS,
                instance_type=instance_type,
            )
            try:
                orchestrator.persist_elapsed(instance, args.ledger)
                instance = orchestrator.wait_for_running_public_ip(
                    instance, ledger_path=args.ledger
                )
                observation = selected_remote.benchmark(
                    instance,
                    environment_steps=_BENCHMARK_ENVIRONMENT_STEPS,
                    max_seconds=int(
                        _BENCHMARK_MAX_HOURS * _SECONDS_PER_HOUR
                    ),
                    on_tick=lambda: orchestrator._paid_tick(
                        instance,
                        args.ledger,
                        absolute_deadline=orchestrator.training_deadline(
                            instance
                        ),
                    ),
                    absolute_deadline=orchestrator.training_deadline(
                        instance
                    ),
                )
                if (
                    not isinstance(observation, BenchmarkObservation)
                    or observation.environment_steps
                    != _BENCHMARK_ENVIRONMENT_STEPS
                ):
                    raise AwsLifecycleError(
                        "benchmark returned an invalid fixed-step observation"
                    )
                measurement = Benchmark.from_observation(
                    instance_type=instance.instance_type,
                    observation=observation,
                    instance_hourly_usd=instance.spot_hourly_usd,
                    volume_hourly_usd=instance.volume_hourly_usd,
                )
            except BaseException as benchmark_error:
                try:
                    orchestrator.terminate_and_settle(
                        instance, args.ledger
                    )
                except BaseException as termination_error:
                    benchmark_error.add_note(
                        "benchmark EC2 termination/final accounting also "
                        f"failed: {type(termination_error).__name__}: "
                        f"{termination_error}"
                    )
                raise
            orchestrator.terminate_and_settle(instance, args.ledger)
            measurements.append(measurement)

        if not measurements:
            raise BudgetExceeded(
                "benchmark allocation cannot fit one guarded candidate"
            )
        selected = select_benchmark(measurements)
        _write_json(
            output,
            {
                "benchmark_steps_unit": "environment_steps",
                "benchmark_environment_steps": (
                    _BENCHMARK_ENVIRONMENT_STEPS
                ),
                "max_spend_usd": str(args.max_spend),
                "decision": decision,
                "candidates": [
                    measurement.to_dict()
                    for measurement in measurements
                ],
                "selected_instance_type": selected.instance_type,
                "env_steps_per_second": selected.env_steps_per_second,
                "cost_per_million_steps": str(
                    selected.cost_per_million_steps
                ),
                "peak_rss_gb": selected.peak_rss_gb,
                "conservative_benchmark_spent_usd": str(
                    _phase_spent_usd(
                        orchestrator.ledger, "benchmark"
                    )
                ),
            },
        )
    return 0


def main(
    argv: list[str] | None = None,
    *,
    runner: Callable[..., Any] = subprocess.run,
    stdout: Any = None,
    clock: Callable[[], Any] | None = None,
    aws_override: Any = None,
    remote: Any = None,
    monotonic: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
    client_token_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
    wall_clock: Callable[[], float] = time.time,
    checkpoint_store: Any = None,
) -> int:
    """Dispatch one explicit lifecycle command."""
    args = build_parser().parse_args(argv)
    output = sys.stdout if stdout is None else stdout
    config = AwsConfig.from_yaml(args.config)
    cli_kwargs: dict[str, Any] = {
        "profile": config.profile,
        "region": config.region,
        "runner": runner,
    }
    if clock is not None:
        cli_kwargs["clock"] = clock
    if aws_override is None:
        readonly = AwsCli(**cli_kwargs)
        aws = AwsCommandAdapter(
            config=config, readonly=readonly, runner=runner
        )
    else:
        aws = aws_override

    resume_bundle: ResumeBundle | PhaseResumeBundle | None = None
    if args.command == "resume":
        selected_store = checkpoint_store
        if selected_store is None:
            selected_store = S3CheckpointStore(
                profile=config.profile,
                region=config.region,
                runner=runner,
            )
        resume_bundle = restore_checkpoint_bundle(
            config=config,
            phase=args.phase,
            checkpoint_s3_uri=args.checkpoint_s3_uri,
            repo_dir=args.repo_dir,
            ledger_path=args.ledger,
            object_store=selected_store,
        )

    if args.command == "preflight":
        _write_json(output, asdict(aws.preflight(config)))
        return 0
    if args.command == "benchmark":
        return _run_benchmark_command(
            args=args,
            config=config,
            aws=aws,
            output=output,
            remote=remote,
            runner=runner,
            monotonic=monotonic,
            sleeper=sleeper,
            client_token_factory=client_token_factory,
            wall_clock=wall_clock,
        )
    if args.command == "status":
        aws.verify_account(config)
        instances = aws.project_instances(instance_id=args.instance_id)
        _write_json(output, {"instances": list(instances)})
        return 0
    if args.command == "terminate":
        aws.verify_account(config)
        instances = aws.project_instances(instance_id=args.instance_id)
        if len(instances) > 1:
            raise AwsLifecycleError(
                "describe-instances returned duplicate project targets"
            )
        termination_requested = bool(instances)
        if termination_requested:
            instance_id = instances[0]["instance_id"]
            if instance_id != args.instance_id:
                raise AwsLifecycleError(
                    "describe-instances returned a different project target"
                )
            aws.terminate_instance(args.instance_id)
        _write_json(
            output,
            {
                "instance_id": args.instance_id,
                "termination_requested": termination_requested,
            },
        )
        return 0
    if args.command == "reconcile":
        with LaunchStateStore(
            args.ledger, wall_clock=wall_clock
        ) as reservation_store:
            aws.verify_account(config)
            ledger = _configured_ledger(args.ledger, config)
            reservation = reservation_store.load()
            instances_by_id: dict[str, dict[str, str | None]] = {}
            for item in aws.project_instances(active_only=True):
                item_id = item.get("instance_id")
                if not isinstance(item_id, str):
                    raise AwsLifecycleError(
                        "active instance summary has no valid instance ID"
                    )
                instances_by_id[item_id] = item
            recovered_id: str | None = None
            old_empty_reservation_confirmed = False
            if reservation is not None:
                recovered = aws.project_instances(
                    client_token=reservation.client_token
                )
                if len(recovered) > 1:
                    raise AwsLifecycleError(
                        "ClientToken reconciliation returned multiple instances"
                    )
                reservation_age = (
                    reservation_store.now_epoch_seconds()
                    - reservation.requested_epoch_seconds
                )
                old_reserved_without_instance = (
                    reservation.state == "reserved"
                    and reservation.instance_id is None
                    and reservation_age >= (
                        (reservation.max_hours + reservation.grace_hours)
                        * _SECONDS_PER_HOUR
                    )
                )
                if not recovered and old_reserved_without_instance:
                    recovered = aws.project_instances(
                        client_token=reservation.client_token
                    )
                    if len(recovered) > 1:
                        raise AwsLifecycleError(
                            "ClientToken reconciliation returned multiple "
                            "instances"
                        )
                    old_empty_reservation_confirmed = not recovered
                recovered_id = reservation.instance_id
                if recovered:
                    token_id = recovered[0].get("instance_id")
                    if not isinstance(token_id, str):
                        raise AwsLifecycleError(
                            "ClientToken reconciliation returned no instance ID"
                        )
                    if (
                        recovered_id is not None
                        and token_id != recovered_id
                    ):
                        raise AwsLifecycleError(
                            "stored instance ID does not match ClientToken"
                        )
                    recovered_id = token_id
                    instances_by_id[token_id] = recovered[0]
                if (
                    recovered_id is not None
                    and recovered_id not in instances_by_id
                ):
                    exact = aws.project_instances(
                        instance_id=recovered_id
                    )
                    if len(exact) > 1:
                        raise AwsLifecycleError(
                            "stored instance lookup returned duplicate targets"
                        )
                    if exact:
                        if exact[0].get("instance_id") != recovered_id:
                            raise AwsLifecycleError(
                                "stored instance lookup returned a different target"
                            )
                        instances_by_id[recovered_id] = exact[0]
                if (
                    recovered_id is not None
                    and reservation.instance_id is None
                ):
                    reservation = replace(
                        reservation,
                        state="launched",
                        instance_id=recovered_id,
                    )
                    reservation_store.save(reservation)
            terminated_ids = []
            for instance in instances_by_id.values():
                instance_id = instance["instance_id"]
                if not isinstance(instance_id, str):
                    raise AwsLifecycleError(
                        "active instance summary has no valid instance ID"
                    )
                aws.terminate_instance(instance_id)
                terminated_ids.append(instance_id)
            reservation_cleared = reservation is None
            if reservation is not None:
                ledger = _settle_reservation(
                    ledger,
                    reservation,
                    instance_id=recovered_id,
                )
                ledger.save(args.ledger)
                if old_empty_reservation_confirmed or (
                    recovered_id is not None
                    and _terminal_target_confirmed(aws, recovered_id)
                ):
                    reservation_store.clear()
                    reservation_cleared = True
            _write_json(
                output,
                {
                    "cap_usd": str(ledger.cap_usd),
                    "remaining_usd": str(ledger.remaining_usd),
                    "reservation_cleared": reservation_cleared,
                    "spent_usd": str(ledger.spent_usd),
                    "terminated_instance_ids": terminated_ids,
                },
            )
        return 0
    if args.command in {"launch", "resume"}:
        with LaunchStateStore(
            args.ledger, wall_clock=wall_clock
        ) as reservation_store:
            if reservation_store.load() is not None:
                raise AwsLifecycleError(
                    "an unresolved launch reservation exists; run reconcile"
                )
            ledger = _configured_ledger(args.ledger, config)
            if resume_bundle is not None:
                for checkpoint_ledger_path in (
                    _resume_budget_ledger_paths(
                        resume_bundle, args.repo_dir
                    )
                ):
                    checkpoint_ledger = _configured_ledger(
                        checkpoint_ledger_path, config
                    )
                    _require_authoritative_ledger_superset(
                        checkpoint_ledger, ledger
                    )
            selected_remote = remote
            if selected_remote is None:
                ssh_key = args.ssh_key
                if ssh_key is None:
                    ssh_key = (
                        Path.home() / ".ssh" / f"{config.key_name}.pem"
                    )
                selected_remote = SshRemoteSupervisor(
                    aws=aws,
                    ssh_key=ssh_key,
                    local_repo=args.repo_dir,
                    runner=runner,
                    monotonic=monotonic,
                    sleeper=sleeper,
                )
            orchestrator = AwsOrchestrator(
                config=config,
                aws=aws,
                ledger=ledger,
                remote=selected_remote,
                monotonic=monotonic,
                sleeper=sleeper,
                client_token_factory=client_token_factory,
                reservation_store=reservation_store,
            )
            orchestrator.preflight()
            active_instances = aws.project_instances(active_only=True)
            if active_instances:
                raise AwsLifecycleError(
                    "active MarioAI-All32 instance exists; run reconcile"
                )

            def verify_resume_before_mutation() -> None:
                if resume_bundle is None:
                    return
                verify_resume_bundle(resume_bundle)
                current_ledger = _configured_ledger(
                    args.ledger, config
                )
                for checkpoint_ledger_path in (
                    _resume_budget_ledger_paths(
                        resume_bundle, args.repo_dir
                    )
                ):
                    _require_authoritative_ledger_superset(
                        _configured_ledger(
                            checkpoint_ledger_path, config
                        ),
                        current_ledger,
                    )
                if current_ledger != ledger:
                    raise AwsLifecycleError(
                        "authoritative ledger changed before mutation"
                    )

            instance = orchestrator.launch_guarded_instance(
                args.phase,
                args.max_hours,
                instance_type=args.instance_type,
                before_mutation=(
                    verify_resume_before_mutation
                    if resume_bundle is not None
                    else None
                ),
            )
            max_seconds = int(
                (args.max_hours * _SECONDS_PER_HOUR).to_integral_value(
                    rounding=ROUND_CEILING
                )
            )
            try:
                orchestrator.persist_elapsed(instance, args.ledger)
                instance = orchestrator.wait_for_running_public_ip(
                    instance, ledger_path=args.ledger
                )
                authoritative_ledger_snapshot = None
                if resume_bundle is not None:
                    authoritative_ledger_snapshot = (
                        _snapshot_authoritative_ledger(
                            resume_bundle, args.ledger, config
                        )
                    )
                fresh_train_args: tuple[str, ...] = ()
                if (
                    resume_bundle is None
                    and args.phase in {"phase_1", "phase_2"}
                ):
                    fresh_train_args = _fresh_phase_training_args(
                        phase=args.phase,
                        repo_dir=args.repo_dir,
                        ledger_path=args.ledger,
                        config=config,
                    )
                start_kwargs = {
                    "phase": args.phase,
                    "max_seconds": max_seconds,
                    "s3_prefix": config.s3_prefix,
                    "on_tick": lambda: orchestrator._paid_tick(
                        instance,
                        args.ledger,
                        absolute_deadline=orchestrator.training_deadline(
                            instance
                        ),
                    ),
                    "absolute_deadline": orchestrator.training_deadline(
                        instance
                    ),
                    "train_args": fresh_train_args,
                }
                if resume_bundle is not None:
                    start_kwargs["train_args"] = _resume_training_args(
                        resume_bundle,
                        args.repo_dir,
                        authoritative_ledger_path=(
                            authoritative_ledger_snapshot
                        ),
                    )
                selected_remote.start(instance, **start_kwargs)
            except BaseException as start_error:
                try:
                    orchestrator.terminate_and_settle(
                        instance, args.ledger
                    )
                except BaseException as termination_error:
                    start_error.add_note(
                        "EC2 termination/final accounting also failed: "
                        f"{type(termination_error).__name__}: "
                        f"{termination_error}"
                    )
                raise
            run = orchestrator.monitor_and_terminate(
                instance, args.ledger
            )
            _write_json(
                output,
                {
                    "costed_run": run.to_dict(),
                    "instance_id": instance.instance_id,
                },
            )
        return 0
    raise AwsLifecycleError(f"unsupported command {args.command!r}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AwsLifecycleError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
