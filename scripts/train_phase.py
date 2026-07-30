#!/usr/bin/env python
"""Chunked shared-policy training with diagnostic-only promotion evidence."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from contextlib import redirect_stdout
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import threading
import time

import marioai.evaluate as evaluation
import marioai.train as training
from stable_baselines3.common.callbacks import BaseCallback

from marioai.results import (
    ACCEPTANCE_POLICY_MODE,
    EvaluationReport,
    is_better_checkpoint,
    sha256_file,
    summarize_stage,
)
from marioai.sampling import assign_worker_levels, regression_weights


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "all32.yaml"
REPORT_ROOT = REPOSITORY_ROOT / "reports" / "diagnostics"
_BENCHMARK_WORKERS = 16
_SAFE_RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_SAFE_ARTIFACT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_CHECKPOINT_MANIFEST_FIELDS = {
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


def _relative_regular_file(path: Path, repository_root: Path) -> str:
    root = Path(repository_root).resolve()
    candidate = Path(path)
    try:
        resolved = candidate.resolve(strict=True)
        relative = resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise ValueError("phase artifact escaped the repository") from error
    if candidate.is_symlink() or not resolved.is_file():
        raise ValueError("phase artifact must be a regular file")
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("phase artifact path is unsafe")
    return relative.as_posix()


def _resolve_identity_path(relative: str, repository_root: Path) -> Path:
    if (
        not isinstance(relative, str)
        or not relative
        or "\\" in relative
        or "%" in relative
    ):
        raise ValueError("phase identity path is unsafe")
    candidate = Path(relative)
    if candidate.is_absolute() or any(
        part in {"", ".", ".."} for part in candidate.parts
    ):
        raise ValueError("phase identity path is unsafe")
    root = Path(repository_root).resolve()
    try:
        resolved = (root / candidate).resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise ValueError("phase identity path escaped the repository") from error
    if (root / candidate).is_symlink() or not resolved.is_file():
        raise ValueError("phase identity path is not a regular file")
    return resolved


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _read_json_mapping(path: Path, *, description: str) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} is not valid UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must be a JSON object")
    return payload


@dataclass(frozen=True)
class BundleIdentity:
    """Repository-relative identity of one complete immutable checkpoint."""

    manifest_path: str
    manifest_sha256: str
    model_path: str
    model_sha256: str
    run_config_path: str
    run_config_sha256: str
    signature_path: str
    signature_sha256: str
    vecnormalize_path: str | None
    vecnormalize_sha256: str | None
    budget_ledger_path: str
    budget_ledger_sha256: str
    phase: str
    run_name: str
    num_timesteps: int

    @classmethod
    def from_manifest(
        cls, manifest_path: Path, *, repository_root: Path
    ) -> BundleIdentity:
        manifest_path = Path(manifest_path)
        relative_manifest = _relative_regular_file(
            manifest_path, repository_root
        )
        manifest = _read_json_mapping(
            manifest_path, description="checkpoint manifest"
        )
        if set(manifest) != _CHECKPOINT_MANIFEST_FIELDS:
            raise ValueError("checkpoint manifest has an invalid schema")
        timesteps = manifest.get("num_timesteps")
        phase = manifest.get("phase")
        run_name = manifest.get("run_name")
        if (
            manifest.get("schema_version") != 1
            or isinstance(timesteps, bool)
            or not isinstance(timesteps, int)
            or timesteps < 0
            or phase not in {"phase_1", "phase_2"}
            or not isinstance(run_name, str)
            or _SAFE_RUN_NAME.fullmatch(run_name) is None
        ):
            raise ValueError("checkpoint manifest identity is invalid")

        def artifact(
            field: str, hash_field: str, suffix: str
        ) -> tuple[str, str]:
            name = manifest.get(field)
            expected_hash = manifest.get(hash_field)
            if (
                not isinstance(name, str)
                or _SAFE_ARTIFACT_NAME.fullmatch(name) is None
                or Path(name).name != name
                or not name.endswith(suffix)
                or not _valid_sha256(expected_hash)
            ):
                raise ValueError(
                    f"checkpoint manifest {field} identity is invalid"
                )
            path = manifest_path.parent / name
            relative = _relative_regular_file(path, repository_root)
            if training.sha256_file(path) != expected_hash:
                raise ValueError(
                    f"checkpoint manifest {field} SHA-256 mismatch"
                )
            return relative, expected_hash

        model_path, model_sha = artifact("model", "sha256", ".zip")
        run_config_path, run_config_sha = artifact(
            "run_config", "run_config_sha256", ".yaml"
        )
        signature_path, signature_sha = artifact(
            "signature", "signature_sha256", ".json"
        )
        ledger_path, ledger_sha = artifact(
            "budget_ledger", "budget_ledger_sha256", ".json"
        )
        vec_path: str | None = None
        vec_sha: str | None = None
        if manifest.get("normalize_reward") is True:
            vec_path, vec_sha = artifact(
                "vecnormalize", "vecnormalize_sha256", ".pkl"
            )
        elif (
            manifest.get("normalize_reward") is not False
            or manifest.get("vecnormalize") is not None
            or manifest.get("vecnormalize_sha256") is not None
        ):
            raise ValueError(
                "checkpoint manifest normalization identity is invalid"
            )
        identity = cls(
            manifest_path=relative_manifest,
            manifest_sha256=training.sha256_file(manifest_path),
            model_path=model_path,
            model_sha256=model_sha,
            run_config_path=run_config_path,
            run_config_sha256=run_config_sha,
            signature_path=signature_path,
            signature_sha256=signature_sha,
            vecnormalize_path=vec_path,
            vecnormalize_sha256=vec_sha,
            budget_ledger_path=ledger_path,
            budget_ledger_sha256=ledger_sha,
            phase=phase,
            run_name=run_name,
            num_timesteps=timesteps,
        )
        identity.validate(repository_root)
        return identity

    @classmethod
    def from_dict(
        cls, payload: object, *, repository_root: Path
    ) -> BundleIdentity:
        if not isinstance(payload, dict):
            raise ValueError("checkpoint bundle identity must be an object")
        try:
            identity = cls(**payload)
        except TypeError as error:
            raise ValueError(
                "checkpoint bundle identity has an invalid schema"
            ) from error
        identity.validate(repository_root)
        return identity

    def to_dict(self) -> dict:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }

    def validate(self, repository_root: Path) -> None:
        if (
            self.phase not in {"phase_1", "phase_2"}
            or _SAFE_RUN_NAME.fullmatch(self.run_name) is None
            or isinstance(self.num_timesteps, bool)
            or not isinstance(self.num_timesteps, int)
            or self.num_timesteps < 0
        ):
            raise ValueError("checkpoint bundle identity is invalid")
        hash_paths = (
            (self.manifest_path, self.manifest_sha256),
            (self.model_path, self.model_sha256),
            (self.run_config_path, self.run_config_sha256),
            (self.signature_path, self.signature_sha256),
            (self.budget_ledger_path, self.budget_ledger_sha256),
        )
        for relative, expected_hash in hash_paths:
            if not _valid_sha256(expected_hash):
                raise ValueError("checkpoint bundle hash is invalid")
            path = _resolve_identity_path(relative, repository_root)
            if training.sha256_file(path) != expected_hash:
                raise ValueError("checkpoint bundle SHA-256 mismatch")
        if (self.vecnormalize_path is None) != (
            self.vecnormalize_sha256 is None
        ):
            raise ValueError(
                "checkpoint bundle VecNormalize identity is incomplete"
            )
        if self.vecnormalize_path is not None:
            if not _valid_sha256(self.vecnormalize_sha256):
                raise ValueError(
                    "checkpoint bundle VecNormalize hash is invalid"
                )
            vec_path = _resolve_identity_path(
                self.vecnormalize_path, repository_root
            )
            if training.sha256_file(vec_path) != self.vecnormalize_sha256:
                raise ValueError(
                    "checkpoint bundle VecNormalize SHA-256 mismatch"
                )
        manifest = _read_json_mapping(
            _resolve_identity_path(
                self.manifest_path, repository_root
            ),
            description="checkpoint manifest",
        )
        expected = {
            "phase": self.phase,
            "run_name": self.run_name,
            "num_timesteps": self.num_timesteps,
            "model": Path(self.model_path).name,
            "sha256": self.model_sha256,
            "run_config": Path(self.run_config_path).name,
            "run_config_sha256": self.run_config_sha256,
            "signature": Path(self.signature_path).name,
            "signature_sha256": self.signature_sha256,
            "vecnormalize": (
                Path(self.vecnormalize_path).name
                if self.vecnormalize_path is not None
                else None
            ),
            "vecnormalize_sha256": self.vecnormalize_sha256,
            "budget_ledger": Path(self.budget_ledger_path).name,
            "budget_ledger_sha256": self.budget_ledger_sha256,
        }
        if any(manifest.get(field) != value for field, value in expected.items()):
            raise ValueError(
                "checkpoint bundle identity disagrees with its manifest"
            )

    def model(self, repository_root: Path) -> Path:
        return _resolve_identity_path(self.model_path, repository_root)


@dataclass(frozen=True)
class ReportIdentity:
    path: str
    sha256: str
    checkpoint_sha256: str

    @classmethod
    def from_report(
        cls, path: Path, *, repository_root: Path
    ) -> ReportIdentity:
        relative = _relative_regular_file(path, repository_root)
        report = EvaluationReport.read(Path(path))
        identity = cls(
            path=relative,
            sha256=training.sha256_file(path),
            checkpoint_sha256=report.checkpoint_sha256,
        )
        identity.load(repository_root)
        return identity

    @classmethod
    def from_dict(
        cls, payload: object, *, repository_root: Path
    ) -> ReportIdentity:
        if not isinstance(payload, dict):
            raise ValueError("diagnostic report identity must be an object")
        try:
            identity = cls(**payload)
        except TypeError as error:
            raise ValueError(
                "diagnostic report identity has an invalid schema"
            ) from error
        identity.load(repository_root)
        return identity

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
        }

    def load(self, repository_root: Path) -> EvaluationReport:
        if not _valid_sha256(self.sha256) or not _valid_sha256(
            self.checkpoint_sha256
        ):
            raise ValueError("diagnostic report hash identity is invalid")
        path = _resolve_identity_path(self.path, repository_root)
        if training.sha256_file(path) != self.sha256:
            raise ValueError("diagnostic report SHA-256 mismatch")
        report = EvaluationReport.read(path)
        if report.checkpoint_sha256 != self.checkpoint_sha256:
            raise ValueError(
                "diagnostic report checkpoint identity mismatch"
            )
        return report


@dataclass(frozen=True)
class PhaseLineage:
    phase: str
    run_name: str
    last_candidate: BundleIdentity | None
    last_diagnostic: ReportIdentity | None
    promoted_best: BundleIdentity | None
    best_report: ReportIdentity | None
    next_weights: Mapping[str, float]
    pending_training: Mapping[str, object] | None

    def validate(self, repository_root: Path) -> None:
        if (
            self.phase not in {"phase_1", "phase_2"}
            or _SAFE_RUN_NAME.fullmatch(self.run_name) is None
            or (self.promoted_best is None) != (self.best_report is None)
            or (
                self.last_candidate is None
                and self.last_diagnostic is not None
            )
            or not isinstance(self.next_weights, Mapping)
            or not all(
                isinstance(level, str)
                and level
                and not isinstance(weight, bool)
                and isinstance(weight, (int, float))
                and math.isfinite(weight)
                and weight > 0
                for level, weight in self.next_weights.items()
            )
            or (
                self.pending_training is not None
                and not isinstance(self.pending_training, Mapping)
            )
        ):
            raise ValueError("phase lineage identity is invalid")
        for bundle in (self.last_candidate, self.promoted_best):
            if bundle is not None:
                bundle.validate(repository_root)
                if bundle.phase != self.phase:
                    raise ValueError(
                        "phase lineage checkpoint phase mismatch"
                    )
        if self.last_diagnostic is not None:
            self.last_diagnostic.load(repository_root)
            if (
                self.last_candidate is None
                or self.last_diagnostic.checkpoint_sha256
                != self.last_candidate.model_sha256
            ):
                raise ValueError(
                    "last diagnostic does not match last candidate"
                )
        if self.best_report is not None:
            self.best_report.load(repository_root)
            if (
                self.promoted_best is None
                or self.best_report.checkpoint_sha256
                != self.promoted_best.model_sha256
            ):
                raise ValueError(
                    "best diagnostic does not match promoted checkpoint"
                )

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "phase": self.phase,
            "run_name": self.run_name,
            "last_candidate": (
                self.last_candidate.to_dict()
                if self.last_candidate is not None
                else None
            ),
            "last_diagnostic": (
                self.last_diagnostic.to_dict()
                if self.last_diagnostic is not None
                else None
            ),
            "promoted_best": (
                self.promoted_best.to_dict()
                if self.promoted_best is not None
                else None
            ),
            "best_report": (
                self.best_report.to_dict()
                if self.best_report is not None
                else None
            ),
            "next_weights": dict(self.next_weights),
            "pending_training": (
                dict(self.pending_training)
                if self.pending_training is not None
                else None
            ),
        }

    @classmethod
    def from_dict(
        cls, payload: object, *, repository_root: Path
    ) -> PhaseLineage:
        expected = {
            "schema_version",
            "phase",
            "run_name",
            "last_candidate",
            "last_diagnostic",
            "promoted_best",
            "best_report",
            "next_weights",
            "pending_training",
        }
        if (
            not isinstance(payload, dict)
            or set(payload) != expected
            or payload.get("schema_version") != 1
        ):
            raise ValueError("phase lineage has an invalid schema")

        def bundle(value: object) -> BundleIdentity | None:
            return (
                None
                if value is None
                else BundleIdentity.from_dict(
                    value, repository_root=repository_root
                )
            )

        def report(value: object) -> ReportIdentity | None:
            return (
                None
                if value is None
                else ReportIdentity.from_dict(
                    value, repository_root=repository_root
                )
            )

        lineage = cls(
            phase=payload["phase"],
            run_name=payload["run_name"],
            last_candidate=bundle(payload["last_candidate"]),
            last_diagnostic=report(payload["last_diagnostic"]),
            promoted_best=bundle(payload["promoted_best"]),
            best_report=report(payload["best_report"]),
            next_weights=payload["next_weights"],
            pending_training=payload["pending_training"],
        )
        lineage.validate(repository_root)
        return lineage


class PhaseLineageStore:
    """Content-addressed phase state with one atomically replaced head."""

    def __init__(
        self, head_path: Path, *, repository_root: Path
    ) -> None:
        self.head_path = Path(head_path)
        self.repository_root = Path(repository_root).resolve()

    @staticmethod
    def _canonical(payload: object) -> bytes:
        return (
            json.dumps(payload, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _write_immutable(self, path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", dir=path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as destination:
                destination.write(payload)
                destination.flush()
                os.fsync(destination.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.read_bytes() != payload:
                    raise
            else:
                self._fsync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)

    def _replace_head(self, payload: bytes) -> None:
        self.head_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.head_path.name}.",
            dir=self.head_path.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as destination:
                destination.write(payload)
                destination.flush()
                os.fsync(destination.fileno())
            os.replace(temporary, self.head_path)
            self._fsync_directory(self.head_path.parent)
        finally:
            temporary.unlink(missing_ok=True)

    def save(self, lineage: PhaseLineage) -> Path:
        if not isinstance(lineage, PhaseLineage):
            raise ValueError("lineage must be a PhaseLineage")
        lineage.validate(self.repository_root)
        state_payload = self._canonical(lineage.to_dict())
        state_hash = hashlib.sha256(state_payload).hexdigest()
        state_name = f"phase_state_{state_hash}.json"
        state_path = self.head_path.parent / state_name
        self._write_immutable(state_path, state_payload)
        head_payload = self._canonical(
            {
                "schema_version": 1,
                "kind": "marioai-phase-lineage",
                "phase": lineage.phase,
                "run_name": lineage.run_name,
                "phase_state": state_name,
                "phase_state_sha256": state_hash,
            }
        )
        self._replace_head(head_payload)
        return self.head_path

    def load(self) -> PhaseLineage:
        head = _read_json_mapping(
            self.head_path, description="phase lineage head"
        )
        expected = {
            "schema_version",
            "kind",
            "phase",
            "run_name",
            "phase_state",
            "phase_state_sha256",
        }
        state_name = head.get("phase_state")
        state_hash = head.get("phase_state_sha256")
        if (
            set(head) != expected
            or head.get("schema_version") != 1
            or head.get("kind") != "marioai-phase-lineage"
            or not isinstance(state_name, str)
            or _SAFE_ARTIFACT_NAME.fullmatch(state_name) is None
            or Path(state_name).name != state_name
            or not state_name.startswith("phase_state_")
            or not state_name.endswith(".json")
            or not _valid_sha256(state_hash)
        ):
            raise ValueError("phase lineage head has an invalid schema")
        state_path = self.head_path.parent / state_name
        _relative_regular_file(state_path, self.repository_root)
        state_payload = state_path.read_bytes()
        if hashlib.sha256(state_payload).hexdigest() != state_hash:
            raise ValueError("phase lineage state SHA-256 mismatch")
        lineage = PhaseLineage.from_dict(
            json.loads(state_payload),
            repository_root=self.repository_root,
        )
        if (
            lineage.phase != head.get("phase")
            or lineage.run_name != head.get("run_name")
        ):
            raise ValueError("phase lineage head identity mismatch")
        return lineage


class PhaseLoop:
    """Train one shared policy in fixed chunks and retain only improvements."""

    def __init__(
        self,
        *,
        phase: str,
        deadline: datetime,
        checkpoint: Path | BundleIdentity | None,
        levels: Sequence[str],
        n_envs: int,
        total_timesteps: int,
        chunk_timesteps: int,
        diagnostic_episodes: int,
        diagnostic_seed: int,
        train_chunk: Callable[..., Path],
        diagnose: Callable[..., EvaluationReport],
        checkpoint_timesteps: Callable[[Path], int],
        now: Callable[[], datetime],
        report_dir: Path,
        lineage_store: PhaseLineageStore | None = None,
        repository_root: Path | None = None,
        run_name: str | None = None,
    ) -> None:
        self.phase = phase
        self.deadline = deadline
        self.checkpoint = (
            checkpoint
            if isinstance(checkpoint, BundleIdentity)
            else Path(checkpoint) if checkpoint is not None else None
        )
        self.levels = tuple(levels)
        self.n_envs = n_envs
        self.total_timesteps = total_timesteps
        self.chunk_timesteps = chunk_timesteps
        self.diagnostic_episodes = diagnostic_episodes
        self.diagnostic_seed = diagnostic_seed
        self._train_chunk = train_chunk
        self._diagnose = diagnose
        self._checkpoint_timesteps = checkpoint_timesteps
        self._now = now
        self.report_dir = Path(report_dir)
        self.lineage_store = lineage_store
        self.repository_root = (
            Path(repository_root).resolve()
            if repository_root is not None
            else None
        )
        self.run_name = run_name
        if self.lineage_store is not None and (
            self.repository_root is None
            or not isinstance(self.run_name, str)
            or _SAFE_RUN_NAME.fullmatch(self.run_name) is None
        ):
            raise ValueError(
                "durable phase loop requires repository root and safe run name"
            )
        self.next_weights = {level: 1.0 for level in self.levels}
        self.next_worker_levels = assign_worker_levels(
            self.levels,
            self.n_envs,
            self.next_weights,
        )

    def run(self) -> Path:
        if self.lineage_store is not None:
            return self._run_durable()
        current_timesteps = (
            self._checkpoint_timesteps(self.checkpoint)
            if self.checkpoint is not None
            else 0
        )
        best_checkpoint = self.checkpoint
        training_checkpoint = self.checkpoint
        best_report: EvaluationReport | None = None
        while (
            current_timesteps < self.total_timesteps
            and self._now() < self.deadline
        ):
            target_timesteps = min(
                current_timesteps + self.chunk_timesteps,
                self.total_timesteps,
            )
            candidate = Path(
                self._train_chunk(
                    checkpoint=training_checkpoint,
                    target_timesteps=target_timesteps,
                    level_weights=self.next_weights,
                )
            )
            training_checkpoint = candidate
            report = self._diagnose(
                checkpoint=candidate,
                active_levels=self.levels,
                episodes=self.diagnostic_episodes,
                seed=self.diagnostic_seed + target_timesteps,
                deterministic=False,
            )
            self._validate_diagnostic(candidate, report)
            self._write_diagnostic(target_timesteps, report)

            incumbent_passing = self._passing(best_report)
            candidate_passing = self._passing(report)
            self.next_weights = regression_weights(
                self.levels,
                incumbent_passing,
                candidate_passing,
            )
            self.next_worker_levels = assign_worker_levels(
                self.levels,
                self.n_envs,
                self.next_weights,
            )
            if is_better_checkpoint(report, best_report):
                best_checkpoint = candidate
                best_report = report
            current_timesteps = target_timesteps

        if best_checkpoint is None:
            raise RuntimeError(
                "phase deadline was reached before a checkpoint was available"
            )
        return best_checkpoint

    def _run_durable(self) -> Path:
        assert self.lineage_store is not None
        assert self.repository_root is not None
        assert self.run_name is not None
        if self.lineage_store.head_path.exists():
            lineage = self.lineage_store.load()
            if (
                lineage.phase != self.phase
                or lineage.run_name != self.run_name
                or set(lineage.next_weights) != set(self.levels)
            ):
                raise ValueError(
                    "persisted phase lineage does not match this phase run"
                )
        else:
            if self.checkpoint is None:
                initial_bundle = None
            elif isinstance(self.checkpoint, BundleIdentity):
                initial_bundle = self.checkpoint
                initial_bundle.validate(self.repository_root)
            else:
                raise ValueError(
                    "durable phase resume requires a complete bundle identity"
                )
            lineage = PhaseLineage(
                phase=self.phase,
                run_name=self.run_name,
                last_candidate=initial_bundle,
                last_diagnostic=None,
                promoted_best=None,
                best_report=None,
                next_weights={
                    level: 1.0 for level in self.levels
                },
                pending_training=None,
            )
            self.lineage_store.save(lineage)
        self.next_weights = dict(lineage.next_weights)
        self.next_worker_levels = assign_worker_levels(
            self.levels,
            self.n_envs,
            self.next_weights,
        )

        while True:
            candidate = lineage.last_candidate
            if candidate is not None and lineage.last_diagnostic is None:
                if self._now() >= self.deadline:
                    break
                try:
                    report = self._diagnose(
                        checkpoint=candidate,
                        active_levels=self.levels,
                        episodes=self.diagnostic_episodes,
                        seed=(
                            self.diagnostic_seed
                            + candidate.num_timesteps
                        ),
                        deterministic=False,
                        deadline=self.deadline,
                    )
                except evaluation.EvaluationDeadlineReached:
                    break
                self._validate_bundle_diagnostic(candidate, report)
                report_identity = self._write_bundle_diagnostic(
                    candidate, report
                )
                incumbent_report = (
                    lineage.best_report.load(self.repository_root)
                    if lineage.best_report is not None
                    else None
                )
                self.next_weights = regression_weights(
                    self.levels,
                    self._passing(incumbent_report),
                    self._passing(report),
                )
                self.next_worker_levels = assign_worker_levels(
                    self.levels,
                    self.n_envs,
                    self.next_weights,
                )
                if is_better_checkpoint(report, incumbent_report):
                    promoted_best = candidate
                    best_report = report_identity
                else:
                    promoted_best = lineage.promoted_best
                    best_report = lineage.best_report
                lineage = replace(
                    lineage,
                    last_diagnostic=report_identity,
                    promoted_best=promoted_best,
                    best_report=best_report,
                    next_weights=dict(self.next_weights),
                    pending_training=None,
                )
                self.lineage_store.save(lineage)
                continue

            current_timesteps = (
                candidate.num_timesteps if candidate is not None else 0
            )
            if (
                current_timesteps >= self.total_timesteps
                or self._now() >= self.deadline
            ):
                break
            target_timesteps = min(
                current_timesteps + self.chunk_timesteps,
                self.total_timesteps,
            )
            lineage = replace(
                lineage,
                pending_training={
                    "source_manifest_sha256": (
                        candidate.manifest_sha256
                        if candidate is not None
                        else None
                    ),
                    "target_timesteps": target_timesteps,
                },
            )
            self.lineage_store.save(lineage)
            trained = self._train_chunk(
                checkpoint=candidate,
                target_timesteps=target_timesteps,
                level_weights=self.next_weights,
                deadline=self.deadline,
            )
            if not isinstance(trained, BundleIdentity):
                raise ValueError(
                    "phase training must return a complete bundle identity"
                )
            trained.validate(self.repository_root)
            if (
                trained.phase != self.phase
                or trained.num_timesteps <= current_timesteps
                or trained.num_timesteps > target_timesteps
            ):
                raise ValueError(
                    "phase candidate has invalid environment-step progress"
                )
            lineage = replace(
                lineage,
                last_candidate=trained,
                last_diagnostic=None,
                pending_training=None,
            )
            self.lineage_store.save(lineage)

        if lineage.promoted_best is None:
            raise RuntimeError(
                "phase stopped before any diagnostic checkpoint was promoted"
            )
        return lineage.promoted_best.model(self.repository_root)

    def _validate_bundle_diagnostic(
        self,
        checkpoint: BundleIdentity,
        report: EvaluationReport,
    ) -> None:
        expected_seeds = [
            self.diagnostic_seed
            + checkpoint.num_timesteps
            + level_index * self.diagnostic_episodes
            + episode_index
            for level_index, _level in enumerate(self.levels)
            for episode_index in range(self.diagnostic_episodes)
        ]
        if (
            not isinstance(report, EvaluationReport)
            or not report._has_valid_schema()
            or report.checkpoint_sha256 != checkpoint.model_sha256
            or report.policy_mode != ACCEPTANCE_POLICY_MODE
            or report.deterministic
            or report.requested_episodes != self.diagnostic_episodes
            or self.diagnostic_episodes != 3
            or set(report.stages) != set(self.levels)
            or len(report.rollouts)
            != len(self.levels) * self.diagnostic_episodes
            or [rollout.seed for rollout in report.rollouts]
            != expected_seeds
            or len(set(expected_seeds)) != len(expected_seeds)
            or report.passed
        ):
            raise ValueError(
                "diagnostic report does not match the shared candidate"
            )
        for level in self.levels:
            stage_rollouts = [
                rollout
                for rollout in report.rollouts
                if rollout.level == level
            ]
            if (
                len(stage_rollouts) != self.diagnostic_episodes
                or report.stages[level]
                != summarize_stage(level, stage_rollouts)
            ):
                raise ValueError(
                    "diagnostic report has inconsistent stage evidence"
                )

    def _write_bundle_diagnostic(
        self,
        checkpoint: BundleIdentity,
        report: EvaluationReport,
    ) -> ReportIdentity:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        path = self.report_dir / (
            f"{self.phase}-{checkpoint.num_timesteps:012d}-"
            f"{report.checkpoint_sha256[:12]}.json"
        )
        report.write(path)
        assert self.repository_root is not None
        return ReportIdentity.from_report(
            path, repository_root=self.repository_root
        )

    def _validate_diagnostic(
        self, checkpoint: Path, report: EvaluationReport
    ) -> None:
        if (
            not isinstance(report, EvaluationReport)
            or report.checkpoint_sha256 != sha256_file(checkpoint)
            or report.deterministic
            or report.requested_episodes != self.diagnostic_episodes
            or set(report.stages) != set(self.levels)
            or report.passed
        ):
            raise ValueError(
                "diagnostic report does not match the shared candidate"
            )

    def _write_diagnostic(
        self, target_timesteps: int, report: EvaluationReport
    ) -> None:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        report.write(
            self.report_dir
            / (
                f"{self.phase}-{target_timesteps:012d}-"
                f"{report.checkpoint_sha256[:12]}.json"
            )
        )

    def _passing(
        self, report: EvaluationReport | None
    ) -> Mapping[str, bool]:
        if report is None:
            return {level: False for level in self.levels}
        return {
            level: report.stages[level].get("passed") is True
            for level in self.levels
        }


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _checkpoint_timesteps(checkpoint: Path) -> int:
    model = training.PPO.load(str(checkpoint), device="cpu")
    timesteps = getattr(model, "num_timesteps", None)
    if (
        isinstance(timesteps, bool)
        or not isinstance(timesteps, int)
        or timesteps < 0
    ):
        raise ValueError("checkpoint has invalid environment-step progress")
    return timesteps


def _train_shared_chunk(
    *,
    phase: str,
    config: Mapping,
    config_path: Path | None = None,
    phase_resolved_config: bool = False,
    run_name_prefix: str | None = None,
    budget_ledger_snapshot: Path | None = None,
    checkpoint_vecnormalize: Path | None = None,
    checkpoint: Path | BundleIdentity | None,
    target_timesteps: int,
    level_weights: Mapping[str, float],
    repository_root: Path | None = None,
    deadline: datetime | None = None,
) -> BundleIdentity:
    del config
    if config_path is None:
        config_path = CONFIG_PATH
    if repository_root is None:
        repository_root = REPOSITORY_ROOT
    if run_name_prefix is None:
        run_name_prefix = f"all32-{phase}"
    run_name = f"{run_name_prefix}-chunk-{target_timesteps:012d}"
    arguments = [
        "--config",
        str(config_path),
        "--phase",
        phase,
        "--timesteps",
        str(target_timesteps),
        "--run-name",
        run_name,
        "--level-weights-json",
        json.dumps(dict(level_weights), sort_keys=True, separators=(",", ":")),
        "--publish-final-checkpoint",
    ]
    if phase_resolved_config:
        arguments.append("--phase-resolved-config")
    if deadline is not None:
        if deadline.tzinfo is None or deadline.utcoffset() is None:
            raise ValueError("chunk deadline must be timezone-aware")
        arguments.extend(
            ["--deadline-epoch", str(int(deadline.timestamp()))]
        )
    if checkpoint is not None:
        if isinstance(checkpoint, BundleIdentity):
            checkpoint_path = checkpoint.model(repository_root)
            candidate_vecnormalize = (
                _resolve_identity_path(
                    checkpoint.vecnormalize_path, repository_root
                )
                if checkpoint.vecnormalize_path is not None
                else None
            )
        else:
            checkpoint_path = Path(checkpoint)
            candidate_vecnormalize = checkpoint_vecnormalize
        arguments.extend(["--resume", str(checkpoint_path)])
        if candidate_vecnormalize is not None:
            arguments.extend(
                [
                    "--resume-vecnormalize",
                    str(candidate_vecnormalize),
                ]
            )
    ledger_snapshot = budget_ledger_snapshot
    if ledger_snapshot is None:
        configured_snapshot = os.environ.get(
            "MARIOAI_BUDGET_LEDGER_SNAPSHOT"
        )
        if configured_snapshot:
            ledger_snapshot = Path(configured_snapshot)
    if ledger_snapshot:
        arguments.extend(
            ["--budget-ledger-snapshot", str(ledger_snapshot)]
        )
    manifest_path = training.main(arguments)
    if manifest_path is None or not Path(manifest_path).is_file():
        raise RuntimeError(
            "training did not produce a complete shared checkpoint manifest"
        )
    return BundleIdentity.from_manifest(
        Path(manifest_path), repository_root=repository_root
    )


def _diagnose_shared_checkpoint(
    *,
    checkpoint: Path | BundleIdentity,
    active_levels: Sequence[str],
    episodes: int,
    seed: int,
    deterministic: bool,
    repository_root: Path | None = None,
    deadline: datetime | None = None,
) -> EvaluationReport:
    if isinstance(checkpoint, BundleIdentity):
        if repository_root is None:
            repository_root = REPOSITORY_ROOT
        model_path = checkpoint.model(repository_root)
    else:
        model_path = Path(checkpoint)
    return evaluation.evaluate_checkpoint(
        model_path,
        active_levels,
        episodes=episodes,
        seed=seed,
        deterministic=deterministic,
        deadline=deadline,
    )


def run_phase(
    phase: str,
    deadline: datetime,
    checkpoint: Path | BundleIdentity | None,
    *,
    config_path: Path | None = None,
    run_name: str | None = None,
    budget_ledger_snapshot: Path | None = None,
    lineage_path: Path | None = None,
    phase_resolved_config: bool = False,
    checkpoint_vecnormalize: Path | None = None,
    repository_root: Path | None = None,
) -> Path:
    """Run fixed environment-step chunks until the target or remote deadline."""
    if repository_root is None:
        repository_root = REPOSITORY_ROOT
    repository_root = Path(repository_root).resolve()
    if config_path is None:
        config_path = CONFIG_PATH
    if deadline.tzinfo is None or deadline.utcoffset() is None:
        raise ValueError("deadline must be timezone-aware")
    if run_name is None:
        run_name = f"all32-{phase}"
    if _SAFE_RUN_NAME.fullmatch(run_name) is None:
        raise ValueError("run_name is unsafe")
    if checkpoint is not None and not isinstance(
        checkpoint, BundleIdentity
    ):
        checkpoint_path = Path(checkpoint)
        if checkpoint_path.suffix != ".json":
            raise ValueError(
                "phase resume requires an immutable checkpoint manifest"
            )
        checkpoint = BundleIdentity.from_manifest(
            checkpoint_path, repository_root=repository_root
        )
    overrides = argparse.Namespace(
        phase_resolved_config=phase_resolved_config,
        levels=None,
        timesteps=None,
        n_envs=None,
        lr=None,
        ent_coef=None,
        level_weights_json=None,
    )
    config = training.load_training_config(
        str(config_path), phase, overrides
    )
    train_config = config["train"]
    evaluation_config = config["evaluation"]
    if evaluation_config.get("diagnostic_episodes") != 3:
        raise ValueError(
            "all-32 phase diagnostics must use exactly three rollouts"
        )
    canonical_head = (
        repository_root / "models" / run_name / "latest.json"
    )
    selected_head = (
        Path(lineage_path) if lineage_path is not None else canonical_head
    )
    lineage_store = PhaseLineageStore(
        selected_head, repository_root=repository_root
    )
    if lineage_path is not None and selected_head != canonical_head:
        restored_lineage = lineage_store.load()
        lineage_store = PhaseLineageStore(
            canonical_head, repository_root=repository_root
        )
        lineage_store.save(restored_lineage)
    loop = PhaseLoop(
        phase=phase,
        deadline=deadline,
        checkpoint=checkpoint,
        levels=config["levels"],
        n_envs=train_config["n_envs"],
        total_timesteps=train_config["total_timesteps"],
        chunk_timesteps=train_config["chunk_timesteps"],
        diagnostic_episodes=3,
        diagnostic_seed=evaluation_config["seed"],
        train_chunk=lambda **kwargs: _train_shared_chunk(
            phase=phase,
            config=config,
            config_path=Path(config_path),
            phase_resolved_config=phase_resolved_config,
            run_name_prefix=run_name,
            budget_ledger_snapshot=budget_ledger_snapshot,
            checkpoint_vecnormalize=checkpoint_vecnormalize,
            repository_root=repository_root,
            **kwargs,
        ),
        diagnose=lambda **kwargs: _diagnose_shared_checkpoint(
            repository_root=repository_root,
            **kwargs,
        ),
        checkpoint_timesteps=_checkpoint_timesteps,
        now=_utc_now,
        report_dir=(
            repository_root / "models" / run_name / "diagnostics"
        ),
        lineage_store=lineage_store,
        repository_root=repository_root,
        run_name=run_name,
    )
    return loop.run()


class _ExactEnvironmentStepCallback(BaseCallback):
    def __init__(self, target_timesteps: int) -> None:
        super().__init__()
        self.target_timesteps = target_timesteps

    def _on_step(self) -> bool:
        return self.model.num_timesteps < self.target_timesteps


class ProcessTreePeakRss:
    """Sample simultaneous Linux RSS for one process and all descendants."""

    def __init__(
        self,
        *,
        root_pid: int,
        proc_root: Path = Path("/proc"),
        interval_seconds: float = 0.05,
    ) -> None:
        if (
            isinstance(root_pid, bool)
            or not isinstance(root_pid, int)
            or root_pid <= 0
        ):
            raise ValueError("root_pid must be a positive integer")
        if (
            isinstance(interval_seconds, bool)
            or not isinstance(interval_seconds, (int, float))
            or not math.isfinite(interval_seconds)
            or interval_seconds <= 0
        ):
            raise ValueError(
                "RSS sample interval must be positive and finite"
            )
        self.root_pid = root_pid
        self.proc_root = Path(proc_root)
        self.interval_seconds = float(interval_seconds)
        self.peak_bytes = 0
        self.sample_count = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    def _snapshot(self) -> dict[int, tuple[int, int]]:
        if not self.proc_root.is_dir():
            raise RuntimeError(
                "Linux /proc is required for process-tree RSS measurement"
            )
        processes: dict[int, tuple[int, int]] = {}
        try:
            entries = tuple(self.proc_root.iterdir())
        except OSError as error:
            raise RuntimeError("could not enumerate Linux /proc") from error
        for entry in entries:
            if not entry.name.isdigit():
                continue
            try:
                lines = (entry / "status").read_text(
                    encoding="utf-8"
                ).splitlines()
                fields = {}
                for line in lines:
                    if ":" in line:
                        name, value = line.split(":", 1)
                        fields[name] = value.strip()
                pid = int(fields["Pid"])
                ppid = int(fields["PPid"])
                rss_parts = fields["VmRSS"].split()
                if (
                    len(rss_parts) != 2
                    or rss_parts[1] != "kB"
                ):
                    raise ValueError
                rss_bytes = int(rss_parts[0]) * 1024
                if pid < 1 or ppid < 0 or rss_bytes < 0:
                    raise ValueError
            except (KeyError, OSError, UnicodeError, ValueError):
                # Processes can vanish or replace their status between reads.
                continue
            processes[pid] = (ppid, rss_bytes)
        if self.root_pid not in processes:
            raise RuntimeError(
                "benchmark root process is absent from Linux /proc"
            )
        return processes

    def sample(self) -> int:
        processes = self._snapshot()
        descendants = {self.root_pid}
        changed = True
        while changed:
            changed = False
            for pid, (ppid, _rss) in processes.items():
                if pid not in descendants and ppid in descendants:
                    descendants.add(pid)
                    changed = True
        aggregate = sum(
            processes[pid][1]
            for pid in descendants
            if pid in processes
        )
        with self._lock:
            self.sample_count += 1
            self.peak_bytes = max(self.peak_bytes, aggregate)
            return self.peak_bytes

    def _sample_until_stopped(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                self.sample()
            except BaseException as error:
                self._error = error
                self._stop.set()
                return

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("RSS sampler is already started")
        self.sample()
        self._thread = threading.Thread(
            target=self._sample_until_stopped,
            name="benchmark-process-tree-rss",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join()
        if self._error is not None:
            raise RuntimeError(
                "process-tree RSS sampling failed"
            ) from self._error
        self.sample()
        if self.sample_count < 1:
            raise RuntimeError(
                "process-tree RSS sampling produced no evidence"
            )


def _run_benchmark_workload(
    environment_steps: int,
) -> tuple[Decimal, float]:
    if (
        isinstance(environment_steps, bool)
        or not isinstance(environment_steps, int)
        or environment_steps <= 0
        or environment_steps % _BENCHMARK_WORKERS
    ):
        raise ValueError(
            "benchmark environment steps must be positive and divisible by 16"
        )
    overrides = argparse.Namespace(
        phase_resolved_config=False,
        levels=None,
        timesteps=environment_steps,
        n_envs=_BENCHMARK_WORKERS,
        lr=None,
        ent_coef=None,
        level_weights_json=None,
    )
    config = training.load_training_config(
        str(CONFIG_PATH), "phase_1", overrides
    )
    arguments = argparse.Namespace(
        resume=None,
        init_from=None,
        run_name="all32-benchmark",
        start_snapshots=None,
        curriculum_threshold=0.5,
    )
    device = training.resolve_device(config["train"]["device"])
    environment = training.build_training_env(config, arguments)
    sampler: ProcessTreePeakRss | None = None
    try:
        model = training.create_model(
            config, arguments, environment, device
        )
        sampler = ProcessTreePeakRss(root_pid=os.getpid())
        sampler.start()
        started = time.monotonic()
        with redirect_stdout(sys.stderr):
            model.learn(
                total_timesteps=environment_steps,
                callback=_ExactEnvironmentStepCallback(
                    environment_steps
                ),
                reset_num_timesteps=True,
            )
        elapsed_seconds = Decimal(str(time.monotonic() - started))
        if model.num_timesteps != environment_steps:
            raise RuntimeError(
                "benchmark did not stop at the exact environment-step target"
            )
    finally:
        if sampler is not None:
            sampler.stop()
        environment.close()
    peak_rss_gb = sampler.peak_bytes / float(1024**3)
    return elapsed_seconds, peak_rss_gb


def _positive_integer(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "must be a positive integer"
        ) from error
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def main(argv=None, *, stdout=None) -> int:
    parser = argparse.ArgumentParser(
        description="Shared-policy phase and benchmark worker"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument(
        "--environment-steps",
        required=True,
        type=_positive_integer,
    )
    phase_worker = subparsers.add_parser("phase")
    phase_worker.add_argument(
        "--phase", required=True, choices=("phase_1", "phase_2")
    )
    phase_worker.add_argument("--run-name", required=True)
    phase_worker.add_argument("--config", required=True, type=Path)
    phase_worker.add_argument(
        "--deadline-epoch", required=True, type=_positive_integer
    )
    phase_worker.add_argument(
        "--budget-ledger-snapshot", required=True, type=Path
    )
    phase_worker.add_argument("--lineage", type=Path)
    phase_worker.add_argument("--resume", type=Path)
    phase_worker.add_argument("--resume-vecnormalize", type=Path)
    phase_worker.add_argument(
        "--phase-resolved-config", action="store_true"
    )
    args = parser.parse_args(argv)
    if args.command == "benchmark":
        elapsed_seconds, peak_rss_gb = _run_benchmark_workload(
            args.environment_steps
        )
        destination = sys.stdout if stdout is None else stdout
        json.dump(
            {
                "environment_steps": args.environment_steps,
                "elapsed_seconds": str(elapsed_seconds),
                "peak_rss_gb": peak_rss_gb,
            },
            destination,
            sort_keys=True,
        )
        destination.write("\n")
        return 0
    if args.command == "phase":
        phase_kwargs = {
            "config_path": args.config,
            "run_name": args.run_name,
            "budget_ledger_snapshot": args.budget_ledger_snapshot,
            "lineage_path": args.lineage,
        }
        if args.phase_resolved_config:
            phase_kwargs["phase_resolved_config"] = True
        if args.resume_vecnormalize is not None:
            phase_kwargs["checkpoint_vecnormalize"] = (
                args.resume_vecnormalize
            )
        best_checkpoint = run_phase(
            args.phase,
            datetime.fromtimestamp(
                args.deadline_epoch, tz=timezone.utc
            ),
            args.resume,
            **phase_kwargs,
        )
        destination = sys.stdout if stdout is None else stdout
        json.dump(
            {
                "best_checkpoint": str(best_checkpoint),
                "phase": args.phase,
                "run_name": args.run_name,
            },
            destination,
            sort_keys=True,
        )
        destination.write("\n")
        return 0
    raise RuntimeError(f"unsupported command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
