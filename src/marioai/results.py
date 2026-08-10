"""Durable evaluation evidence and checkpoint acceptance rules."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import string
import tempfile
from typing import Any

from marioai.levels import ALL_LEVELS

ACCEPTANCE_POLICY_MODE = "shared_complex_impala"
LEGACY_DIAGNOSTIC_POLICY_MODE = "legacy_simple_nature_diagnostic"


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file's exact bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class RolloutResult:
    level: str
    seed: int
    cleared: bool
    terminal_cause: str
    max_x: int
    reward: float
    steps: int
    wall_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RolloutResult:
        return cls(**value)


def _is_integer(value: Any) -> bool:
    return type(value) is int


def _is_finite_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _is_valid_rollout(rollout: Any, stages: dict) -> bool:
    return (
        isinstance(rollout, RolloutResult)
        and type(rollout.level) is str
        and rollout.level in stages
        and _is_integer(rollout.seed)
        and type(rollout.cleared) is bool
        and type(rollout.terminal_cause) is str
        and rollout.terminal_cause in {"flag", "timeout", "time", "death"}
        and rollout.cleared == (rollout.terminal_cause == "flag")
        and _is_integer(rollout.max_x)
        and rollout.max_x >= 0
        and _is_finite_number(rollout.reward)
        and _is_integer(rollout.steps)
        and rollout.steps > 0
        and _is_finite_number(rollout.wall_seconds)
        and rollout.wall_seconds >= 0
    )


def _is_valid_stage_summary(stage: Any) -> bool:
    if not isinstance(stage, dict):
        return False
    if (
        type(stage.get("passed")) is not bool
        or not _is_integer(stage.get("clears"))
        or not _is_integer(stage.get("episodes"))
    ):
        return False
    optional_numbers = ("clear_rate", "mean_max_x", "mean_reward")
    return all(
        field not in stage or _is_finite_number(stage[field])
        for field in optional_numbers
    )


def summarize_stage(
    level: str, rollouts: list[RolloutResult]
) -> dict[str, bool | int | float]:
    """Summarize one stage without discarding its underlying rollout evidence."""
    if any(rollout.level != level for rollout in rollouts):
        raise ValueError(f"rollout does not belong to stage {level}")

    episodes = len(rollouts)
    clears = sum(rollout.cleared for rollout in rollouts)
    denominator = episodes or 1
    return {
        "passed": clears >= 1,
        "clears": clears,
        "episodes": episodes,
        "clear_rate": clears / denominator,
        "mean_max_x": sum(rollout.max_x for rollout in rollouts) / denominator,
        "mean_reward": sum(rollout.reward for rollout in rollouts) / denominator,
    }


@dataclass(frozen=True)
class EvaluationReport:
    checkpoint_sha256: str
    deterministic: bool
    requested_episodes: int
    stages: dict[str, dict[str, bool | int | float]]
    rollouts: list[RolloutResult]
    policy_mode: str = ACCEPTANCE_POLICY_MODE

    def _has_valid_schema(self) -> bool:
        return (
            type(self.checkpoint_sha256) is str
            and len(self.checkpoint_sha256) == 64
            and all(
                character in string.hexdigits
                for character in self.checkpoint_sha256
            )
            and type(self.deterministic) is bool
            and _is_integer(self.requested_episodes)
            and type(self.policy_mode) is str
            and self.policy_mode
            in {
                ACCEPTANCE_POLICY_MODE,
                LEGACY_DIAGNOSTIC_POLICY_MODE,
            }
            and isinstance(self.stages, dict)
            and all(
                type(level) is str and _is_valid_stage_summary(stage)
                for level, stage in self.stages.items()
            )
            and type(self.rollouts) is list
            and all(
                _is_valid_rollout(rollout, self.stages)
                for rollout in self.rollouts
            )
        )

    @property
    def passed(self) -> bool:
        if not self._has_valid_schema():
            return False
        if self.policy_mode != ACCEPTANCE_POLICY_MODE:
            return False
        if self.deterministic or self.requested_episodes != 15:
            return False
        if set(self.stages) != set(ALL_LEVELS):
            return False
        if len(self.rollouts) != len(ALL_LEVELS) * 15:
            return False
        seeds = [rollout.seed for rollout in self.rollouts]
        if len(set(seeds)) != len(seeds):
            return False

        by_level = {level: [] for level in ALL_LEVELS}
        for rollout in self.rollouts:
            by_level[rollout.level].append(rollout)
        for level in ALL_LEVELS:
            stage_rollouts = by_level[level]
            if len(stage_rollouts) != 15:
                return False
            recomputed = summarize_stage(level, stage_rollouts)
            if self.stages[level] != recomputed:
                return False
            if recomputed["passed"] is not True:
                return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_sha256": self.checkpoint_sha256,
            "deterministic": self.deterministic,
            "requested_episodes": self.requested_episodes,
            "policy_mode": self.policy_mode,
            "stages": self.stages,
            "rollouts": [rollout.to_dict() for rollout in self.rollouts],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> EvaluationReport:
        return cls(
            checkpoint_sha256=value["checkpoint_sha256"],
            deterministic=value["deterministic"],
            requested_episodes=value["requested_episodes"],
            policy_mode=value.get(
                "policy_mode",
                LEGACY_DIAGNOSTIC_POLICY_MODE,
            ),
            stages=value["stages"],
            rollouts=[
                RolloutResult.from_dict(rollout)
                for rollout in value["rollouts"]
            ],
        )

    @classmethod
    def read(cls, path: Path) -> EvaluationReport:
        with path.open(encoding="utf-8") as source:
            return cls.from_dict(json.load(source))

    def write(self, path: Path, *, overwrite: bool = False) -> None:
        """Atomically write a report without clobbering evidence by default."""
        path = Path(path)
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"evaluation report already exists at {path}; pass "
                "overwrite=True to replace it explicitly"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as destination:
                temporary_path = Path(destination.name)
                json.dump(self.to_dict(), destination, indent=2)
                destination.write("\n")
                destination.flush()
                os.fsync(destination.fileno())
            if overwrite:
                temporary_path.replace(path)
            else:
                try:
                    os.link(temporary_path, path)
                except FileExistsError as exc:
                    raise FileExistsError(
                        f"evaluation report already exists at {path}; pass "
                        "overwrite=True to replace it explicitly"
                    ) from exc
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()


def checkpoint_score(report: EvaluationReport) -> tuple[int, int, float]:
    """Rank a checkpoint by stage coverage, then clears, then progress."""
    coverage = sum(stage.get("passed") is True for stage in report.stages.values())
    clears = sum(int(stage.get("clears", 0)) for stage in report.stages.values())
    progress_totals: dict[str, tuple[int, int]] = {}
    for rollout in report.rollouts:
        total, count = progress_totals.get(rollout.level, (0, 0))
        progress_totals[rollout.level] = (
            total + rollout.max_x,
            count + 1,
        )
    progress = sum(
        (
            progress_totals[level][0] / progress_totals[level][1]
            if level in progress_totals
            else float(stage.get("mean_max_x", 0.0))
        )
        for level, stage in report.stages.items()
    )
    return coverage, clears, progress


def is_better_checkpoint(
    candidate: EvaluationReport, incumbent: EvaluationReport | None
) -> bool:
    return incumbent is None or checkpoint_score(candidate) > checkpoint_score(
        incumbent
    )
