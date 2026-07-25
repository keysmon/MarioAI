"""Durable evaluation evidence and checkpoint acceptance rules."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from marioai.levels import ALL_LEVELS


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

    @property
    def passed(self) -> bool:
        if self.deterministic or self.requested_episodes != 15:
            return False
        if set(self.stages) != set(ALL_LEVELS):
            return False
        return all(
            stage.get("passed") is True
            and stage.get("clears", 0) >= 1
            and stage.get("episodes") == 15
            for stage in self.stages.values()
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_sha256": self.checkpoint_sha256,
            "deterministic": self.deterministic,
            "requested_episodes": self.requested_episodes,
            "stages": self.stages,
            "rollouts": [rollout.to_dict() for rollout in self.rollouts],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> EvaluationReport:
        return cls(
            checkpoint_sha256=value["checkpoint_sha256"],
            deterministic=value["deterministic"],
            requested_episodes=value["requested_episodes"],
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

    def write(self, path: Path) -> None:
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
            temporary_path.replace(path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()


def checkpoint_score(report: EvaluationReport) -> tuple[int, int, float]:
    """Rank a checkpoint by stage coverage, then clears, then progress."""
    coverage = sum(stage.get("passed") is True for stage in report.stages.values())
    clears = sum(int(stage.get("clears", 0)) for stage in report.stages.values())
    if report.rollouts:
        progress = sum(rollout.max_x for rollout in report.rollouts) / len(
            report.rollouts
        )
    else:
        progress = sum(
            float(stage.get("mean_max_x", 0.0))
            for stage in report.stages.values()
        )
    return coverage, clears, progress


def is_better_checkpoint(
    candidate: EvaluationReport, incumbent: EvaluationReport | None
) -> bool:
    return incumbent is None or checkpoint_score(candidate) > checkpoint_score(
        incumbent
    )
