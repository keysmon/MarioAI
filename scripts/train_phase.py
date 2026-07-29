#!/usr/bin/env python
"""Chunked shared-policy training with diagnostic-only promotion evidence."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from contextlib import redirect_stdout
from datetime import datetime, timezone
from decimal import Decimal
import json
import os
from pathlib import Path
import resource
import sys
import time

import marioai.evaluate as evaluation
import marioai.train as training
from stable_baselines3.common.callbacks import BaseCallback

from marioai.results import (
    EvaluationReport,
    is_better_checkpoint,
    sha256_file,
)
from marioai.sampling import assign_worker_levels, regression_weights


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "all32.yaml"
REPORT_ROOT = REPOSITORY_ROOT / "reports" / "diagnostics"
_BENCHMARK_WORKERS = 16


class PhaseLoop:
    """Train one shared policy in fixed chunks and retain only improvements."""

    def __init__(
        self,
        *,
        phase: str,
        deadline: datetime,
        checkpoint: Path | None,
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
    ) -> None:
        self.phase = phase
        self.deadline = deadline
        self.checkpoint = Path(checkpoint) if checkpoint is not None else None
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
        self.next_weights = {level: 1.0 for level in self.levels}
        self.next_worker_levels = assign_worker_levels(
            self.levels,
            self.n_envs,
            self.next_weights,
        )

    def run(self) -> Path:
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
    checkpoint: Path | None,
    target_timesteps: int,
    level_weights: Mapping[str, float],
) -> Path:
    run_name = f"all32-{phase}-chunk-{target_timesteps:012d}"
    arguments = [
        "--config",
        str(CONFIG_PATH),
        "--phase",
        phase,
        "--timesteps",
        str(target_timesteps),
        "--run-name",
        run_name,
        "--level-weights-json",
        json.dumps(dict(level_weights), sort_keys=True, separators=(",", ":")),
    ]
    if checkpoint is not None:
        arguments.extend(["--resume", str(checkpoint)])
    ledger_snapshot = os.environ.get("MARIOAI_BUDGET_LEDGER_SNAPSHOT")
    if ledger_snapshot:
        arguments.extend(
            ["--budget-ledger-snapshot", ledger_snapshot]
        )
    training.main(arguments)
    candidate = Path("models") / run_name / "final.zip"
    if not candidate.is_file():
        raise RuntimeError(
            f"training did not produce the shared checkpoint {candidate}"
        )
    return candidate


def _diagnose_shared_checkpoint(
    *,
    checkpoint: Path,
    active_levels: Sequence[str],
    episodes: int,
    seed: int,
    deterministic: bool,
) -> EvaluationReport:
    return evaluation.evaluate_checkpoint(
        checkpoint,
        active_levels,
        episodes=episodes,
        seed=seed,
        deterministic=deterministic,
    )


def run_phase(
    phase: str,
    deadline: datetime,
    checkpoint: Path | None,
) -> Path:
    """Run fixed environment-step chunks until the target or remote deadline."""
    if deadline.tzinfo is None or deadline.utcoffset() is None:
        raise ValueError("deadline must be timezone-aware")
    overrides = argparse.Namespace(
        phase_resolved_config=False,
        levels=None,
        timesteps=None,
        n_envs=None,
        lr=None,
        ent_coef=None,
        level_weights_json=None,
    )
    config = training.load_training_config(
        str(CONFIG_PATH), phase, overrides
    )
    train_config = config["train"]
    evaluation_config = config["evaluation"]
    if evaluation_config.get("diagnostic_episodes") != 3:
        raise ValueError(
            "all-32 phase diagnostics must use exactly three rollouts"
        )
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
            **kwargs,
        ),
        diagnose=_diagnose_shared_checkpoint,
        checkpoint_timesteps=_checkpoint_timesteps,
        now=_utc_now,
        report_dir=REPORT_ROOT / phase,
    )
    return loop.run()


class _ExactEnvironmentStepCallback(BaseCallback):
    def __init__(self, target_timesteps: int) -> None:
        super().__init__()
        self.target_timesteps = target_timesteps

    def _on_step(self) -> bool:
        return self.model.num_timesteps < self.target_timesteps


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
    try:
        model = training.create_model(
            config, arguments, environment, device
        )
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
        environment.close()
    self_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    child_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    peak_rss_gb = float(self_rss + child_rss) / (1024 * 1024)
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
    raise RuntimeError(f"unsupported command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
