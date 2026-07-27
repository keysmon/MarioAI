"""Evaluate one checkpoint and emit immutable stochastic evidence."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import time

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from marioai.actions import action_set_size
from marioai.envs import make_mario_env
from marioai.levels import ALL_LEVELS, validate_levels
from marioai.results import (
    EvaluationReport,
    LEGACY_DIAGNOSTIC_POLICY_MODE,
    RolloutResult,
    ACCEPTANCE_POLICY_MODE,
    sha256_file,
    summarize_stage,
)
from marioai.train import validate_resume_model


def _make_evaluation_env(
    *,
    level: str,
    frame_stack: int,
    skip: int,
    shape: int,
    action_set: str = "complex",
):
    environment = DummyVecEnv(
        [
            lambda: make_mario_env(
                level=level,
                skip=skip,
                shape=shape,
                action_set=action_set,
            )
        ]
    )
    return VecFrameStack(
        environment,
        n_stack=frame_stack,
        channels_order="last",
    )


def evaluate_rollout(
    model,
    level: str,
    seed: int,
    deterministic: bool,
    *,
    max_steps: int = 3000,
    frame_stack: int = 4,
    skip: int = 4,
    shape: int = 84,
    action_set: str = "complex",
) -> RolloutResult:
    """Run one independently seeded rollout on one Mario stage."""
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    environment = _make_evaluation_env(
        level=level,
        frame_stack=frame_stack,
        skip=skip,
        shape=shape,
        action_set=action_set,
    )
    started = time.monotonic()
    try:
        environment.seed(seed)
        model.set_random_seed(seed)
        observation = environment.reset()

        cleared = False
        max_x = 0
        reward_total = 0.0
        last_info: dict = {}
        steps = 0
        for steps in range(1, max_steps + 1):
            action, _ = model.predict(
                observation,
                deterministic=deterministic,
            )
            observation, rewards, dones, infos = environment.step(action)
            last_info = infos[0]
            reward_total += float(rewards[0])
            cleared = cleared or bool(last_info.get("flag_get", False))
            max_x = max(max_x, int(last_info.get("x_pos", 0)))
            if bool(dones[0]):
                break

        if cleared:
            terminal_cause = "flag"
        elif steps >= max_steps:
            terminal_cause = "timeout"
        elif last_info.get("time") == 0:
            terminal_cause = "time"
        else:
            terminal_cause = "death"

        return RolloutResult(
            level=level,
            seed=seed,
            cleared=cleared,
            terminal_cause=terminal_cause,
            max_x=max_x,
            reward=reward_total,
            steps=steps,
            wall_seconds=time.monotonic() - started,
        )
    finally:
        environment.close()


def evaluate_checkpoint(
    model_path: Path,
    levels: Sequence[str],
    episodes: int = 15,
    seed: int = 42000,
    deterministic: bool = False,
    legacy_diagnostic: bool = False,
) -> EvaluationReport:
    """Evaluate the same compatible checkpoint over every requested stage."""
    selected_levels = validate_levels(levels)
    if episodes <= 0:
        raise ValueError("episodes must be positive")

    checkpoint_sha256 = sha256_file(model_path)
    model = PPO.load(model_path, device="cpu")
    if legacy_diagnostic:
        action_set = "simple"
        extractor_name = "nature"
        features_dim = None
        channels = None
        policy_mode = LEGACY_DIAGNOSTIC_POLICY_MODE
    else:
        action_set = "complex"
        extractor_name = "impala"
        features_dim = 512
        channels = (16, 32, 32)
        policy_mode = ACCEPTANCE_POLICY_MODE
    validate_resume_model(
        model,
        action_count=action_set_size(action_set),
        observation_shape=(84, 84),
        frame_stack=4,
        extractor_name=extractor_name,
        features_dim=features_dim,
        channels=channels,
    )

    rollouts = []
    stages = {}
    for level_index, level in enumerate(selected_levels):
        stage_rollouts = [
            evaluate_rollout(
                model,
                level,
                seed=seed + level_index * episodes + episode_index,
                deterministic=deterministic,
                action_set=action_set,
            )
            for episode_index in range(episodes)
        ]
        rollouts.extend(stage_rollouts)
        stages[level] = summarize_stage(level, stage_rollouts)

    return EvaluationReport(
        checkpoint_sha256=checkpoint_sha256,
        deterministic=deterministic,
        requested_episodes=episodes,
        stages=stages,
        rollouts=rollouts,
        policy_mode=policy_mode,
    )


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--levels", nargs="+", required=True)
    parser.add_argument("--episodes", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42000)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace an existing evaluation report.",
    )
    parser.add_argument(
        "--legacy-diagnostic",
        action="store_true",
        help=(
            "Evaluate a seven-action NatureCNN checkpoint as explicitly "
            "non-acceptance legacy evidence."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--stochastic",
        dest="deterministic",
        action="store_false",
        help="Sample policy actions (the default and acceptance mode).",
    )
    mode.add_argument(
        "--deterministic",
        dest="deterministic",
        action="store_true",
        help="Run greedy diagnostic rollouts that cannot pass acceptance.",
    )
    parser.set_defaults(deterministic=False)
    return parser.parse_args(argv)


def _resolve_levels(levels: Sequence[str]) -> tuple[str, ...]:
    if tuple(levels) == ("all",):
        return ALL_LEVELS
    return validate_levels(levels)


def _print_summary(report: EvaluationReport) -> None:
    print(
        f"{'level':<6} {'result':<6} {'clears':>7} "
        f"{'mean_x':>10} {'mean_reward':>12}"
    )
    for level, stage in report.stages.items():
        result = "pass" if stage["passed"] else "fail"
        clears = f"{stage['clears']}/{stage['episodes']}"
        print(
            f"{level:<6} {result:<6} {clears:>7} "
            f"{stage['mean_max_x']:>10.1f} {stage['mean_reward']:>12.1f}"
        )
    print("PASS" if report.passed else "FAIL")


def main(argv=None) -> EvaluationReport:
    args = parse_args(argv)
    try:
        levels = _resolve_levels(args.levels)
        report = evaluate_checkpoint(
            args.model,
            levels,
            episodes=args.episodes,
            seed=args.seed,
            deterministic=args.deterministic,
            legacy_diagnostic=args.legacy_diagnostic,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    report.write(args.out, overwrite=args.overwrite)
    _print_summary(report)
    return report


if __name__ == "__main__":
    main()
