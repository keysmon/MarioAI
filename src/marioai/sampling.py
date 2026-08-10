"""Deterministic fixed-worker allocation for multi-stage training."""
from collections.abc import Mapping, Sequence
import math

from .levels import validate_levels


def assign_worker_levels(
    levels: Sequence[str],
    n_envs: int,
    weights: Mapping[str, float] | None = None,
) -> tuple[str, ...]:
    """Assign each worker a stage, with one worker reserved per stage."""
    levels = validate_levels(levels)
    if n_envs < len(levels):
        raise ValueError("n_envs must be at least the number of active levels")

    weights = weights or {}
    unknown = set(weights) - set(levels)
    if unknown:
        stages = ", ".join(sorted(unknown))
        raise ValueError(f"weights contain unknown Mario stage(s): {stages}")

    resolved = {level: float(weights.get(level, 1.0)) for level in levels}
    invalid = [
        level
        for level, weight in resolved.items()
        if not math.isfinite(weight) or weight <= 0
    ]
    if invalid:
        raise ValueError("level weights must be finite positive values")

    counts = {level: 1 for level in levels}
    remaining = n_envs - len(levels)
    total_weight = sum(resolved.values())
    quotas = {level: remaining * resolved[level] / total_weight for level in levels}
    for level in levels:
        counts[level] += int(quotas[level])

    leftovers = n_envs - sum(counts.values())
    level_index = {level: index for index, level in enumerate(levels)}
    ranked = sorted(
        levels,
        key=lambda level: (-(quotas[level] % 1), level_index[level]),
    )
    for level in ranked[:leftovers]:
        counts[level] += 1

    return tuple(level for level in levels for _ in range(counts[level]))


def regression_weights(
    levels: Sequence[str],
    previous: Mapping[str, bool],
    current: Mapping[str, bool],
    multiplier: float = 2.0,
) -> dict[str, float]:
    """Prioritize stages that regressed from passing to failing."""
    levels = validate_levels(levels)
    multiplier = float(multiplier)
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("multiplier must be a finite positive value")
    return {
        level: (
            multiplier
            if previous.get(level, False) and not current.get(level, False)
            else 1.0
        )
        for level in levels
    }
