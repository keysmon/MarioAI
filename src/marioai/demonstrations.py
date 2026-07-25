"""Validated, policy-cadence demonstrations for shared-policy recovery."""
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from .actions import resolve_action_set
from .curriculum import load_route
from .envs import make_mario_env
from .levels import validate_levels


@dataclass(frozen=True)
class DemonstrationBatch:
    """Concatenated route observations with level-balanced sample weights."""

    observations: np.ndarray
    actions: np.ndarray
    sample_weights: np.ndarray
    levels: np.ndarray
    route_results: tuple[dict[str, object], ...]

    @property
    def labels(self) -> np.ndarray:
        """Compatibility alias for callers that call action targets labels."""
        return self.actions

    @property
    def weights(self) -> np.ndarray:
        """Short alias for level-balanced sample weights."""
        return self.sample_weights


def _make_vec_env(
    level: str,
    *,
    action_set: str,
    skip: int = 4,
    shape: int = 84,
    frame_stack: int = 4,
):
    return VecFrameStack(
        DummyVecEnv(
            [
                lambda: make_mario_env(
                    level=level,
                    action_set=action_set,
                    skip=skip,
                    shape=shape,
                )
            ]
        ),
        n_stack=frame_stack,
        channels_order="last",
    )


def _decision_actions(actions: Sequence[int], skip: int) -> list[int]:
    if skip < 1:
        raise ValueError("skip must be >= 1")
    if len(actions) % skip:
        raise ValueError(
            f"route has {len(actions)} frames, not divisible by skip={skip}"
        )
    decisions = []
    for frame in range(0, len(actions), skip):
        block = actions[frame : frame + skip]
        if len(set(block)) != 1:
            raise ValueError(
                f"route action block at frame {frame} is not constant: "
                f"{block}"
            )
        decisions.append(block[0])
    return decisions


def _validate_route(
    route: dict,
    route_dir: Path,
    *,
    action_set: str,
    skip: int,
    allow_legacy_metadata: bool,
) -> list[int]:
    validate_levels((route["level"],))
    expected_actions = resolve_action_set(action_set)
    stored_action_set = route.get("action_set")
    stored_skip = route.get("decision_skip")
    is_legacy = stored_action_set is None and stored_skip is None
    if is_legacy and allow_legacy_metadata and action_set == "simple":
        stored_action_set = "simple"
        stored_skip = skip
    if stored_action_set != action_set:
        if stored_action_set is None:
            detail = "missing (legacy routes imply 'simple')"
        else:
            detail = repr(stored_action_set)
        raise ValueError(
            f"route {route_dir} action set is {detail}, requested "
            f"{action_set!r}"
        )
    if (
        isinstance(stored_skip, bool)
        or not isinstance(stored_skip, int)
        or stored_skip != skip
    ):
        raise ValueError(
            f"route {route_dir} decision skip is {stored_skip!r}, "
            f"requested {skip}"
        )
    if route.get("partial", False):
        raise ValueError(f"route {route_dir} is partial, not completed")

    actions = route["actions"]
    if not isinstance(actions, list):
        raise ValueError(f"route {route_dir} actions must be a JSON array")
    action_count = len(expected_actions)
    for frame, action in enumerate(actions):
        if (
            isinstance(action, bool)
            or not isinstance(action, int)
            or not 0 <= action < action_count
        ):
            raise ValueError(
                f"route {route_dir} action {action!r} at frame {frame} "
                f"is outside action range [0, {action_count})"
            )
    return _decision_actions(actions, skip)


def _collect_validated_route(
    route: dict,
    decisions: Sequence[int],
    *,
    action_set: str,
    skip: int,
    shape: int,
    frame_stack: int,
    require_clear: bool,
):
    level = route["level"]
    env = _make_vec_env(
        level,
        action_set=action_set,
        skip=skip,
        shape=shape,
        frame_stack=frame_stack,
    )
    obs = env.reset()
    observations = []
    labels = []
    cleared = False
    terminal = False
    max_x = 0
    try:
        for action in decisions:
            observations.append(np.asarray(obs[0]).copy())
            labels.append(action)
            obs, _, dones, infos = env.step(
                np.asarray([action], dtype=np.int64)
            )
            info = infos[0]
            cleared = cleared or bool(info.get("flag_get", False))
            max_x = max(max_x, int(info.get("x_pos", 0)))
            terminal = bool(dones[0])
            if terminal:
                break
    finally:
        env.close()

    result = {
        "level": level,
        "cleared": cleared,
        "terminal": terminal,
        "max_x": max_x,
        "decisions": len(labels),
    }
    if len(labels) != len(decisions):
        raise RuntimeError(
            f"route for {level} terminated after {len(labels)}/"
            f"{len(decisions)} decisions"
        )
    if require_clear and not (cleared and terminal):
        raise RuntimeError(
            f"aligned route for {level} did not clear at a completed "
            f"terminal state: decisions={len(labels)}/{len(decisions)} "
            f"max_x={max_x}"
        )
    return (
        np.asarray(observations, dtype=np.uint8),
        np.asarray(labels, dtype=np.int64),
        result,
    )


def collect_demonstration(
    route_dir: Path,
    level: str,
    skip: int = 4,
    shape: int = 84,
    frame_stack: int = 4,
    require_clear: bool = True,
    action_set: str = "simple",
):
    """Replay one route through the exact policy observation pipeline.

    Metadata-free routes are retained only for the legacy simple-action
    single-route workflow. Shared complex-action loading is always explicit.
    """
    route_dir = Path(route_dir)
    route = load_route(route_dir)
    if route["level"] != level:
        raise ValueError(
            f"route is for level {route['level']!r}, requested {level!r}"
        )
    decisions = _validate_route(
        route,
        route_dir,
        action_set=action_set,
        skip=skip,
        allow_legacy_metadata=True,
    )
    return _collect_validated_route(
        route,
        decisions,
        action_set=action_set,
        skip=skip,
        shape=shape,
        frame_stack=frame_stack,
        require_clear=require_clear,
    )


def load_demonstrations(
    route_dirs: Sequence[Path], action_set: str, skip: int
) -> DemonstrationBatch:
    """Validate and replay solved routes into one level-balanced batch."""
    route_dirs = tuple(Path(route_dir) for route_dir in route_dirs)
    if not route_dirs:
        raise ValueError("at least one route directory is required")
    resolve_action_set(action_set)

    validated = []
    for route_dir in route_dirs:
        route = load_route(route_dir)
        decisions = _validate_route(
            route,
            route_dir,
            action_set=action_set,
            skip=skip,
            allow_legacy_metadata=False,
        )
        validated.append((route, decisions))

    collected = [
        _collect_validated_route(
            route,
            decisions,
            action_set=action_set,
            skip=skip,
            shape=84,
            frame_stack=4,
            require_clear=True,
        )
        for route, decisions in validated
    ]
    observations = np.concatenate([item[0] for item in collected], axis=0)
    actions = np.concatenate([item[1] for item in collected], axis=0)
    levels = np.concatenate(
        [
            np.full(len(item[1]), route["level"], dtype=object)
            for (route, _), item in zip(validated, collected, strict=True)
        ]
    )
    level_counts = Counter(levels)
    sample_weights = np.asarray(
        [1.0 / level_counts[level] for level in levels],
        dtype=np.float32,
    )
    return DemonstrationBatch(
        observations=observations,
        actions=actions,
        sample_weights=sample_weights,
        levels=levels,
        route_results=tuple(item[2] for item in collected),
    )
