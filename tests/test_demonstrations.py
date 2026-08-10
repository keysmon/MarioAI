"""Shared-policy demonstration loading and compatibility checks."""
from pathlib import Path

import numpy as np
import pytest

from marioai.curriculum import save_route
from marioai.demonstrations import load_demonstrations


def _save_route(
    route_dir: Path,
    *,
    level: str,
    actions: list[int],
    action_set: str = "complex",
    decision_skip: int = 4,
    **extra,
) -> None:
    save_route(
        {
            "level": level,
            "action_set": action_set,
            "decision_skip": decision_skip,
            "actions": actions,
            "waypoints": [],
            **extra,
        },
        route_dir,
    )


def test_route_rejects_mismatched_action_set_or_skip(tmp_path):
    _save_route(
        tmp_path,
        level="2-2",
        action_set="simple",
        actions=[1, 1, 1, 1],
    )

    with pytest.raises(ValueError, match="action set"):
        load_demonstrations([tmp_path], action_set="complex", skip=4)

    _save_route(
        tmp_path,
        level="2-2",
        decision_skip=1,
        actions=[1, 1, 1, 1],
    )
    with pytest.raises(ValueError, match="decision skip"):
        load_demonstrations([tmp_path], action_set="complex", skip=4)


@pytest.mark.parametrize("actions", [[12] * 4, [-1] * 4])
def test_route_rejects_actions_outside_requested_action_set(tmp_path, actions):
    _save_route(tmp_path, level="2-2", actions=actions)

    with pytest.raises(ValueError, match="action.*range"):
        load_demonstrations([tmp_path], action_set="complex", skip=4)


def test_route_rejects_partial_solver_output_before_replay(tmp_path):
    _save_route(tmp_path, level="2-2", actions=[1] * 4, partial=True)

    with pytest.raises(ValueError, match="partial"):
        load_demonstrations([tmp_path], action_set="complex", skip=4)


class _ClearingVecEnv:
    def __init__(self, decisions):
        self._decisions = decisions
        self._steps = 0

    def reset(self):
        return np.zeros((1, 84, 84, 4), dtype=np.uint8)

    def step(self, action):
        self._steps += 1
        cleared = self._steps == self._decisions
        obs = np.full((1, 84, 84, 4), self._steps, dtype=np.uint8)
        return (
            obs,
            np.zeros(1),
            np.array([cleared]),
            [{"flag_get": cleared, "x_pos": self._steps * 10}],
        )

    def close(self):
        pass


def test_shared_batch_gives_each_level_equal_total_sample_weight(
    tmp_path, monkeypatch
):
    short = tmp_path / "short"
    long = tmp_path / "long"
    _save_route(short, level="1-1", actions=[3] * 4)
    _save_route(long, level="1-2", actions=[3] * 12)
    decisions = {"1-1": 1, "1-2": 3}

    monkeypatch.setattr(
        "marioai.demonstrations._make_vec_env",
        lambda level, **kwargs: _ClearingVecEnv(decisions[level]),
    )

    batch = load_demonstrations(
        [short, long], action_set="complex", skip=4
    )

    assert batch.observations.shape == (4, 84, 84, 4)
    assert batch.actions.tolist() == [3, 3, 3, 3]
    assert batch.levels.tolist() == ["1-1", "1-2", "1-2", "1-2"]
    by_level = {
        level: batch.sample_weights[batch.levels == level].sum()
        for level in set(batch.levels)
    }
    assert by_level["1-1"] == pytest.approx(by_level["1-2"])


def test_shared_loader_requires_a_completed_terminal_clear(
    tmp_path, monkeypatch
):
    _save_route(tmp_path, level="1-1", actions=[3] * 4)

    class _NonTerminalVecEnv(_ClearingVecEnv):
        def step(self, action):
            obs, rewards, _, infos = super().step(action)
            infos[0]["flag_get"] = False
            return obs, rewards, np.array([False]), infos

    monkeypatch.setattr(
        "marioai.demonstrations._make_vec_env",
        lambda level, **kwargs: _NonTerminalVecEnv(1),
    )

    with pytest.raises(RuntimeError, match="did not clear"):
        load_demonstrations([tmp_path], action_set="complex", skip=4)
