"""Tests for policy-cadence behavior-cloning demonstrations."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from marioai.curriculum import save_route


def _load_bc_module():
    path = Path(__file__).parents[1] / "scripts" / "behavior_clone_route.py"
    spec = importlib.util.spec_from_file_location("behavior_clone_route", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _save_actions(tmp_path, actions):
    save_route(
        {
            "level": "1-1",
            "actions": actions,
            "waypoints": [{"index": 0, "frame": 0, "x_pos": 40}],
        },
        tmp_path,
    )


def test_collect_demonstration_emits_one_sample_per_aligned_block(tmp_path):
    bc = _load_bc_module()
    _save_actions(tmp_path, [3] * 8)

    observations, actions, result = bc.collect_demonstration(
        tmp_path, level="1-1", skip=4, require_clear=False
    )

    assert observations.shape == (2, 84, 84, 4)
    assert observations.dtype == np.uint8
    assert actions.tolist() == [3, 3]
    assert result["decisions"] == 2


def test_collect_demonstration_rejects_nonconstant_action_block(tmp_path):
    bc = _load_bc_module()
    _save_actions(tmp_path, [3, 3, 4, 3])

    with pytest.raises(ValueError, match="not constant"):
        bc.collect_demonstration(
            tmp_path, level="1-1", skip=4, require_clear=False
        )
