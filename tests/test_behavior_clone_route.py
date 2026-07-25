"""Tests for policy-cadence behavior-cloning demonstrations."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

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


def test_parser_accepts_repeated_routes_for_one_shared_checkpoint():
    bc = _load_bc_module()

    args = bc.parse_args(
        [
            "--route-dir",
            "routes/2-2",
            "--route-dir",
            "routes/7-2",
            "--init-from",
            "models/shared.zip",
            "--out",
            "models/recovered.zip",
        ]
    )

    assert args.route_dirs == [Path("routes/2-2"), Path("routes/7-2")]
    assert args.action_set == "complex"


def test_minibatch_losses_preserve_equal_level_influence_across_epoch():
    bc = _load_bc_module()
    logits = torch.zeros((4, 2), dtype=torch.float32)
    actions = torch.zeros(4, dtype=torch.long)
    weights = torch.tensor([1.0, 1 / 3, 1 / 3, 1 / 3])
    epoch_weight = weights.sum()

    short_level = bc.level_balanced_minibatch_loss(
        logits[:1], actions[:1], weights[:1], epoch_weight
    )
    long_level = bc.level_balanced_minibatch_loss(
        logits[1:], actions[1:], weights[1:], epoch_weight
    )

    assert short_level.item() == pytest.approx(long_level.item())
    assert (short_level + long_level).item() == pytest.approx(
        torch.log(torch.tensor(2.0)).item()
    )
