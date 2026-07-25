from collections import Counter

import pytest

from marioai.sampling import assign_worker_levels, regression_weights


def test_balanced_assignment_covers_32_stages_twice_with_64_workers():
    levels = tuple(f"{world}-{stage}" for world in range(1, 9) for stage in range(1, 5))

    assigned = assign_worker_levels(levels, 64)

    assert Counter(assigned) == {level: 2 for level in levels}


def test_regression_weight_gets_extra_fixed_workers():
    assigned = assign_worker_levels(
        ("1-1", "1-2", "1-3"), 8, {"1-1": 2.0, "1-2": 1.0, "1-3": 1.0}
    )

    assert Counter(assigned) == {"1-1": 4, "1-2": 2, "1-3": 2}


def test_weights_reject_unknown_or_nonpositive_values():
    with pytest.raises(ValueError):
        assign_worker_levels(("1-1",), 1, {"1-2": 1.0})
    with pytest.raises(ValueError):
        assign_worker_levels(("1-1",), 1, {"1-1": 0.0})


def test_regressed_stage_weight_doubles_for_next_worker_assignment():
    weights = regression_weights(
        ("1-1", "1-2"),
        previous={"1-1": True, "1-2": False},
        current={"1-1": False, "1-2": False},
    )

    assert weights == {"1-1": 2.0, "1-2": 1.0}
