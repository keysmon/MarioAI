"""Pure-logic tests for the reverse-curriculum schedule and route IO."""
import random

import pytest

from marioai.curriculum import (
    CurriculumSchedule,
    load_route,
    save_route,
)


def test_frontier_starts_at_last_waypoint():
    s = CurriculumSchedule(8)
    assert s.frontier == 7


def test_window_must_be_positive():
    with pytest.raises(ValueError):
        CurriculumSchedule(8, window=0)


def test_history_must_be_positive():
    with pytest.raises(ValueError):
        CurriculumSchedule(8, history=0)


def test_sample_start_stays_in_window():
    s = CurriculumSchedule(8, window=3, rng=random.Random(1))
    s.frontier = 4
    samples = {s.sample_start() for _ in range(100)}
    assert samples == {4, 5, 6}


def test_sample_window_clamps_at_end():
    s = CurriculumSchedule(8, window=3, rng=random.Random(1))
    samples = {s.sample_start() for _ in range(50)}
    assert samples == {7}


def test_no_advance_below_threshold():
    s = CurriculumSchedule(8, history=10, advance_threshold=0.5)
    for _ in range(10):
        s.record(False)
    assert s.frontier == 7


def test_advance_on_threshold_and_history_reset():
    s = CurriculumSchedule(8, history=4, advance_threshold=0.5)
    for cleared in (True, True, False, False):
        s.record(cleared)
    assert s.frontier == 6
    # history was cleared on advance: 3 more results must not advance again
    for _ in range(3):
        s.record(True)
    assert s.frontier == 6
    s.record(True)
    assert s.frontier == 5


def test_frontier_never_goes_below_zero():
    s = CurriculumSchedule(2, history=1, advance_threshold=0.5)
    for _ in range(5):
        s.record(True)
    assert s.frontier == 0


def test_route_roundtrip_through_disk(tmp_path):
    route = {
        "level": "1-1",
        "actions": [3, 3, 4],
        "waypoints": [
            {"index": 0, "frame": 0, "x_pos": 40},
            {"index": 1, "frame": 3, "x_pos": 60},
        ],
    }
    save_route(route, tmp_path)
    assert load_route(tmp_path) == route


def test_load_route_rejects_missing_keys(tmp_path):
    (tmp_path / "route.json").write_text('{"actions": []}')
    with pytest.raises(ValueError):
        load_route(tmp_path)
