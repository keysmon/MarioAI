"""Pure-logic tests for the reverse-curriculum schedule and waypoint IO."""
import random

import pytest

from marioai.curriculum import (
    CurriculumSchedule,
    Waypoint,
    load_waypoints,
    save_waypoints,
)


def test_frontier_starts_at_last_waypoint():
    s = CurriculumSchedule(8)
    assert s.frontier == 7


def test_window_must_be_positive():
    with pytest.raises(ValueError):
        CurriculumSchedule(8, window=0)


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


def test_waypoint_roundtrip_through_disk(tmp_path):
    wps = [Waypoint(i, i * 100, 40 + 150 * i, {"blob": i}) for i in range(3)]
    save_waypoints(wps, tmp_path)
    loaded = load_waypoints(tmp_path)
    assert [w.index for w in loaded] == [0, 1, 2]
    assert [w.x_pos for w in loaded] == [40, 190, 340]
    assert loaded[2].state == {"blob": 2}
