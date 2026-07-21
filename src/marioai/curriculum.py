"""Reverse-curriculum schedule + route persistence for snapshot starts.

A route is a solved trajectory (see scripts/solve_level.py) persisted as an
action sequence plus waypoint markers - not emulator snapshots directly,
since nes-py snapshots are same-process-only and can't cross disk or a
subprocess boundary. Waypoint index 0 is the level start; the last index is
nearest the flag. SnapshotStartWrapper replays the actions once per worker to
rebuild in-process snapshots at each waypoint frame. The schedule starts
episodes near the flag and slides the start earlier as the policy masters
each segment (Salimans & Chen 2018).
"""
import json
import random
from collections import deque
from pathlib import Path


class CurriculumSchedule:
    """Sliding-window reverse curriculum over waypoint indices [0, n).

    Episodes sample a start uniformly from the `window` waypoints beginning
    at the frontier (the frontier itself plus already-mastered later ones,
    so old segments keep getting rehearsed). Once the clear-rate over the
    last `history` episodes reaches `advance_threshold`, the frontier slides
    one waypoint earlier and the history resets.
    """

    def __init__(self, n_waypoints, window=3, advance_threshold=0.5,
                 history=10, rng=None):
        if n_waypoints < 1:
            raise ValueError("need at least one waypoint")
        if window < 1:
            raise ValueError("window must be >= 1")
        if history < 1:
            raise ValueError("history must be >= 1")
        self.n = n_waypoints
        self.window = window
        self.advance_threshold = advance_threshold
        self.frontier = n_waypoints - 1
        self._results = deque(maxlen=history)
        self._rng = rng or random.Random(0)

    def sample_start(self):
        hi = min(self.frontier + self.window, self.n)
        return self._rng.randrange(self.frontier, hi)

    def record(self, cleared):
        self._results.append(bool(cleared))
        full = len(self._results) == self._results.maxlen
        rate = sum(self._results) / len(self._results)
        if self.frontier > 0 and full and rate >= self.advance_threshold:
            self.frontier -= 1
            self._results.clear()


def save_route(route, out_dir):
    """Write a solved route (action sequence + waypoint markers) as JSON."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "route.json", "w") as f:
        json.dump(route, f, indent=2)


def load_route(in_dir):
    """Load route.json; raises ValueError if required keys are missing."""
    with open(Path(in_dir) / "route.json") as f:
        route = json.load(f)
    for key in ("level", "actions", "waypoints"):
        if key not in route:
            raise ValueError(f"route.json missing key: {key}")
    return route
