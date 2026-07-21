"""Reverse-curriculum schedule + waypoint persistence for snapshot starts.

Waypoints are emulator snapshots along a solved trajectory (see
scripts/solve_level.py). Index 0 is the level start; the last index is
nearest the flag. The schedule starts episodes near the flag and slides the
start earlier as the policy masters each segment (Salimans & Chen 2018).
"""
import json
import pickle
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Waypoint:
    index: int
    frame: int
    x_pos: int
    state: object  # opaque nes-py emulator snapshot (dump_state())


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
        rate = sum(self._results) / max(len(self._results), 1)
        if self.frontier > 0 and full and rate >= self.advance_threshold:
            self.frontier -= 1
            self._results.clear()


def save_waypoints(waypoints, out_dir):
    """Write snapshots as wp_NNN.pkl plus a manifest.json describing them."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for wp in waypoints:
        fname = f"wp_{wp.index:03d}.pkl"
        with open(out / fname, "wb") as f:
            pickle.dump(wp.state, f)
        manifest.append(dict(index=wp.index, frame=wp.frame,
                             x_pos=int(wp.x_pos), file=fname))
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def load_waypoints(in_dir):
    src = Path(in_dir)
    with open(src / "manifest.json") as f:
        manifest = json.load(f)
    waypoints = []
    for entry in sorted(manifest, key=lambda e: e["index"]):
        with open(src / entry["file"], "rb") as f:
            state = pickle.load(f)
        waypoints.append(Waypoint(entry["index"], entry["frame"],
                                  entry["x_pos"], state))
    return waypoints
