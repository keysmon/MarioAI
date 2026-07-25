"""Unit tests for solver-step route logging."""
import importlib.util
from pathlib import Path

import pytest


def _load_solver():
    path = Path(__file__).parents[1] / "scripts" / "solve_level.py"
    spec = importlib.util.spec_from_file_location("solve_level", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _TerminalEnv:
    def __init__(self, terminal_call):
        self.calls = 0
        self.terminal_call = terminal_call

    def step(self, action):
        self.calls += 1
        terminated = self.calls == self.terminal_call
        return None, 0.0, terminated, False, {"flag_get": terminated}


def test_advance_logs_decision_that_reaches_terminal_subframe():
    solver = _load_solver()
    solver.SKIP = 4
    env = _TerminalEnv(terminal_call=3)
    actions = []

    done, info = solver.advance(env, action=4, n=1, log=actions)

    assert done
    assert info["flag_get"]
    assert actions == [4]


@pytest.mark.parametrize(
    ("level", "mode"),
    [
        ("2-2", "water"),
        ("7-2", "water"),
        ("4-4", "maze"),
        ("7-4", "maze"),
        ("8-4", "ground"),
    ],
)
def test_level_mode(level, mode):
    solver = _load_solver()

    assert solver.level_mode(level) == mode


def test_swim_candidates_are_aligned_to_four_frame_decisions():
    solver = _load_solver()

    candidates = solver.swim_candidates()

    assert candidates
    assert all(macro.frames % 4 == 0 for macro in candidates)
    assert all(
        pulse.frames % 4 == 0
        and pulse.action in (solver.RIGHT_A_B, solver.A)
        for macro in candidates
        for pulse in macro.pulses
    )


class _SwimEnv:
    def __init__(self):
        self.unwrapped = self
        self.calls = 0

    @property
    def ram(self):
        raise AssertionError("water candidates must not use grounded()")

    def load_state(self, snapshot):
        self.calls = 0

    def step(self, action):
        self.calls += 1
        return (
            None,
            0.0,
            False,
            False,
            {"flag_get": False, "x_pos": 100 + 5 * self.calls},
        )


class _RestoreEnv:
    def __init__(self):
        self.unwrapped = self

    def load_state(self, snapshot):
        pass


def test_swim_candidate_judges_progress_without_ground_state():
    solver = _load_solver()
    solver.SKIP = 4
    macro = solver.Macro(
        (
            solver.Pulse(solver.RIGHT_A_B, 4),
            solver.Pulse(solver.A, 4),
        )
    )

    ok, flag_got, info, trace = solver.try_swim_candidate(
        _SwimEnv(), object(), frontier=100, macro=macro
    )

    assert ok is True
    assert flag_got is False
    assert info["x_pos"] == 140
    assert trace == [solver.RIGHT_A_B, solver.A]


def test_maze_backward_snap_marks_wrong_branch():
    solver = _load_solver()

    progress = solver.maze_progress(
        previous_x=1400, current_x=900, branch=2
    )

    assert progress.wrong_branch is True
    assert progress.next_branch == 3


def test_maze_small_backward_motion_keeps_current_branch():
    solver = _load_solver()

    progress = solver.maze_progress(
        previous_x=1400, current_x=1145, branch=2
    )

    assert progress.wrong_branch is False
    assert progress.next_branch == 2


def test_maze_search_excludes_rejected_branch_from_expansion(monkeypatch):
    solver = _load_solver()
    monkeypatch.setattr(solver, "BACKTRACK_DEPTH", 1)
    monkeypatch.setattr(solver, "WAITS", (0,))
    monkeypatch.setattr(solver, "RIDES", (8,))
    monkeypatch.setattr(solver, "JUMP_ACTIONS", (solver.RIGHT_A_B,))
    monkeypatch.setattr(solver, "OFFSETS", (0,))
    monkeypatch.setattr(solver, "HOLDS", (8,))
    monkeypatch.setattr(solver, "ARC_ACTIONS", (solver.RIGHT_B,))
    monkeypatch.setattr(
        solver,
        "try_candidate",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("rejected branch was expanded")
        ),
    )

    result = solver._solve_ground_obstacle(
        object(),
        [(0, 40, 79, object())],
        excluded_branches={0},
        include_branch=True,
    )

    assert result is None


def test_maze_branch_identity_survives_history_truncation(monkeypatch):
    solver = _load_solver()
    solver.SKIP = 4
    monkeypatch.setattr(solver, "BACKTRACK_DEPTH", 3)
    monkeypatch.setattr(solver, "WAITS", (0,))
    monkeypatch.setattr(solver, "RIDES", (8,))
    monkeypatch.setattr(
        solver, "JUMP_ACTIONS", (solver.RIGHT_A_B, solver.RIGHT_A)
    )
    monkeypatch.setattr(solver, "OFFSETS", (0,))
    monkeypatch.setattr(solver, "HOLDS", (8,))
    monkeypatch.setattr(solver, "ARC_ACTIONS", (solver.RIGHT_B,))
    history = [
        (0, 100, 79, object()),
        (1, 200, 79, object()),
        (2, 300, 79, object()),
    ]
    retrying = False

    def fake_candidate(
        env,
        snap,
        x0,
        y0,
        frontier,
        known,
        wait,
        offset,
        jump,
        hold,
        arc,
        ride,
    ):
        if retrying:
            ok = jump == solver.RIGHT_A
            x = 500
        else:
            ok = jump == solver.RIGHT_A_B
            x = 1000 if x0 == 100 else 400
        return ok, False, {"x_pos": x, "y_pos": y0}, []

    monkeypatch.setattr(solver, "try_candidate", fake_candidate)
    first = solver._solve_ground_obstacle(
        _RestoreEnv(), history, include_branch=True
    )
    assert first is not None
    frame, _, _, branch = first
    assert frame == 0

    retrying = True
    retry = solver._solve_ground_obstacle(
        _RestoreEnv(),
        history[:2],
        excluded_branches={branch},
        minimum_branch=branch + 1,
        include_branch=True,
    )

    assert retry is not None
    assert retry[3] == branch + 1


def test_maze_rejections_are_scoped_to_their_junction():
    solver = _load_solver()
    branches = solver.MazeBranchMemory()
    first_junction = (20, 600, 80, object())
    # Route rewrites can revisit the same frame/x/y with a different emulator
    # snapshot; that is still a distinct junction search.
    later_junction = (20, 600, 80, object())

    branches.reject(first_junction, 2)

    assert branches.excluded(first_junction) == frozenset({2})
    assert branches.excluded(later_junction) == frozenset()


@pytest.mark.parametrize("mutation", ["same_locale_replacement", "cap_eviction"])
def test_maze_retry_commit_preserves_restore_outside_mutable_history(mutation):
    solver = _load_solver()
    snapshot = object()
    restore = (10, 600, 80, snapshot)
    mutable_history = [restore]

    if mutation == "same_locale_replacement":
        mutable_history = solver.push_history(
            mutable_history, (11, 605, 82, object())
        )
    else:
        for index in range(solver.BACKTRACK_DEPTH + 5):
            mutable_history = solver.push_history(
                mutable_history,
                (20 + index, 700 + 32 * index, 80, object()),
            )

    assert all(entry is not restore for entry in mutable_history)

    selected, committed_history = solver.commit_maze_candidate(
        search_history=[restore],
        mutable_history=mutable_history,
        frame0=10,
    )

    assert selected is restore
    assert any(entry is restore for entry in committed_history)
    assert selected[3] is snapshot


def test_save_native_route_persists_policy_compatibility_metadata(
    tmp_path, monkeypatch
):
    solver = _load_solver()
    solver.SKIP = 4
    saved = {}
    monkeypatch.setattr(
        solver,
        "save_route",
        lambda route, out_dir: saved.update(route),
    )

    solver.save_native_route(
        tmp_path,
        "1-1",
        [3, 4],
        [{"frame": 0, "x_pos": 40}],
        action_set="complex",
        decision_skip=4,
    )

    assert saved["action_set"] == "complex"
    assert saved["decision_skip"] == 4
    assert saved["actions"] == [3] * 4 + [4] * 4
