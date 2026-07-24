"""Unit tests for solver-step route logging."""
import importlib.util
from pathlib import Path


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
