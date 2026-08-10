"""Named Mario controller action sets."""
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, SIMPLE_MOVEMENT


_ACTION_SETS = {
    "simple": SIMPLE_MOVEMENT,
    "complex": COMPLEX_MOVEMENT,
}


def resolve_action_set(name: str) -> list[list[str]]:
    try:
        return _ACTION_SETS[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown action set {name!r}; expected one of {sorted(_ACTION_SETS)}"
        ) from exc


def action_set_size(name: str) -> int:
    return len(resolve_action_set(name))
