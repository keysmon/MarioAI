import pytest
from marioai.actions import action_set_size, resolve_action_set
from marioai.envs import make_mario_env
from marioai.levels import ALL_LEVELS, WORLD_GROUPS, validate_levels


def test_manifest_contains_all_32_stages_in_world_order():
    assert ALL_LEVELS == tuple(
        f"{world}-{stage}"
        for world in range(1, 9)
        for stage in range(1, 5)
    )
    assert WORLD_GROUPS["worlds_1_4"] == ALL_LEVELS[:16]
    assert WORLD_GROUPS["worlds_5_8"] == ALL_LEVELS[16:]


def test_validate_levels_rejects_unknown_and_duplicates():
    with pytest.raises(ValueError, match="unknown Mario stage"):
        validate_levels(["9-1"])
    with pytest.raises(ValueError, match="duplicate Mario stage"):
        validate_levels(["1-1", "1-1"])


def test_complex_action_set_has_down_and_12_actions():
    actions = resolve_action_set("complex")
    assert len(actions) == 12
    assert ["down"] in actions
    assert action_set_size("complex") == 12


def test_unknown_action_set_fails_loudly():
    with pytest.raises(ValueError, match="unknown action set"):
        resolve_action_set("wide")


def test_complex_mario_env_has_12_discrete_actions():
    env = make_mario_env("1-1", action_set="complex")
    try:
        assert env.action_space.n == 12
    finally:
        env.close()
