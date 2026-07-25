import pytest
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
