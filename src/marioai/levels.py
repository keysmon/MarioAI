from collections.abc import Sequence


ALL_LEVELS = tuple(
    f"{world}-{stage}"
    for world in range(1, 9)
    for stage in range(1, 5)
)
WORLD_GROUPS = {
    "worlds_1_4": ALL_LEVELS[:16],
    "worlds_5_8": ALL_LEVELS[16:],
    "all": ALL_LEVELS,
}


def validate_levels(levels: Sequence[str]) -> tuple[str, ...]:
    result = tuple(levels)
    unknown = sorted(set(result) - set(ALL_LEVELS))
    if unknown:
        raise ValueError(f"unknown Mario stage(s): {', '.join(unknown)}")
    if len(set(result)) != len(result):
        raise ValueError("duplicate Mario stage in level list")
    if not result:
        raise ValueError("at least one Mario stage is required")
    return result
