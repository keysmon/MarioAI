from types import SimpleNamespace

import gymnasium as gym
import numpy as np
from PIL import Image
import pytest

import marioai.record_gif as recording
from marioai.features import ImpalaCnnFeaturesExtractor


def _frame(value: int) -> np.ndarray:
    return np.full((6, 8, 3), value, dtype=np.uint8)


def test_playback_selects_every_second_frame_and_preserves_terminal_frame():
    frames = [_frame(value) for value in range(6)]

    selected = recording._select_playback_frames(frames)

    assert [int(frame[0, 0, 0]) for frame in selected] == [0, 2, 4, 5]


def test_gif_encoding_sets_infinite_loop_and_browser_safe_uniform_duration(
    tmp_path,
):
    output = tmp_path / "recording.gif"

    recording._write_gif(
        output,
        [_frame(value * 40) for value in range(6)],
        fps=20,
    )

    with Image.open(output) as gif:
        durations = []
        sizes = []
        for frame_index in range(gif.n_frames):
            gif.seek(frame_index)
            durations.append(gif.info["duration"])
            sizes.append(gif.size)
        assert gif.info["loop"] == 0
        assert gif.n_frames == 4
    assert set(sizes) == {(8, 6)}
    assert set(durations) == {50}


def test_gif_coalescing_preserves_frame_quanta_and_terminal_image(tmp_path):
    output = tmp_path / "repeated-frames.gif"
    frames = [
        _frame(0),
        _frame(10),
        _frame(0),
        _frame(20),
        _frame(80),
        _frame(80),
    ]

    metadata = recording._write_gif(output, frames, fps=20)

    assert metadata["frames"] == 2
    assert metadata["semantic_frames"] == 4
    assert metadata["durations"] == (100, 100)
    assert metadata["total_duration_ms"] == 200
    with Image.open(output) as gif:
        gif.seek(gif.n_frames - 1)
        terminal = np.asarray(gif.convert("RGB"))
    assert np.array_equal(terminal, _frame(80))


def test_record_rejects_non_complex_checkpoint_before_rollout(tmp_path):
    incompatible = SimpleNamespace(
        action_space=gym.spaces.Discrete(7),
        observation_space=gym.spaces.Box(
            0,
            255,
            shape=(4, 84, 84),
            dtype=np.uint8,
        ),
        policy=SimpleNamespace(features_extractor=object()),
    )

    with pytest.raises(ValueError, match="checkpoint action count 7"):
        recording.record(
            incompatible,
            "1-1",
            tmp_path / "recording.gif",
            rollouts=1,
            max_steps=1,
        )


def test_record_returns_verified_semantic_playback_metadata(
    tmp_path, monkeypatch
):
    observation_space = gym.spaces.Box(
        0,
        255,
        shape=(4, 84, 84),
        dtype=np.uint8,
    )
    model = SimpleNamespace(
        action_space=gym.spaces.Discrete(12),
        observation_space=observation_space,
        policy=SimpleNamespace(
            features_extractor=ImpalaCnnFeaturesExtractor(
                observation_space,
                features_dim=512,
                channels=(16, 32, 32),
            )
        ),
    )
    monkeypatch.setattr(
        recording,
        "_rollout",
        lambda *_args, **_kwargs: (
            [_frame(value * 40) for value in range(6)],
            False,
            100,
            1,
        ),
    )

    metadata = recording.record(
        model,
        "1-1",
        tmp_path / "recording.gif",
        rollouts=1,
        fps=20,
        max_steps=1,
    )

    assert metadata["semantic_frames"] == 4
    assert metadata["total_duration_ms"] == 200
    assert metadata["outcome"] == "partial (x_pos=100)"


def test_recording_environment_uses_complex_actions():
    environment = recording._make_recording_env(
        level="1-1",
        skip=4,
        shape=84,
        frame_stack=4,
    )
    try:
        assert environment.action_space.n == 12
    finally:
        environment.close()
