"""Record N rollouts of a policy on one level; keep the cleanest as a GIF."""
import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from marioai.actions import action_set_size
from marioai.envs import make_mario_env
from marioai.train import validate_resume_model


def _make_recording_env(
    *,
    level,
    skip,
    shape,
    frame_stack,
):
    """Build the complex-action policy environment used for GIF rollouts."""
    return VecFrameStack(
        DummyVecEnv(
            [
                lambda: make_mario_env(
                    level=level,
                    skip=skip,
                    shape=shape,
                    capture_frames=True,
                    action_set="complex",
                )
            ]
        ),
        n_stack=frame_stack,
        channels_order="last",
    )


def _rollout(model, level, max_steps, skip, shape, frame_stack):
    """One rollout. Returns (frames, cleared, max_x, steps).

    Captures EVERY native game frame (all `skip` intra-step frames), not just
    1-of-`skip`, so the GIF is smooth rather than choppy.
    """
    venv = _make_recording_env(
        level=level,
        skip=skip,
        shape=shape,
        frame_stack=frame_stack,
    )
    try:
        obs = venv.reset()
        frames = [venv.render()]
        cleared, max_x = False, 0
        step = 0
        for step in range(max_steps):
            # Sampling makes "keep the cleanest of N" meaningful.
            action, _ = model.predict(obs, deterministic=False)
            obs, _, dones, infos = venv.step(action)
            frames.extend(venv.get_attr("last_frames")[0])
            cleared = cleared or bool(infos[0].get("flag_get", False))
            max_x = max(max_x, int(infos[0].get("x_pos", 0)))
            if bool(dones[0]):
                break
        return frames, cleared, max_x, step + 1
    finally:
        venv.close()


def _select_playback_frames(frames):
    """Select deterministic 2x playback frames and retain the terminal frame."""
    if not frames:
        raise ValueError("cannot encode a GIF without frames")
    selected_indices = list(range(0, len(frames), 2))
    terminal_index = len(frames) - 1
    if selected_indices[-1] != terminal_index:
        selected_indices.append(terminal_index)
    return [frames[index] for index in selected_indices]


def _inspect_gif(path: Path) -> dict:
    """Read back encoded metadata so invalid media never passes the gate."""
    with Image.open(path) as gif:
        loop = gif.info.get("loop")
        durations = []
        sizes = []
        for frame_index in range(gif.n_frames):
            gif.seek(frame_index)
            durations.append(gif.info.get("duration"))
            sizes.append(gif.size)
        return {
            "loop": loop,
            "frames": gif.n_frames,
            "durations": tuple(durations),
            "sizes": tuple(sizes),
        }


def _write_gif(path, frames, fps):
    """Encode and verify infinite-loop playback in uniform frame quanta."""
    if fps <= 0 or fps > 50:
        raise ValueError("fps must be between 1 and 50 for browser-safe GIFs")
    selected = _select_playback_frames(frames)
    expected_size = selected[0].shape[:2]
    if any(frame.shape[:2] != expected_size for frame in selected):
        raise ValueError("all GIF frames must have matching dimensions")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_duration_ms = max(20, round((1000 / fps) / 10) * 10)
    imageio.mimsave(
        path,
        selected,
        format="GIF",
        duration=frame_duration_ms,
        loop=0,
    )
    metadata = _inspect_gif(path)
    if metadata["loop"] != 0:
        raise RuntimeError("encoded GIF is missing infinite-loop metadata")
    if len(set(metadata["sizes"])) != 1:
        raise RuntimeError("encoded GIF contains inconsistent dimensions")
    durations = metadata["durations"]
    if any(
        duration is None
        or duration < 20
        or duration % frame_duration_ms
        for duration in durations
    ):
        raise RuntimeError(
            "encoded GIF durations must be browser-safe multiples of one "
            "frame quantum"
        )
    total_duration_ms = sum(durations)
    expected_duration_ms = len(selected) * frame_duration_ms
    if total_duration_ms != expected_duration_ms:
        raise RuntimeError(
            "encoded GIF duration does not preserve every selected frame "
            "quantum"
        )

    with Image.open(path) as gif:
        gif.seek(gif.n_frames - 1)
        encoded_terminal = np.asarray(gif.convert("RGB"), dtype=np.int16)
    expected_terminal = np.asarray(selected[-1], dtype=np.int16)
    if encoded_terminal.shape != expected_terminal.shape:
        raise RuntimeError("encoded GIF terminal frame has the wrong shape")
    terminal_mean_error = float(
        np.abs(encoded_terminal - expected_terminal).mean()
    )
    if terminal_mean_error > 8.0:
        raise RuntimeError(
            "encoded GIF does not preserve the terminal frame image"
        )
    metadata["frame_duration_ms"] = frame_duration_ms
    metadata["semantic_frames"] = len(selected)
    metadata["total_duration_ms"] = total_duration_ms
    metadata["terminal_mean_error"] = terminal_mean_error
    return metadata


def record(model, level, out, rollouts=5, fps=30, max_steps=3000,
           skip=4, shape=84, frame_stack=4):
    if rollouts <= 0:
        raise ValueError("rollouts must be positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    validate_resume_model(
        model,
        action_count=action_set_size("complex"),
        observation_shape=(shape, shape),
        frame_stack=frame_stack,
        extractor_name="impala",
        features_dim=512,
        channels=(16, 32, 32),
    )
    runs = [_rollout(model, level, max_steps, skip, shape, frame_stack)
            for _ in range(rollouts)]
    clears = [r for r in runs if r[1]]
    if clears:
        best = min(clears, key=lambda r: r[3])      # cleanest clear = fewest steps
        outcome = f"clear ({best[3]} steps)"
    else:
        best = max(runs, key=lambda r: r[2])        # best partial = furthest x
        outcome = f"partial (x_pos={best[2]})"
    metadata = _write_gif(out, best[0], fps)
    print(
        f"{level}: {outcome} -> {out} "
        f"({metadata['frames']} frames, loop=0)"
    )
    metadata["outcome"] = outcome
    return metadata


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--level", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--rollouts", type=int, default=5)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--skip", type=int, default=4)
    p.add_argument("--shape", type=int, default=84)
    p.add_argument("--frame-stack", type=int, default=4)
    args = p.parse_args()
    model = PPO.load(args.model, device="cpu")
    record(model, args.level, args.out, rollouts=args.rollouts,
           fps=args.fps, max_steps=args.max_steps, skip=args.skip,
           shape=args.shape, frame_stack=args.frame_stack)


if __name__ == "__main__":
    main()
