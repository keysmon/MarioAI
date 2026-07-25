#!/usr/bin/env python
"""Search a clearing trajectory for a level with emulator snapshots (no learning).

Greedy depth-first search with backtracking: hold run-right; on death or
stall, rewind to a recent snapshot and try a menu of jump macros until one
gets Mario past the obstacle; splice that jump in and keep running. The
surviving route is saved as an ACTION SEQUENCE plus waypoint markers
(nes-py snapshots are same-process-only, so consumers rebuild snapshots by
deterministic replay - see SnapshotStartWrapper).

Usage: .venv/bin/python scripts/solve_level.py --level 1-3
"""
import argparse
import sys
from dataclasses import dataclass
from typing import Literal

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace

from marioai.actions import resolve_action_set
from marioai.curriculum import save_route

NOOP, RIGHT, RIGHT_A, RIGHT_B, RIGHT_A_B, A, LEFT = range(7)

# The solver runs in SOLVER-STEPS: each step holds one action for SKIP native
# frames - the policy's action cadence. A route searched at skip=4 only ever
# switches actions on 4-frame boundaries, so a frame-skip-4 agent can execute
# it (an arbitrarily-aligned skip=1 route dies under skip-4 quantization).
# SKIP is set from --skip in main. All frame-denominated constants below stay
# in NATIVE frames and are divided by SKIP at use; the saved route is expanded
# back to native frames so consumers (wrapper replay, curriculum) are unchanged
# and it replays identically at both cadences.
SKIP = 1

SNAP_EVERY = 20         # frames between backtrack snapshots
WAYPOINT_EVERY_X = 150  # min x-distance (px) between emitted waypoints
BACKTRACK_DEPTH = 16    # how many snapshots back the solver may rewind
STALL_FRAMES = 90       # no-x-progress frames before treating as an obstacle
ARC_CAP = 90            # max airborne frames before a candidate is judged
LAND_STABLE = 4         # consecutive grounded frames that count as landed
HOP_MIN = 8             # px past the grounded frontier that counts as a hop
HOP_UP = 24             # px height gain that counts as a hop (stepping stones)
K_PASSES = 12           # stop collecting after this many passing hops
AFTER_FIRST_BUDGET = 800  # extra candidates tried after the first pass
# NOTE: an explicit post-ride grounded "verify" phase was tried (20, 8, 2,
# and 1 NOOP frames) and removed: every fixed length rejects some
# legitimate keep-moving pit-edge landing (runs 10-13). Best-of-N scoring
# plus un-commit retries own dead-end avoidance; the ride-phase grounded
# check remains the only landing gate.

banned = set()          # taboo (x//16, y//16) landing locales: hops that
                        # were committed and later un-committed after the
                        # search from them exhausted. Without this memory,
                        # deterministic best-of-N re-picks the same hop
                        # after every un-commit and the search cycles
                        # (run 14: 474 -> 634 -> fail 745 -> re-pick 474)
EVENT_CAP = 300         # max obstacle events before giving up (safety)
MAX_TRIES = 60000       # per-solve candidate cap (retries multiply this)
OFFSETS = range(0, 60, 4)             # run-up frames before takeoff
HOLDS = (8, 12, 16, 20, 24, 28, 32)   # frames the jump is held
JUMP_ACTIONS = (RIGHT_A_B, RIGHT_A)   # running jump, walking jump
ARC_ACTIONS = (RIGHT_B, NOOP)         # ride the arc running / cut it short
WAITS = tuple(range(0, 289, 16))      # NOOP frames before run-up: covers a
                                      # full slow-lift cycle (~300 frames)
RIDES = (8, 60, 120, 180)             # NOOP frames after landing (lifts)


@dataclass(frozen=True)
class Pulse:
    """One native-frame-aligned action pulse inside a solver macro."""

    action: int
    frames: int


@dataclass(frozen=True)
class Macro:
    """A sequence of action pulses generated on four-frame boundaries."""

    pulses: tuple[Pulse, ...]

    @property
    def frames(self):
        return sum(pulse.frames for pulse in self.pulses)


@dataclass(frozen=True)
class MazeProgress:
    wrong_branch: bool
    next_branch: int


def level_mode(level: str) -> Literal["ground", "water", "maze"]:
    if level in {"2-2", "7-2"}:
        return "water"
    if level in {"4-4", "7-4"}:
        return "maze"
    return "ground"


def swim_candidates() -> tuple[Macro, ...]:
    """Return 4-aligned right/swim and neutral/swim pulse sequences."""
    macros = []
    for right_frames in (8, 12, 16, 24):
        for neutral_frames in (4, 8, 12):
            base = (
                Pulse(RIGHT_A_B, right_frames),
                Pulse(A, neutral_frames),
            )
            macros.append(Macro(base))
            macros.append(Macro(base * 2))
    return tuple(macros)


def maze_progress(
    previous_x: int, current_x: int, branch: int
) -> MazeProgress:
    """Classify a castle maze's non-terminal backward route reset."""
    wrong_branch = previous_x - current_x >= 256
    return MazeProgress(
        wrong_branch=wrong_branch,
        next_branch=branch + 1 if wrong_branch else branch,
    )


def advance(env, action, n, log=None):
    """Run `action` for up to n SOLVER-STEPS; stop early when the episode ends.

    Each solver-step executes `action` for SKIP native frames (SkipFrame's
    reward-less core). Appends one action per STARTED solver-step to `log`,
    including a terminal partial step: omitting the action that reaches the
    flag leaves a saved route one decision short. Death traces are rewound or
    rejected by callers. The route is stored in solver-steps and expanded to
    native frames at save time (a no-op at skip=1).
    """
    info = {}
    for _ in range(n):
        if log is not None:
            log.append(action)
        for _ in range(SKIP):
            _, _, term, trunc, info = env.step(action)
            if term or trunc:
                return True, info
    return False, info


def save_native_route(out_dir, level, actions, waypoints, **extra):
    """Expand solver-step actions to native frames (each repeated SKIP times)
    and scale waypoint frames by SKIP, matching the native-frame route schema.

    The result is 4-aligned by construction (`actions[i] == actions[i-i%SKIP]`)
    and a no-op at skip=1. Re-indexes waypoints. Returns (native_actions,
    native_waypoints) for reporting.
    """
    native = [a for a in actions for _ in range(SKIP)]
    wps = [{"index": i, "frame": w["frame"] * SKIP, "x_pos": w["x_pos"]}
           for i, w in enumerate(waypoints)]
    save_route({"level": level, "actions": native, "waypoints": wps, **extra},
               out_dir)
    return native, wps


def push_history(history, entry):
    """Append a grounded state, replacing the newest entry when it sits in
    the same 16px x/y locale. Micro-hops must not crowd out rewind reach:
    BACKTRACK_DEPTH counts entries, and a chain of tiny landings would
    otherwise shrink the effective rewind span to a few dozen pixels
    (this exact crowding sent run 8 backward vs run 7).
    """
    if history and abs(entry[1] - history[-1][1]) < 16 \
            and abs(entry[2] - history[-1][2]) < 16:
        history[-1] = entry
        return history
    history.append(entry)
    return history[-(BACKTRACK_DEPTH + 4):]


def grounded(env):
    """True when Mario stands on solid ground or rides a platform.

    SMB RAM $001D is the player float state: 0x00 = on ground (including
    standing on a moving lift), nonzero = airborne/sliding/climbing. This
    is the emulator's ground truth - unlike y-stability inference, it
    accepts landings on vertically moving platforms.
    """
    return env.unwrapped.ram[0x001D] == 0


def try_candidate(env, snap, x0, y0, frontier, known, wait, offset, jump,
                  hold, arc, ride):
    """Restore snap; wait, run, jump, ride the arc, land, stand/ride, judge.

    Phases: `wait` NOOP frames (timing vs lifts and enemies), `offset`
    run-right frames, `hold` jump frames, the arc under `arc` until
    grounded for LAND_STABLE consecutive frames, then `ride` NOOP frames
    (standing, or letting a lift carry Mario). Success = alive, GROUNDED
    at judge time (a landing that slides off a lip during `ride` must not
    count), and real progress: past the route's grounded frontier by
    HOP_MIN px, or a HOP_UP px height gain over the restore point -
    stepping stones and column tops are progress even when x barely moves,
    so wide crossings decompose into chained hops.
    Returns (ok, flag_got, info, trace); on ok the env is left in the
    judged state.
    """
    env.unwrapped.load_state(snap)
    trace = []
    # menu values are NATIVE frame counts; convert to solver-steps (// SKIP)
    land_stable = max(1, LAND_STABLE // SKIP)
    done, info = advance(env, NOOP, wait // SKIP, trace)
    if not done:
        done, info = advance(env, RIGHT_B, offset // SKIP, trace)
    if not done:
        done, info = advance(env, jump, hold // SKIP, trace)
    stable, frames = 0, 0
    while not done and frames < ARC_CAP // SKIP and stable < land_stable:
        done, info = advance(env, arc, 1, trace)
        frames += 1
        stable = stable + 1 if grounded(env) else 0
    if info.get("flag_get"):
        return True, True, info, trace
    if done or stable < land_stable:
        return False, False, info, trace  # died, or never came down grounded
    done, info = advance(env, NOOP, ride // SKIP, trace)
    if info.get("flag_get"):
        return True, True, info, trace
    if done or not grounded(env):
        return False, False, info, trace  # slid off a lip during the ride
    x, y = int(info.get("x_pos", 0)), int(info.get("y_pos", 0))
    if (x // 16, y // 16) in banned:
        return False, False, info, trace  # taboo: known dead-end landing
    # height hops must land on NOVEL terrain: without the `known` check a
    # y-gain hop onto already-committed ground counts as progress forever,
    # and the splice discards the route after it (run 18 looped 300 events
    # re-trading 634-progress for a "new" height hop onto old 474 ground)
    ok = (x > frontier + HOP_MIN
          or (y - y0 >= HOP_UP and (x // 16, y // 16) not in known))
    return ok, False, info, trace


def _solve_ground_obstacle(
    env,
    history,
    *,
    excluded_branches=frozenset(),
    minimum_branch=0,
    include_branch=False,
):
    """Search the macro menu and pick the BEST passing hop, not the first.

    Greedy first-ok selection committed descending dead-end hops (run 9:
    a stable ground-floor cul-de-sac beat the productive high approach).
    Passing candidates are collected - up to K_PASSES, or until
    AFTER_FIRST_BUDGET further candidates after the first pass - and the
    highest landing score (x + 2*y) wins: height keeps future options
    open in a platformer, descent rarely does; the preference applies
    only among passes, so a necessary descent still wins when it is the
    only option. The winner is RE-EXECUTED so the env is left in its
    landed state, not the last-tried candidate's.
    Returns (restore_frame, info, trace) or None.
    """
    frontier = max(h[1] for h in history)
    known = {(h[1] // 16, h[2] // 16) for h in history}
    extras = sorted(((w, r) for w in WAITS for r in RIDES),
                    key=lambda e: e[0] + e[1])
    depth = min(BACKTRACK_DEPTH, len(history))
    # EXTRAS-OUTER, BACK-INNER: global cheapest-first. With back as the
    # outer loop, one back level costs len(extras) * 420 core combos
    # (~32k), so MAX_TRIES starved depth to ~2 launch points and the
    # winning launch at back=3+ was never reached (run 16, x=925). This
    # order sweeps ALL launch points with the cheap core menu before any
    # expensive wait slice is touched.
    combos = ((back, wait, ride, jump, offset, hold, arc)
              for wait, ride in extras
              for back in range(1, depth + 1)
              for jump in JUMP_ACTIONS
              for offset in OFFSETS
              for hold in HOLDS
              for arc in ARC_ACTIONS)
    tried = 0
    first_pass_at = None
    passes = []  # (score, frame0, x0, snap, params, info, trace)
    for branch, (
        back,
        wait,
        ride,
        jump,
        offset,
        hold,
        arc,
    ) in enumerate(combos):
        if branch < minimum_branch or branch in excluded_branches:
            continue
        if tried >= MAX_TRIES or len(passes) >= K_PASSES:
            break
        if (first_pass_at is not None
                and tried - first_pass_at > AFTER_FIRST_BUDGET):
            break
        tried += 1
        if tried % 2000 == 0:
            print(f"  ...{tried} tried, {len(passes)} passing")
        frame0, x0, y0, snap = history[-back]
        ok, flag_got, info, trace = try_candidate(
            env, snap, x0, y0, frontier, known, wait, offset, jump, hold,
            arc, ride
        )
        if not ok:
            continue
        if flag_got:
            print(f"  FLAG reached in candidate: rewind to x={x0}")
            result = (frame0, info, trace)
            if include_branch:
                return (*result, branch)
            return result
        if first_pass_at is None:
            first_pass_at = tried
        # rewind penalty: with extras-outer enumeration, best-of-N compares
        # launch points GLOBALLY, and a near-tie must resolve toward the
        # shallowest splice - a marginally-higher deep rewind discards more
        # route and shifts every downstream enemy phase (run 17: a 3-point
        # win at x=321 broke the previously-instant x=417). Genuinely
        # better terrain (+50..150 score) still justifies deep rewinds.
        score = int(info["x_pos"]) + 2 * int(info["y_pos"]) - 10 * back
        passes.append(
            (
                score,
                frame0,
                x0,
                snap,
                (wait, offset, jump, hold, arc, ride),
                info,
                trace,
                branch,
            )
        )
    if not passes:
        print(f"  giving up after {tried} candidates")
        return None
    passes.sort(key=lambda p: p[0], reverse=True)
    score, frame0, x0, snap, params, info, trace, branch = passes[0]
    wait, offset, jump, hold, arc, ride = params
    # re-execute the winner so the env holds its landed state (trace is in
    # solver-actions; replay each at SKIP cadence)
    env.unwrapped.load_state(snap)
    for a in trace:
        advance(env, a, 1)
    print(f"  solved (best of {len(passes)}, score {score}): rewind to "
          f"x={x0}, wait={wait} offset={offset} jump={jump} hold={hold} "
          f"arc={arc} ride={ride} -> x={info['x_pos']} y={info['y_pos']}")
    result = (frame0, info, trace)
    if include_branch:
        return (*result, branch)
    return result


def try_swim_candidate(env, snap, frontier, macro):
    """Judge a swim sequence by survival/flag state and horizontal progress."""
    env.unwrapped.load_state(snap)
    trace = []
    done = False
    info = {}
    for pulse in macro.pulses:
        if pulse.frames % SKIP:
            raise ValueError(
                f"swim pulse {pulse.frames} is not aligned to skip={SKIP}"
            )
        done, info = advance(
            env, pulse.action, pulse.frames // SKIP, trace
        )
        if info.get("flag_get"):
            return True, True, info, trace
        if done:
            return False, False, info, trace
    x = int(info.get("x_pos", 0))
    return x > frontier + HOP_MIN, False, info, trace


def _solve_swim_obstacle(env, history):
    """Search pulse sequences without applying the ground-only landing judge."""
    frontier = max(h[1] for h in history)
    depth = min(BACKTRACK_DEPTH, len(history))
    passes = []
    for back in range(1, depth + 1):
        frame0, x0, _, snap = history[-back]
        for macro in swim_candidates():
            ok, flag_got, info, trace = try_swim_candidate(
                env, snap, frontier, macro
            )
            if not ok:
                continue
            if flag_got:
                print(f"  FLAG reached in swim candidate: rewind to x={x0}")
                return frame0, info, trace
            score = int(info.get("x_pos", 0)) - 10 * back
            passes.append((score, frame0, x0, snap, info, trace, macro))
    if not passes:
        print("  giving up after all aligned swim candidates")
        return None
    passes.sort(key=lambda item: item[0], reverse=True)
    score, frame0, x0, snap, info, trace, macro = passes[0]
    env.unwrapped.load_state(snap)
    for action in trace:
        advance(env, action, 1)
    print(
        f"  solved swim (score {score}): rewind to x={x0}, "
        f"{len(macro.pulses)} pulses/{macro.frames} native frames -> "
        f"x={info['x_pos']}"
    )
    return frame0, info, trace


def solve_obstacle(
    env,
    history,
    mode="ground",
    *,
    excluded_branches=frozenset(),
    minimum_branch=0,
):
    """Dispatch obstacle recovery without changing the solver-step cadence."""
    if mode == "water":
        return _solve_swim_obstacle(env, history)
    return _solve_ground_obstacle(
        env,
        history,
        excluded_branches=excluded_branches,
        minimum_branch=minimum_branch,
        include_branch=mode == "maze",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="1-3")
    ap.add_argument("--action-set", default="complex")
    ap.add_argument("--out", default=None,
                    help="route dir (default models/ft_<level>/waypoints)")
    ap.add_argument("--max-frames", type=int, default=60000,
                    help="route-length budget (NATIVE frames) before giving up")
    ap.add_argument("--skip", type=int, default=4,
                    help="native frames per solver step (policy cadence)")
    args = ap.parse_args()
    global SKIP
    SKIP = args.skip
    if SKIP < 1:
        ap.error("--skip must be >= 1")
    mode = level_mode(args.level)
    action_space = resolve_action_set(args.action_set)
    if mode == "water" and any(
        pulse.frames % SKIP
        for macro in swim_candidates()
        for pulse in macro.pulses
    ):
        ap.error("water pulse durations must align to --skip")
    out_dir = args.out or f"models/ft_{args.level}/waypoints"
    route_metadata = {
        "action_set": args.action_set,
        "decision_skip": SKIP,
    }

    env = JoypadSpace(
        gym_super_mario_bros.make(f"SuperMarioBros-{args.level}-v0",
                                  render_mode="rgb_array"),
        action_space,
    )
    _, info = env.reset(seed=0)
    raw = env.unwrapped

    start_x = int(info.get("x_pos", 40))
    start_y = int(info.get("y_pos", 79))
    actions = []            # the route: one action per executed frame
    history = [(0, start_x, start_y, raw.dump_state())]  # solver rewinds
    waypoints = [{"index": 0, "frame": 0, "x_pos": start_x}]
    last_x, last_progress_frame = start_x, 0
    was_grounded = True
    events = 0
    commit_counts = {}
    flag = False
    previous_x = start_x
    maze_branch = 0
    maze_excluded = set()
    maze_restore = None
    maze_minimum_branch = 0

    def handle_obstacle(obstacle_x):
        nonlocal history, waypoints, last_x, last_progress_frame, flag
        nonlocal was_grounded, events, commit_counts
        nonlocal maze_branch, maze_restore, maze_minimum_branch
        events += 1
        solved = None
        if events > EVENT_CAP:
            print(f"FAILED: {events} obstacle events without reaching the "
                  f"flag (EVENT_CAP). Likely a hop loop - inspect the "
                  f"partial route.")
        else:
            for retry in range(3):
                solved = solve_obstacle(
                    env,
                    history,
                    mode,
                    excluded_branches=maze_excluded,
                    minimum_branch=maze_minimum_branch,
                )
                if solved is not None or len(history) <= 2:
                    break
                # dead-end perch: BAN the newest landings (else the
                # deterministic best-of-N re-picks them and cycles), then
                # un-commit and retry from deeper history. The ban radius
                # escalates per retry: the first un-commit is surgical,
                # repeated failure at the same obstacle condemns the whole
                # launch region (16px point-bans invite shuffling along a
                # platform one locale at a time - run 15).
                if mode != "water":
                    k = retry + 1
                    for h in history[-2:]:
                        for dx in range(-k, k + 1):
                            for dy in range(-k, k + 1):
                                banned.add(
                                    (
                                        h[1] // 16 + dx,
                                        h[2] // 16 + dy,
                                    )
                                )
                    print(
                        f"  retry {retry + 1}: banned {len(banned)} "
                        f"locales (radius {k}), dropping 2 newest history "
                        f"entries ({len(history) - 2} left)"
                    )
                history = history[:-2]
        if solved is None:
            save_native_route(
                out_dir + "-partial",
                args.level,
                actions,
                waypoints,
                **route_metadata,
                partial=True,
                blocked_x=obstacle_x,
            )
            print(f"FAILED: no macro cleared x={obstacle_x}. Partial route "
                  f"saved to {out_dir}-partial for diagnosis.")
            sys.exit(1)
        if mode == "maze":
            frame0, info, trace, selected_branch = solved
            restore_entry = next(
                entry for entry in history if entry[0] == frame0
            )
            maze_restore = restore_entry
            maze_branch = selected_branch
            maze_minimum_branch = 0
        else:
            frame0, info, trace = solved
        del actions[frame0:]          # rewind the route to the restore point
        actions.extend(trace)         # splice the successful macro in
        history = [h for h in history if h[0] <= frame0]
        waypoints = [w for w in waypoints if w["frame"] <= frame0]
        last_x = int(info["x_pos"])
        last_progress_frame = len(actions)
        was_grounded = mode != "ground" or grounded(env)
        # belt-and-braces loop breaker: a locale committed twice without a
        # breakthrough is banned - no loop class survives two beats
        locale = (last_x // 16, int(info["y_pos"]) // 16)
        commit_counts[locale] = commit_counts.get(locale, 0) + 1
        if mode != "water" and commit_counts[locale] >= 2:
            banned.add(locale)
        if info.get("flag_get"):
            flag = True
        else:
            history = push_history(history, (len(actions), last_x,
                                             int(info["y_pos"]),
                                             raw.dump_state()))

    while len(actions) * SKIP < args.max_frames and not flag:
        cruise_action = RIGHT_A_B if mode == "water" else RIGHT_B
        done, info = advance(env, cruise_action, 1, actions)
        frame = len(actions)
        if info.get("flag_get"):
            flag = True
            break
        if done:
            print(f"death at x={info['x_pos']} (frame {frame}); solving...")
            handle_obstacle(int(info["x_pos"]))
            previous_x = last_x
            continue
        x = int(info["x_pos"])
        if mode == "maze":
            progress = maze_progress(previous_x, x, maze_branch)
            if progress.wrong_branch:
                if maze_restore is None:
                    print(
                        f"maze reset {previous_x}->{x} before a branch "
                        "candidate was recorded; recovering from history"
                    )
                    handle_obstacle(x)
                else:
                    reset_from_x = previous_x
                    maze_excluded.add(maze_branch)
                    maze_branch = progress.next_branch
                    maze_minimum_branch = maze_branch
                    frame0, restore_x, restore_y, snap = maze_restore
                    raw.load_state(snap)
                    del actions[frame0:]
                    history = [
                        entry for entry in history if entry[0] <= frame0
                    ]
                    waypoints = [
                        waypoint
                        for waypoint in waypoints
                        if waypoint["frame"] <= frame0
                    ]
                    last_x = restore_x
                    last_progress_frame = frame0
                    was_grounded = grounded(env)
                    previous_x = restore_x
                    print(
                        f"maze branch {maze_branch - 1} reset "
                        f"{reset_from_x}->{x}; restored x={restore_x}, "
                        f"trying branch {maze_branch}"
                    )
                    handle_obstacle(restore_x)
                previous_x = last_x
                continue
        if x > last_x:
            last_x, last_progress_frame = x, frame
        elif frame - last_progress_frame > STALL_FRAMES // SKIP:
            print(f"stall at x={last_x} (frame {frame}); solving...")
            handle_obstacle(last_x)
            previous_x = last_x
            continue
        is_grounded = mode == "water" or grounded(env)
        # snapshot every landing (airborne -> grounded transition) plus the
        # SNAP_EVERY cadence, but only ever GROUNDED states: landings are
        # the restore points that matter (platform tops, lift boardings),
        # and a lift-riding window shorter than SNAP_EVERY would otherwise
        # never be captured (this exact miss blocked run 4 at x=745)
        if (is_grounded and not was_grounded) or (
            frame % max(1, SNAP_EVERY // SKIP) == 0 and is_grounded
        ):
            history = push_history(history, (frame, x, int(info["y_pos"]),
                                             raw.dump_state()))
            if x - waypoints[-1]["x_pos"] >= WAYPOINT_EVERY_X:
                waypoints.append(
                    {"index": len(waypoints), "frame": frame, "x_pos": x}
                )
        was_grounded = is_grounded
        previous_x = x

    if not flag:
        save_native_route(
            out_dir + "-partial",
            args.level,
            actions,
            waypoints,
            **route_metadata,
            partial=True,
            blocked_x=last_x,
        )
        print(f"FAILED: frame budget exhausted before the flag (x={last_x}). "
              f"Partial route saved to {out_dir}-partial.")
        sys.exit(1)

    native, wps = save_native_route(
        out_dir,
        args.level,
        actions,
        waypoints,
        **route_metadata,
    )
    print(f"CLEARED {args.level}: {len(wps)} waypoints, "
          f"{len(native)}-frame route (skip={SKIP}) -> {out_dir}")
    for w in wps:
        print(f"  wp{w['index']:03d} frame={w['frame']:5d} x={w['x_pos']}")


if __name__ == "__main__":
    main()
