#!/usr/bin/env python
"""Behavior-clone a policy-cadence solver route into an existing PPO model."""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from marioai.curriculum import load_route
from marioai.envs import make_mario_env


def _make_vec_env(level, skip=4, shape=84, frame_stack=4):
    return VecFrameStack(
        DummyVecEnv([
            lambda: make_mario_env(level=level, skip=skip, shape=shape)
        ]),
        n_stack=frame_stack,
        channels_order="last",
    )


def _decision_actions(actions, skip):
    if len(actions) % skip:
        raise ValueError(
            f"route has {len(actions)} frames, not divisible by skip={skip}"
        )
    decisions = []
    for frame in range(0, len(actions), skip):
        block = actions[frame:frame + skip]
        if len(set(block)) != 1:
            raise ValueError(
                f"route action block at frame {frame} is not constant: {block}"
            )
        decisions.append(block[0])
    return decisions


def collect_demonstration(route_dir, level, skip=4, shape=84, frame_stack=4,
                          require_clear=True):
    """Replay an aligned route through the exact policy observation pipeline."""
    route = load_route(route_dir)
    if route["level"] != level:
        raise ValueError(
            f"route is for level {route['level']!r}, requested {level!r}"
        )
    decisions = _decision_actions(route["actions"], skip)
    env = _make_vec_env(level, skip=skip, shape=shape,
                        frame_stack=frame_stack)
    obs = env.reset()
    observations = []
    labels = []
    cleared = False
    max_x = 0
    info = {}
    try:
        for action in decisions:
            observations.append(np.asarray(obs[0]).copy())
            labels.append(action)
            obs, _, dones, infos = env.step(np.array([action]))
            info = infos[0]
            cleared = cleared or bool(info.get("flag_get", False))
            max_x = max(max_x, int(info.get("x_pos", 0)))
            if bool(dones[0]):
                break
    finally:
        env.close()
    result = {
        "cleared": cleared,
        "max_x": max_x,
        "decisions": len(labels),
    }
    if require_clear and not cleared:
        raise RuntimeError(
            f"aligned route did not clear: decisions={len(labels)}/"
            f"{len(decisions)} max_x={max_x}"
        )
    if len(labels) != len(decisions) and not cleared:
        raise RuntimeError(
            f"route terminated after {len(labels)}/{len(decisions)} decisions"
        )
    return (
        np.asarray(observations, dtype=np.uint8),
        np.asarray(labels, dtype=np.int64),
        result,
    )


def _policy_logits(policy, observations):
    features = policy.extract_features(observations)
    latent = policy.mlp_extractor.forward_actor(features)
    return policy.action_net(latent)


def action_accuracy(policy, observations, actions):
    policy.set_training_mode(False)
    with torch.no_grad():
        logits = _policy_logits(policy, observations)
        correct = (logits.argmax(dim=1) == actions).sum().item()
    return correct / len(actions)


def rollout(model, level, skip=4, shape=84, frame_stack=4,
            deterministic=True):
    """Run one true-start episode and return its terminal outcome."""
    env = _make_vec_env(level, skip=skip, shape=shape,
                        frame_stack=frame_stack)
    obs = env.reset()
    cleared = False
    max_x = 0
    decisions = 0
    try:
        while True:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, _, dones, infos = env.step(action)
            decisions += 1
            info = infos[0]
            cleared = cleared or bool(info.get("flag_get", False))
            max_x = max(max_x, int(info.get("x_pos", 0)))
            if bool(dones[0]):
                break
    finally:
        env.close()
    return {"cleared": cleared, "max_x": max_x, "decisions": decisions}


def train(args):
    observations, labels, route_result = collect_demonstration(
        args.route_dir,
        level=args.level,
        skip=args.skip,
        shape=args.shape,
        frame_stack=args.frame_stack,
    )
    print(
        f"dataset: {len(labels)} decisions, route clear="
        f"{route_result['cleared']} x={route_result['max_x']}",
        flush=True,
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = PPO.load(args.init_from, device=args.device)
    policy = model.policy
    obs_tensor, _ = policy.obs_to_tensor(observations)
    action_tensor = torch.as_tensor(
        labels, dtype=torch.long, device=policy.device
    )

    counts = np.bincount(labels, minlength=model.action_space.n)
    class_weights = np.ones(model.action_space.n, dtype=np.float32)
    present = counts > 0
    class_weights[present] = counts[present].max() / counts[present]
    weight_tensor = torch.as_tensor(class_weights, device=policy.device)

    parameters = (
        list(policy.features_extractor.parameters())
        + list(policy.mlp_extractor.policy_net.parameters())
        + list(policy.action_net.parameters())
    )
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    initial_accuracy = action_accuracy(policy, obs_tensor, action_tensor)
    initial_rollout = rollout(model, args.level, args.skip, args.shape,
                              args.frame_stack)
    print(
        f"initial: accuracy={initial_accuracy:.4f} "
        f"clear={initial_rollout['cleared']} "
        f"x={initial_rollout['max_x']}",
        flush=True,
    )
    if initial_rollout["cleared"]:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        model.save(args.out)
        return True

    sample_count = len(action_tensor)
    best_accuracy = initial_accuracy
    for epoch in range(1, args.epochs + 1):
        policy.set_training_mode(True)
        permutation = torch.randperm(sample_count, device=policy.device)
        total_loss = 0.0
        for start in range(0, sample_count, args.batch_size):
            index = permutation[start:start + args.batch_size]
            logits = _policy_logits(policy, obs_tensor[index])
            loss = F.cross_entropy(
                logits, action_tensor[index], weight=weight_tensor
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(index)

        should_measure = (
            epoch == 1
            or epoch % args.log_every == 0
            or epoch == args.epochs
        )
        if not should_measure:
            continue
        accuracy = action_accuracy(policy, obs_tensor, action_tensor)
        best_accuracy = max(best_accuracy, accuracy)
        print(
            f"epoch {epoch}: loss={total_loss / sample_count:.6f} "
            f"accuracy={accuracy:.4f}",
            flush=True,
        )
        if accuracy < 1.0 and epoch % args.rollout_every:
            continue
        result = rollout(model, args.level, args.skip, args.shape,
                         args.frame_stack)
        print(
            f"  greedy: clear={result['cleared']} x={result['max_x']} "
            f"decisions={result['decisions']}",
            flush=True,
        )
        if result["cleared"]:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            model.save(args.out)
            print(f"SAVED {args.out}", flush=True)
            return True

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    print(
        f"NO CLEAR after {args.epochs} epochs; "
        f"best offline accuracy={best_accuracy:.4f}; saved {args.out}",
        flush=True,
    )
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--route-dir", required=True)
    parser.add_argument("--level", default="1-3")
    parser.add_argument("--init-from", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--skip", type=int, default=4)
    parser.add_argument("--shape", type=int, default=84)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--rollout-every", type=int, default=50)
    args = parser.parse_args()
    raise SystemExit(0 if train(args) else 1)


if __name__ == "__main__":
    main()
