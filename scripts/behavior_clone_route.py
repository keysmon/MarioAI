#!/usr/bin/env python
"""Behavior-clone a policy-cadence solver route into an existing PPO model."""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from marioai.actions import resolve_action_set
from marioai.demonstrations import (
    _decision_actions,
    collect_demonstration,
    load_demonstrations,
)
from marioai.envs import make_mario_env


def _make_vec_env(
    level,
    skip=4,
    shape=84,
    frame_stack=4,
    action_set="complex",
):
    return VecFrameStack(
        DummyVecEnv([
            lambda: make_mario_env(
                level=level,
                skip=skip,
                shape=shape,
                action_set=action_set,
            )
        ]),
        n_stack=frame_stack,
        channels_order="last",
    )


def _policy_logits(policy, observations):
    features = policy.extract_features(observations)
    latent = policy.mlp_extractor.forward_actor(features)
    return policy.action_net(latent)


def action_accuracy(policy, observations, actions, sample_weights=None):
    policy.set_training_mode(False)
    with torch.no_grad():
        logits = _policy_logits(policy, observations)
        correct = (logits.argmax(dim=1) == actions).float()
        if sample_weights is None:
            return float(correct.mean().item())
        return float(
            (correct * sample_weights).sum().item()
            / sample_weights.sum().item()
        )


def rollout(model, level, skip=4, shape=84, frame_stack=4,
            deterministic=True, action_set="complex"):
    """Run one true-start episode and return its terminal outcome."""
    env = _make_vec_env(level, skip=skip, shape=shape,
                        frame_stack=frame_stack, action_set=action_set)
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


def _rollouts(model, levels, args):
    return {
        level: rollout(
            model,
            level,
            skip=args.skip,
            deterministic=True,
            action_set=args.action_set,
        )
        for level in levels
    }


def _print_rollouts(prefix, results):
    summary = " ".join(
        f"{level}:clear={result['cleared']},x={result['max_x']}"
        for level, result in results.items()
    )
    print(f"{prefix}: {summary}", flush=True)


def train(args):
    batch = load_demonstrations(
        args.route_dirs,
        action_set=args.action_set,
        skip=args.skip,
    )
    levels = tuple(dict.fromkeys(batch.levels.tolist()))
    print(
        f"dataset: {len(batch.actions)} decisions across "
        f"{len(levels)} levels; routes="
        + ", ".join(
            f"{result['level']}:{result['decisions']}"
            for result in batch.route_results
        ),
        flush=True,
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = PPO.load(args.init_from, device=args.device)
    expected_actions = len(resolve_action_set(args.action_set))
    if model.action_space.n != expected_actions:
        raise ValueError(
            f"checkpoint action count {model.action_space.n} does not match "
            f"{args.action_set!r} action set ({expected_actions})"
        )
    policy = model.policy
    obs_tensor, _ = policy.obs_to_tensor(batch.observations)
    action_tensor = torch.as_tensor(
        batch.actions, dtype=torch.long, device=policy.device
    )
    sample_weight_tensor = torch.as_tensor(
        batch.sample_weights, dtype=torch.float32, device=policy.device
    )

    counts = np.bincount(
        batch.actions,
        weights=batch.sample_weights,
        minlength=model.action_space.n,
    )
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
    initial_accuracy = action_accuracy(
        policy, obs_tensor, action_tensor, sample_weight_tensor
    )
    initial_rollouts = _rollouts(model, levels, args)
    print(
        f"initial: level-balanced accuracy={initial_accuracy:.4f}",
        flush=True,
    )
    _print_rollouts("initial greedy", initial_rollouts)
    if all(result["cleared"] for result in initial_rollouts.values()):
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
            per_sample_loss = F.cross_entropy(
                logits,
                action_tensor[index],
                weight=weight_tensor,
                reduction="none",
            )
            minibatch_weights = sample_weight_tensor[index]
            loss = (
                per_sample_loss * minibatch_weights
            ).sum() / minibatch_weights.sum()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
            optimizer.step()
            total_loss += float(
                (per_sample_loss.detach() * minibatch_weights).sum()
            )

        should_measure = (
            epoch == 1
            or epoch % args.log_every == 0
            or epoch == args.epochs
        )
        if not should_measure:
            continue
        accuracy = action_accuracy(
            policy, obs_tensor, action_tensor, sample_weight_tensor
        )
        best_accuracy = max(best_accuracy, accuracy)
        print(
            f"epoch {epoch}: level-balanced loss="
            f"{total_loss / sample_weight_tensor.sum().item():.6f} "
            f"accuracy={accuracy:.4f}",
            flush=True,
        )
        if accuracy < 1.0 and epoch % args.rollout_every:
            continue
        results = _rollouts(model, levels, args)
        _print_rollouts("  greedy", results)
        if all(result["cleared"] for result in results.values()):
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--route-dir",
        dest="route_dirs",
        action="append",
        type=Path,
        required=True,
        help="solved route directory; repeat for shared multi-level cloning",
    )
    parser.add_argument("--action-set", default="complex")
    parser.add_argument("--init-from", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--skip", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--rollout-every", type=int, default=50)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    raise SystemExit(0 if train(args) else 1)


if __name__ == "__main__":
    main()
