#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch
from tensordict import TensorDictBase
from torchrl.envs.utils import ExplorationType, set_exploration_type

from benchmarl.experiment import Experiment
from benchmarl.hydra_config import reload_experiment_from_file
from benchmarl.temm import TEMMConfig, analyze_rollouts
from benchmarl.utils import seed_everything


def reload_experiment_for_temm(checkpoint: str):
    """Reload a checkpoint, falling back to config.pkl if Hydra metadata is stale."""

    try:
        return reload_experiment_from_file(checkpoint)
    except Exception:
        config_file = Path(checkpoint).parent.parent / "config.pkl"
        if not config_file.exists():
            raise
        return Experiment.reload_from_file(checkpoint)


def collect_evaluation_rollouts(experiment, config: TEMMConfig) -> List[TensorDictBase]:
    """Collect evaluation trajectories without logging or mutating training state."""

    if config.seed is not None:
        seed_everything(config.seed)
        try:
            experiment.test_env.set_seed(config.seed)
        except NotImplementedError:
            pass
    elif config.static_evaluation or (
        config.static_evaluation is None and experiment.config.evaluation_static
    ):
        seed_everything(experiment.seed)
        try:
            experiment.test_env.set_seed(experiment.seed)
        except NotImplementedError:
            pass

    exploration_type = (
        ExplorationType.DETERMINISTIC
        if experiment.config.evaluation_deterministic_actions
        else ExplorationType.RANDOM
    )
    with torch.no_grad(), set_exploration_type(exploration_type):
        if experiment.test_env.batch_size == ():
            rollouts = [
                experiment.test_env.rollout(
                    max_steps=experiment.max_steps,
                    policy=experiment.policy,
                    auto_cast_to_device=True,
                    break_when_any_done=True,
                )
                for _ in range(config.episodes)
            ]
        else:
            batched_rollouts = experiment.test_env.rollout(
                max_steps=experiment.max_steps,
                policy=experiment.policy,
                auto_cast_to_device=True,
                break_when_any_done=False,
            )
            rollouts = list(batched_rollouts.unbind(0))[: config.episodes]
    return rollouts


def write_summary(result, path: Path) -> None:
    lines = [
        "TEMM analysis summary",
        "",
        f"Organizational fit: {result.fit.organizational:.4f}",
        f"Structural fit: {result.fit.structural:.4f}",
        f"Functional fit: {result.fit.functional:.4f}",
        f"Mean return: {result.mean_return:.4f}",
        f"Reward std: {result.reward_std:.4f}",
        f"Roles: {len(result.roles)}",
        f"Goals: {len(result.goals)}",
        f"Missions: {len(result.missions)}",
        f"Permissions: {len(result.permissions)}",
        f"Obligations: {len(result.obligations)}",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run post-hoc TEMM analysis on a BenchMARL checkpoint."
    )
    parser.add_argument("checkpoint", type=str, help="BenchMARL checkpoint file")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--out", type=str, default="temm_result.json")
    parser.add_argument("--success-quantile", type=float, default=0.75)
    parser.add_argument(
        "--distance-metric", choices=("euclidean", "cosine"), default="euclidean"
    )
    parser.add_argument("--permission-min-support", type=float, default=0.25)
    parser.add_argument("--obligation-min-support", type=float, default=0.75)
    parser.add_argument("--obligation-exclusivity-threshold", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = TEMMConfig(
        episodes=args.episodes,
        success_quantile=args.success_quantile,
        distance_metric=args.distance_metric,
        permission_min_support=args.permission_min_support,
        obligation_min_support=args.obligation_min_support,
        obligation_exclusivity_threshold=args.obligation_exclusivity_threshold,
        output_path=args.out,
        seed=args.seed,
    )

    experiment = reload_experiment_for_temm(str(Path(args.checkpoint).resolve()))
    rollouts = collect_evaluation_rollouts(experiment, config)
    result = analyze_rollouts(rollouts, experiment.group_map, config)

    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_json(output_path)
    summary_path = output_path.with_name("temm_summary.txt")
    write_summary(result, summary_path)
    print(f"TEMM results written to {output_path}")
    print(f"TEMM summary written to {summary_path}")


if __name__ == "__main__":
    main()
