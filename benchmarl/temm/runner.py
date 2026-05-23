#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from pathlib import Path
from typing import List

import torch
from tensordict import TensorDictBase
from torchrl.envs.utils import ExplorationType, set_exploration_type

from benchmarl.utils import seed_everything

from .analysis import analyze_rollouts
from .config import TEMMConfig
from .types import TEMMResult
from .wandb import publish_temm_to_wandb


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


def write_summary(result: TEMMResult, path: Path) -> None:
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


def run_temm_for_experiment(
    experiment,
    config: TEMMConfig,
    output_path: Path,
    publish_wandb: bool = True,
    create_wandb_section: bool = True,
) -> TEMMResult:
    print(f"Collecting {config.episodes} TEMM evaluation episodes...", flush=True)
    rollouts = collect_evaluation_rollouts(experiment, config)
    print("Analyzing TEMM trajectories...", flush=True)
    result = analyze_rollouts(rollouts, experiment.group_map, config)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_json(output_path)
    summary_path = output_path.with_name("temm_summary.txt")
    write_summary(result, summary_path)
    print(f"TEMM results written to {output_path}", flush=True)
    print(f"TEMM summary written to {summary_path}", flush=True)

    if publish_wandb and create_wandb_section:
        entity = experiment.config.wandb_extra_kwargs.get("entity")
        published = publish_temm_to_wandb(
            result=result,
            result_path=output_path,
            summary_path=summary_path,
            project=experiment.config.project_name,
            entity=entity,
            run_id=experiment.name,
            run_name=experiment.name,
            run_dir=experiment.folder_name,
            organizational_model_id=getattr(
                experiment.config, "organizational_model", None
            ),
            organizational_model=getattr(experiment, "organizational_model", None),
        )
        if published:
            print("TEMM & MOISE+MARL section published to W&B.", flush=True)
    return result
