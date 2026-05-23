#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Sequence

import torch
from tensordict import TensorDictBase
from torch import Tensor

from .clustering import choose_best_clustering
from .config import TEMMConfig
from .trajectory import (
    TEMMTrajectoryDataset,
    extract_trajectories,
    representative_plan_for,
    trajectory_embedding,
    transition_pattern,
)
from .types import (
    FitScores,
    InferredGoal,
    InferredMission,
    InferredNorm,
    InferredRole,
    TEMMResult,
)


def analyze_rollouts(
    rollouts: Sequence[TensorDictBase],
    group_map: Dict[str, Sequence[str]],
    config: TEMMConfig | None = None,
) -> TEMMResult:
    config = config or TEMMConfig()
    dataset = extract_trajectories(rollouts, group_map)
    return analyze_dataset(dataset, config)


def analyze_dataset(dataset: TEMMTrajectoryDataset, config: TEMMConfig) -> TEMMResult:
    roles, role_labels, role_threshold, role_rep_threshold, sof = _infer_roles(
        dataset, config
    )
    goals, goal_by_step, goal_threshold, goal_rep_threshold, fof = _infer_goals(
        dataset, config
    )
    missions, episode_missions = _infer_missions(goal_by_step)
    permissions, obligations = _infer_norms(
        roles, role_labels, dataset, missions, episode_missions, config
    )

    organizational_fit = 0.5 * (sof + fof)
    returns = dataset.episode_returns
    mean_return = float(returns.mean().item()) if returns.numel() else 0.0
    reward_std = float(returns.std(unbiased=False).item()) if returns.numel() else 0.0
    return TEMMResult(
        fit=FitScores(
            structural=float(sof),
            functional=float(fof),
            organizational=float(organizational_fit),
        ),
        mean_return=mean_return,
        reward_std=reward_std,
        roles=roles,
        goals=goals,
        missions=missions,
        permissions=permissions,
        obligations=obligations,
        selected_parameters={
            "distance_metric": config.distance_metric,
            "role_distance_threshold": role_threshold,
            "goal_distance_threshold": goal_threshold,
            "role_representativeness_threshold": role_rep_threshold,
            "goal_representativeness_threshold": goal_rep_threshold,
            "success_quantile": config.success_quantile,
        },
        metadata={
            "episodes": int(returns.numel()),
            "agent_trajectories": len(dataset.agent_trajectories),
            "role_clusters": len(roles),
            "goal_clusters": len(goals),
            "missions": len(missions),
        },
    )


def _infer_roles(
    dataset: TEMMTrajectoryDataset, config: TEMMConfig
) -> tuple[List[InferredRole], Tensor, float, float, float]:
    if not dataset.agent_trajectories:
        return [], torch.empty(0, dtype=torch.long), 0.0, 0.0, 0.0
    embeddings = torch.stack(
        [
            trajectory_embedding(traj, max_action_bins=config.max_action_bins)
            for traj in dataset.agent_trajectories
        ]
    )
    clustering, threshold = choose_best_clustering(
        embeddings,
        config.role_distance_thresholds,
        config.distance_metric,
        config.cluster_penalty,
    )
    representativeness_values = [
        _representativeness(float(variance.item())) for variance in clustering.variances
    ]
    representativeness_threshold = _select_representativeness_threshold(
        representativeness_values, config.representativeness_thresholds
    )
    roles = []
    for role_index, medoid_index in enumerate(clustering.medoid_indices.tolist()):
        member_indices = (clustering.labels == role_index).nonzero(as_tuple=True)[0]
        assigned_agents = sorted(
            {
                dataset.agent_trajectories[int(index)].agent_name
                for index in member_indices
            }
        )
        variance = float(clustering.variances[role_index].item())
        representativeness = _representativeness(variance)
        if representativeness < representativeness_threshold:
            continue
        medoid = dataset.agent_trajectories[medoid_index]
        roles.append(
            InferredRole(
                id=f"role_{role_index}",
                assigned_agents=assigned_agents,
                support=int(member_indices.numel()),
                variance=variance,
                representativeness=representativeness,
                medoid=medoid.id,
                representative_pattern=transition_pattern(medoid),
            )
        )
    sof = 1.0 - clustering.normalized_intra_variance
    return (
        roles,
        clustering.labels,
        threshold,
        representativeness_threshold,
        _clamp01(sof),
    )


def _infer_goals(
    dataset: TEMMTrajectoryDataset, config: TEMMConfig
) -> tuple[List[InferredGoal], Dict[tuple[int, int], str], float, float, float]:
    successful_steps = _successful_joint_steps(dataset, config.success_quantile)
    if not successful_steps:
        return [], {}, 0.0, 0.0, 0.0
    embeddings = torch.stack([step.embedding.float() for step in successful_steps])
    clustering, threshold = choose_best_clustering(
        embeddings,
        config.goal_distance_thresholds,
        config.distance_metric,
        config.cluster_penalty,
    )
    representativeness_values = [
        _representativeness(float(variance.item())) for variance in clustering.variances
    ]
    representativeness_threshold = _select_representativeness_threshold(
        representativeness_values, config.representativeness_thresholds
    )
    goals: List[InferredGoal] = []
    goal_by_step: Dict[tuple[int, int], str] = {}
    for goal_index, medoid_index in enumerate(clustering.medoid_indices.tolist()):
        member_indices = (clustering.labels == goal_index).nonzero(as_tuple=True)[0]
        variance = float(clustering.variances[goal_index].item())
        representativeness = _representativeness(variance)
        if representativeness < representativeness_threshold:
            continue
        medoid_step = successful_steps[medoid_index]
        goal_id = f"goal_{goal_index}"
        for index in member_indices:
            step = successful_steps[int(index)]
            goal_by_step[(step.episode_index, step.time_index)] = goal_id
        goals.append(
            InferredGoal(
                id=goal_id,
                support=int(member_indices.numel()),
                variance=variance,
                representativeness=representativeness,
                medoid_episode=medoid_step.episode_index,
                medoid_time=medoid_step.time_index,
                centroid=embeddings[member_indices].mean(0).tolist(),
                representative_plan=representative_plan_for(
                    dataset,
                    medoid_step.episode_index,
                    medoid_step.time_index,
                    config.plan_window,
                ),
            )
        )
    reach_consistency = _goal_reach_consistency(dataset, goal_by_step)
    fof = (1.0 - clustering.normalized_intra_variance) * reach_consistency
    return goals, goal_by_step, threshold, representativeness_threshold, _clamp01(fof)


def _infer_missions(
    goal_by_step: Dict[tuple[int, int], str]
) -> tuple[List[InferredMission], Dict[int, List[str]]]:
    goals_by_episode = defaultdict(set)
    for (episode_index, _), goal_id in goal_by_step.items():
        goals_by_episode[episode_index].add(goal_id)
    mission_counts = Counter(
        tuple(sorted(goal_ids))
        for goal_ids in goals_by_episode.values()
        if len(goal_ids) > 0
    )
    missions = [
        InferredMission(id=f"mission_{index}", goals=list(goals), support=support)
        for index, (goals, support) in enumerate(sorted(mission_counts.items()))
    ]
    mission_by_goals = {tuple(mission.goals): mission.id for mission in missions}
    episode_missions = {
        episode_index: [
            mission_by_goals[tuple(sorted(goal_ids))]
        ]
        for episode_index, goal_ids in goals_by_episode.items()
        if tuple(sorted(goal_ids)) in mission_by_goals
    }
    return missions, episode_missions


def _infer_norms(
    roles: List[InferredRole],
    role_labels: Tensor,
    dataset: TEMMTrajectoryDataset,
    missions: List[InferredMission],
    episode_missions: Dict[int, List[str]],
    config: TEMMConfig,
) -> tuple[List[InferredNorm], List[InferredNorm]]:
    if not roles or not missions or role_labels.numel() == 0:
        return [], []

    mission_ids = [mission.id for mission in missions]
    permissions: List[InferredNorm] = []
    obligations: List[InferredNorm] = []
    for role in roles:
        role_index = int(role.id.rsplit("_", 1)[1])
        member_indices = (role_labels == role_index).nonzero(as_tuple=True)[0]
        if member_indices.numel() == 0:
            continue
        mission_hits = Counter()
        for index in member_indices.tolist():
            traj = dataset.agent_trajectories[int(index)]
            for mission_id in episode_missions.get(traj.episode_index, []):
                mission_hits[mission_id] += 1
        total = int(member_indices.numel())
        support_by_mission = {
            mission_id: mission_hits[mission_id] / max(1, total)
            for mission_id in mission_ids
        }
        support_sum = sum(support_by_mission.values())
        for mission_id, support in support_by_mission.items():
            if support >= config.permission_min_support:
                permissions.append(
                    InferredNorm(
                        kind="PER",
                        role=role.id,
                        mission=mission_id,
                        temporal_constraint="Any",
                        support=float(support),
                    )
                )
            exclusivity = support / support_sum if support_sum > 0 else 0.0
            if (
                support >= config.obligation_min_support
                and exclusivity >= config.obligation_exclusivity_threshold
            ):
                obligations.append(
                    InferredNorm(
                        kind="OBL",
                        role=role.id,
                        mission=mission_id,
                        temporal_constraint="Any",
                        support=float(support),
                        exclusivity=float(exclusivity),
                    )
                )
    return permissions, obligations


def _successful_joint_steps(dataset: TEMMTrajectoryDataset, success_quantile: float):
    if dataset.episode_returns.numel() == 0:
        return []
    threshold = torch.quantile(dataset.episode_returns, success_quantile)
    successful_episodes = {
        int(index)
        for index in (dataset.episode_returns >= threshold).nonzero(as_tuple=True)[0]
    }
    return [
        step
        for step in dataset.joint_observations
        if step.episode_index in successful_episodes
    ]


def _goal_reach_consistency(
    dataset: TEMMTrajectoryDataset,
    goal_by_step: Dict[tuple[int, int], str],
) -> float:
    if dataset.episode_returns.numel() == 0:
        return 0.0
    reached = {
        episode_index
        for episode_index, _ in goal_by_step
    }
    return len(reached) / max(1, int(dataset.episode_returns.numel()))


def _representativeness(variance: float) -> float:
    return _clamp01(1.0 / (1.0 + max(0.0, variance)))


def _select_representativeness_threshold(
    values: Sequence[float], thresholds: Sequence[float]
) -> float:
    if not thresholds:
        return 0.0
    sorted_thresholds = sorted(float(threshold) for threshold in thresholds)
    for threshold in reversed(sorted_thresholds):
        if any(value >= threshold for value in values):
            return threshold
    return sorted_thresholds[0]


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))
