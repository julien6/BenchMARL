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
    representative_plan_semantic_for,
    symbolic_timeline,
    trajectory_action_histogram,
    trajectory_embedding,
    transition_pattern,
    transition_pattern_semantic,
)
from .semantic import resolve_semantic_adapter
from .types import (
    FitScores,
    InferredGoal,
    InferredMission,
    InferredNorm,
    InferredRole,
    TEMMDiagnostics,
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


def analyze_rollouts_with_diagnostics(
    rollouts: Sequence[TensorDictBase],
    group_map: Dict[str, Sequence[str]],
    config: TEMMConfig | None = None,
    semantic_adapter=None,
) -> tuple[TEMMResult, TEMMDiagnostics]:
    config = config or TEMMConfig()
    dataset = extract_trajectories(rollouts, group_map)
    return analyze_dataset_with_diagnostics(dataset, config, semantic_adapter)


def analyze_dataset(dataset: TEMMTrajectoryDataset, config: TEMMConfig) -> TEMMResult:
    result, _ = analyze_dataset_with_diagnostics(dataset, config)
    return result


def analyze_dataset_with_diagnostics(
    dataset: TEMMTrajectoryDataset, config: TEMMConfig, semantic_adapter=None
) -> tuple[TEMMResult, TEMMDiagnostics]:
    semantic_adapter = semantic_adapter or resolve_semantic_adapter(config)
    roles, role_labels, role_threshold, role_rep_threshold, sof = _infer_roles(
        dataset, config, semantic_adapter
    )
    goals, goal_by_step, goal_labels, goal_steps, goal_threshold, goal_rep_threshold, fof = _infer_goals(
        dataset, config, semantic_adapter
    )
    missions, episode_missions = _infer_missions(goal_by_step)
    permissions, obligations = _infer_norms(
        roles, role_labels, dataset, missions, episode_missions, config
    )

    organizational_fit = 0.5 * (sof + fof)
    returns = dataset.episode_returns
    mean_return = float(returns.mean().item()) if returns.numel() else 0.0
    reward_std = float(returns.std(unbiased=False).item()) if returns.numel() else 0.0
    result = TEMMResult(
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
            "semantic_adapter": semantic_adapter.name,
        },
        metadata={
            "episodes": int(returns.numel()),
            "agent_trajectories": len(dataset.agent_trajectories),
            "role_clusters": len(roles),
            "goal_clusters": len(goals),
            "missions": len(missions),
            "semantic_adapter": semantic_adapter.name,
        },
    )
    diagnostics = _build_diagnostics(
        dataset=dataset,
        roles=roles,
        role_labels=role_labels,
        goals=goals,
        goal_labels=goal_labels,
        goal_steps=goal_steps,
        missions=missions,
        permissions=permissions,
        obligations=obligations,
        config=config,
        semantic_adapter=semantic_adapter,
    )
    return result, diagnostics


def _infer_roles(
    dataset: TEMMTrajectoryDataset, config: TEMMConfig, semantic_adapter
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
                representative_pattern_semantic=transition_pattern_semantic(
                    medoid, semantic_adapter
                ),
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
    dataset: TEMMTrajectoryDataset, config: TEMMConfig, semantic_adapter
) -> tuple[
    List[InferredGoal],
    Dict[tuple[int, int], str],
    Tensor,
    list,
    float,
    float,
    float,
]:
    successful_steps = _successful_joint_steps(dataset, config.success_quantile)
    if not successful_steps:
        return [], {}, torch.empty(0, dtype=torch.long), [], 0.0, 0.0, 0.0
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
                representative_plan_semantic=representative_plan_semantic_for(
                    dataset,
                    medoid_step.episode_index,
                    medoid_step.time_index,
                    config.plan_window,
                    semantic_adapter,
                ),
            )
        )
    reach_consistency = _goal_reach_consistency(dataset, goal_by_step)
    fof = (1.0 - clustering.normalized_intra_variance) * reach_consistency
    return (
        goals,
        goal_by_step,
        clustering.labels,
        successful_steps,
        threshold,
        representativeness_threshold,
        _clamp01(fof),
    )


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


def _build_diagnostics(
    dataset: TEMMTrajectoryDataset,
    roles: List[InferredRole],
    role_labels: Tensor,
    goals: List[InferredGoal],
    goal_labels: Tensor,
    goal_steps: list,
    missions: List[InferredMission],
    permissions: List[InferredNorm],
    obligations: List[InferredNorm],
    config: TEMMConfig,
    semantic_adapter,
) -> TEMMDiagnostics:
    role_index_to_id = {int(role.id.rsplit("_", 1)[1]): role.id for role in roles}
    trajectory_timelines = {
        traj.id: symbolic_timeline(traj, semantic_adapter)
        for traj in dataset.agent_trajectories
    }
    role_embeddings = [
        trajectory_embedding(traj, max_action_bins=config.max_action_bins).tolist()
        for traj in dataset.agent_trajectories
    ]
    role_label_names = [
        role_index_to_id.get(int(label), f"role_{int(label)}")
        for label in role_labels.tolist()
    ]
    action_bin_labels = [f"action_{index}" for index in range(config.max_action_bins)]
    role_action_histograms = {
        role.id: [0.0 for _ in range(config.max_action_bins)] for role in roles
    }
    role_counts = {role.id: 0 for role in roles}
    role_members = {role.id: [] for role in roles}
    for traj, role_id in zip(dataset.agent_trajectories, role_label_names):
        if role_id not in role_action_histograms:
            continue
        role_members[role_id].append(traj.id)
        hist = trajectory_action_histogram(traj, config.max_action_bins).tolist()
        role_action_histograms[role_id] = [
            current + value
            for current, value in zip(role_action_histograms[role_id], hist)
        ]
        role_counts[role_id] += 1
    for role_id, count in role_counts.items():
        if count:
            role_action_histograms[role_id] = [
                value / count for value in role_action_histograms[role_id]
            ]

    goal_index_to_id = {int(goal.id.rsplit("_", 1)[1]): goal.id for goal in goals}
    goal_label_names = [
        goal_index_to_id.get(int(label), f"goal_{int(label)}")
        for label in goal_labels.tolist()
    ]
    goal_embeddings = [step.embedding.float().tolist() for step in goal_steps]
    role_ids = [role.id for role in roles]
    mission_ids = [mission.id for mission in missions]
    matrix = [[0.0 for _ in mission_ids] for _ in role_ids]
    norm_by_pair = {}
    for norm in [*permissions, *obligations]:
        norm_by_pair[(norm.role, norm.mission)] = max(
            norm_by_pair.get((norm.role, norm.mission), 0.0),
            float(norm.support),
        )
    for role_i, role_id in enumerate(role_ids):
        for mission_i, mission_id in enumerate(mission_ids):
            matrix[role_i][mission_i] = norm_by_pair.get((role_id, mission_id), 0.0)

    role_prototypes = {
        role.id: {
            "trajectory_id": role.medoid,
            "timeline": trajectory_timelines.get(role.medoid, []),
            "pattern": role.representative_pattern,
            "semantic_pattern": role.representative_pattern_semantic,
        }
        for role in roles
    }
    goal_prototypes = {
        goal.id: {
            "episode": goal.medoid_episode,
            "time": goal.medoid_time,
            "plan": goal.representative_plan,
            "semantic_plan": goal.representative_plan_semantic,
        }
        for goal in goals
    }
    role_distance_labels, role_distance_matrix = _prototype_distance_matrix(
        [role.id for role in roles],
        [role.representative_pattern for role in roles],
    )
    goal_distance_labels, goal_distance_matrix = _prototype_distance_matrix(
        [goal.id for goal in goals],
        [goal.representative_plan for goal in goals],
    )
    role_hierarchy_edges = _hierarchy_edges(
        [role.id for role in roles],
        [role.representative_pattern for role in roles],
    )
    goal_hierarchy_edges = _goal_hierarchy_edges(missions, goals)

    return TEMMDiagnostics(
        role_embeddings=role_embeddings,
        role_labels=role_label_names,
        role_trajectory_ids=[traj.id for traj in dataset.agent_trajectories],
        role_agent_names=[traj.agent_name for traj in dataset.agent_trajectories],
        role_episode_indices=[
            traj.episode_index for traj in dataset.agent_trajectories
        ],
        role_action_histograms=role_action_histograms,
        action_bin_labels=action_bin_labels,
        goal_embeddings=goal_embeddings,
        goal_labels=goal_label_names,
        goal_episode_indices=[step.episode_index for step in goal_steps],
        goal_time_indices=[step.time_index for step in goal_steps],
        role_mission_matrix=matrix,
        role_mission_roles=role_ids,
        role_mission_missions=mission_ids,
        trajectory_timelines=trajectory_timelines,
        role_members=role_members,
        role_prototypes=role_prototypes,
        goal_prototypes=goal_prototypes,
        role_distance_matrix=role_distance_matrix,
        role_distance_labels=role_distance_labels,
        goal_distance_matrix=goal_distance_matrix,
        goal_distance_labels=goal_distance_labels,
        role_hierarchy_edges=role_hierarchy_edges,
        goal_hierarchy_edges=goal_hierarchy_edges,
        semantic_action_map=semantic_adapter.action_map(),
        semantic_adapter=semantic_adapter.name,
    )


def _prototype_distance_matrix(
    labels: List[str], sequences: List[List[str]]
) -> tuple[List[str], List[List[float]]]:
    matrix = []
    for left in sequences:
        row = []
        for right in sequences:
            row.append(1.0 - _jaccard(left, right))
        matrix.append(row)
    return labels, matrix


def _hierarchy_edges(labels: List[str], sequences: List[List[str]]) -> List[Dict]:
    edges = []
    for child_index, child_sequence in enumerate(sequences):
        best_parent = None
        best_score = 0.0
        child_set = set(child_sequence)
        if not child_set:
            continue
        for parent_index, parent_sequence in enumerate(sequences):
            if parent_index == child_index:
                continue
            parent_set = set(parent_sequence)
            overlap = len(child_set & parent_set) / len(child_set)
            if overlap > best_score:
                best_score = overlap
                best_parent = labels[parent_index]
        if best_parent is not None and best_score > 0.0:
            edges.append(
                {
                    "parent": best_parent,
                    "child": labels[child_index],
                    "support": float(best_score),
                }
            )
    return edges


def _goal_hierarchy_edges(
    missions: List[InferredMission], goals: List[InferredGoal]
) -> List[Dict]:
    edges = []
    goal_ids = {goal.id for goal in goals}
    for mission in missions:
        for goal in mission.goals:
            if goal in goal_ids:
                edges.append(
                    {
                        "parent": mission.id,
                        "child": goal,
                        "support": float(mission.support),
                    }
                )
    return edges


def _jaccard(left: List[str], right: List[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set and not right_set:
        return 1.0
    union = left_set | right_set
    if not union:
        return 0.0
    return len(left_set & right_set) / len(union)


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
