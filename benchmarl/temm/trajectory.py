#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence

import torch
from tensordict import TensorDictBase
from torch import Tensor


@dataclass
class AgentTrajectory:
    episode_index: int
    group: str
    agent_index: int
    agent_name: str
    observations: Tensor
    actions: Tensor
    next_observations: Tensor
    rewards: Tensor
    dones: Tensor

    @property
    def id(self) -> str:
        return f"episode_{self.episode_index}/{self.group}/{self.agent_name}"

    @property
    def episode_return(self) -> float:
        return float(self.rewards.sum().item())


@dataclass
class JointObservationStep:
    episode_index: int
    time_index: int
    embedding: Tensor


@dataclass
class TEMMTrajectoryDataset:
    agent_trajectories: List[AgentTrajectory]
    joint_observations: List[JointObservationStep]
    episode_returns: Tensor
    group_map: Dict[str, List[str]]


def extract_trajectories(
    rollouts: Sequence[TensorDictBase],
    group_map: Mapping[str, Sequence[str]],
) -> TEMMTrajectoryDataset:
    group_map = {group: list(agents) for group, agents in group_map.items()}
    agent_trajectories: List[AgentTrajectory] = []
    joint_observations: List[JointObservationStep] = []
    episode_returns = []

    for episode_index, rollout in enumerate(rollouts):
        rollout = _trim_at_first_done(rollout)
        group_returns = []
        time_len = int(rollout.batch_size[0])

        for time_index in range(time_len):
            joint_parts = []
            for group in group_map:
                observations = rollout.get((group, "observation"))
                joint_parts.append(observations[time_index].reshape(-1).float().cpu())
            joint_observations.append(
                JointObservationStep(
                    episode_index=episode_index,
                    time_index=time_index,
                    embedding=torch.cat(joint_parts) if joint_parts else torch.empty(0),
                )
            )

        for group, agent_names in group_map.items():
            observations = rollout.get((group, "observation")).detach().cpu()
            next_observations = (
                rollout.get(("next", group, "observation")).detach().cpu()
            )
            actions = rollout.get((group, "action")).detach().cpu()
            rewards = _get_group_value(rollout, group, "reward").detach().cpu()
            dones = _get_group_value(rollout, group, "done").detach().cpu()
            group_returns.append(rewards.sum(0).mean())

            for agent_index, agent_name in enumerate(agent_names):
                agent_trajectories.append(
                    AgentTrajectory(
                        episode_index=episode_index,
                        group=group,
                        agent_index=agent_index,
                        agent_name=agent_name,
                        observations=observations[:, agent_index],
                        actions=actions[:, agent_index],
                        next_observations=next_observations[:, agent_index],
                        rewards=rewards[:, agent_index].reshape(time_len, -1),
                        dones=dones[:, agent_index].reshape(time_len, -1),
                    )
                )

        if group_returns:
            episode_returns.append(torch.stack(group_returns).mean())

    returns = (
        torch.stack(episode_returns).float()
        if episode_returns
        else torch.empty(0, dtype=torch.float)
    )
    return TEMMTrajectoryDataset(
        agent_trajectories=agent_trajectories,
        joint_observations=joint_observations,
        episode_returns=returns,
        group_map=group_map,
    )


def trajectory_embedding(
    trajectory: AgentTrajectory,
    max_action_bins: int = 32,
) -> Tensor:
    observations = trajectory.observations.float().reshape(
        trajectory.observations.shape[0], -1
    )
    next_observations = trajectory.next_observations.float().reshape(
        trajectory.next_observations.shape[0], -1
    )
    delta = next_observations - observations
    action_hist = trajectory_action_histogram(trajectory, max_action_bins)
    return torch.cat(
        [
            observations.mean(0),
            observations.std(0, unbiased=False),
            delta.mean(0),
            delta.std(0, unbiased=False),
            action_hist,
        ]
    )


def trajectory_action_histogram(
    trajectory: AgentTrajectory, max_action_bins: int = 32
) -> Tensor:
    return _action_histogram(trajectory.actions, max_action_bins)


def transition_pattern(
    trajectory: AgentTrajectory,
    max_items: int = 8,
) -> List[str]:
    observations = trajectory.observations.float().reshape(
        trajectory.observations.shape[0], -1
    )
    next_observations = trajectory.next_observations.float().reshape(
        trajectory.next_observations.shape[0], -1
    )
    actions = trajectory.actions.reshape(trajectory.actions.shape[0], -1)
    items = []
    stride = max(1, int(len(observations) / max(1, max_items)))
    for time_index in range(0, len(observations), stride):
        obs_bucket = _bucket(float(observations[time_index].mean().item()))
        delta = next_observations[time_index] - observations[time_index]
        delta_bucket = _bucket(float(delta.mean().item()))
        action_value = _action_label(actions[time_index])
        items.append(f"obs:{obs_bucket}|act:{action_value}|delta:{delta_bucket}")
        if len(items) >= max_items:
            break
    return items


def symbolic_timeline(trajectory: AgentTrajectory) -> List[Dict[str, object]]:
    observations = trajectory.observations.float().reshape(
        trajectory.observations.shape[0], -1
    )
    next_observations = trajectory.next_observations.float().reshape(
        trajectory.next_observations.shape[0], -1
    )
    actions = trajectory.actions.reshape(trajectory.actions.shape[0], -1)
    rewards = trajectory.rewards.reshape(trajectory.rewards.shape[0], -1)
    items = []
    for time_index in range(len(observations)):
        delta = next_observations[time_index] - observations[time_index]
        items.append(
            {
                "t": time_index,
                "observation": _bucket(float(observations[time_index].mean().item())),
                "action": _action_label(actions[time_index]),
                "delta": _bucket(float(delta.mean().item())),
                "reward": float(rewards[time_index].mean().item()),
            }
        )
    return items


def representative_plan_for(
    dataset: TEMMTrajectoryDataset,
    episode_index: int,
    time_index: int,
    plan_window: int,
) -> List[str]:
    episode_agent_trajs = [
        traj
        for traj in dataset.agent_trajectories
        if traj.episode_index == episode_index
    ]
    if not episode_agent_trajs:
        return []
    start = max(0, time_index - plan_window)
    items = []
    for traj in episode_agent_trajs:
        observations = traj.observations.float().reshape(traj.observations.shape[0], -1)
        actions = traj.actions.reshape(traj.actions.shape[0], -1)
        for step in range(start, min(time_index + 1, len(observations))):
            obs_bucket = _bucket(float(observations[step].mean().item()))
            action_label = _action_label(actions[step])
            items.append(
                f"{traj.group}/{traj.agent_name}:obs:{obs_bucket}|act:{action_label}"
            )
    return items


def _trim_at_first_done(rollout: TensorDictBase) -> TensorDictBase:
    done = rollout.get(("next", "done"), None)
    if done is None:
        return rollout
    done = done.squeeze(-1)
    done_index = done.nonzero(as_tuple=True)[0]
    if done_index.numel() == 0:
        return rollout
    return rollout[: int(done_index[0].item()) + 1]


def _get_group_value(rollout: TensorDictBase, group: str, key: str) -> Tensor:
    value = rollout.get(("next", group, key), None)
    if value is not None:
        return value
    value = rollout.get(("next", key))
    group_shape = rollout.get(group).shape
    return value.expand(group_shape).unsqueeze(-1)


def _action_histogram(actions: Tensor, max_action_bins: int) -> Tensor:
    flat = actions.reshape(actions.shape[0], -1).float()
    if flat.shape[-1] == 1 and torch.allclose(flat, flat.round()):
        bins = torch.zeros(max_action_bins, dtype=torch.float)
        indices = flat.long().flatten().clamp(min=0, max=max_action_bins - 1)
        bins.scatter_add_(0, indices, torch.ones_like(indices, dtype=torch.float))
        return bins / bins.sum().clamp_min(1.0)
    return torch.cat([flat.mean(0), flat.std(0, unbiased=False)])


def _action_label(action: Tensor) -> str:
    action = action.flatten()
    if action.numel() == 1 and float(action[0].item()).is_integer():
        return str(int(action[0].item()))
    return ",".join(f"{float(value):.2f}" for value in action[:4])


def _bucket(value: float) -> str:
    if value < -0.33:
        return "low"
    if value > 0.33:
        return "high"
    return "mid"
