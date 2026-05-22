#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from itertools import product
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import torch
from tensordict import TensorDictBase
from torch import Tensor
from torchrl.data import Binary, Categorical, Composite, Unbounded
from torchrl.envs import Transform

from .core import GoalContext, OrganizationalModel


def _batch_indices(batch_shape: torch.Size) -> Iterable[Tuple[int, ...]]:
    if len(batch_shape) == 0:
        return [()]
    return product(*(range(size) for size in batch_shape))


def _leaf_at(tensor: Tensor, batch_index: Tuple[int, ...], agent_index: int) -> Tensor:
    return tensor[batch_index + (agent_index,)]


class MMARoleMaskTransform(Transform):
    """Write BenchMARL action masks from MMA role assignments."""

    def __init__(
        self,
        model: OrganizationalModel,
        group_map: Mapping[str, Sequence[str]],
        action_spec: Composite,
    ):
        super().__init__(in_keys=[], out_keys=[], in_keys_inv=[], out_keys_inv=[])
        self.model = model
        self.group_map = {group: tuple(agents) for group, agents in group_map.items()}
        self.action_spec = action_spec.clone()
        self.num_actions: Dict[str, int] = {}

        for group in self.group_map:
            group_action_spec = self.action_spec[group, "action"]
            if not isinstance(group_action_spec, Categorical):
                raise TypeError(
                    "MMA role masks currently require Categorical discrete "
                    f"actions. Group '{group}' uses {type(group_action_spec)}."
                )
            self.num_actions[group] = int(group_action_spec.n)

    def transform_observation_spec(self, observation_spec: Composite) -> Composite:
        observation_spec = observation_spec.clone()
        for group, num_actions in self.num_actions.items():
            observation_spec[group, "action_mask"] = Binary(
                n=num_actions,
                shape=(*observation_spec[group].shape, num_actions),
                device=self.action_spec.device,
                dtype=torch.bool,
            )
        return observation_spec

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        return self._write_masks(tensordict_reset)

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        return self._write_masks(next_tensordict)

    def _write_masks(self, tensordict: TensorDictBase) -> TensorDictBase:
        for group, agent_names in self.group_map.items():
            observations = tensordict.get((group, "observation"))
            existing = tensordict.get((group, "action_mask"), None)
            mask = torch.ones(
                (*observations.shape[:-1], self.num_actions[group]),
                dtype=torch.bool,
                device=observations.device,
            )

            batch_shape = observations.shape[:-2]
            for batch_index in _batch_indices(batch_shape):
                for agent_index, agent_name in enumerate(agent_names):
                    role = self.model.role_for(agent_name)
                    if role is None:
                        continue
                    allowed_actions = tuple(
                        int(action)
                        for action in role.allowed_actions(
                            _leaf_at(observations, batch_index, agent_index), agent_name
                        )
                    )
                    if not allowed_actions:
                        raise ValueError(
                            f"MMA role assigned to '{agent_name}' allowed no actions."
                        )
                    if (
                        min(allowed_actions) < 0
                        or max(allowed_actions) >= self.num_actions[group]
                    ):
                        raise ValueError(
                            f"MMA role assigned to '{agent_name}' returned an action "
                            f"outside [0, {self.num_actions[group] - 1}]."
                        )
                    agent_mask = mask[batch_index + (agent_index,)]
                    agent_mask.fill_(False)
                    agent_mask[list(allowed_actions)] = True

            if existing is not None:
                mask &= existing.to(dtype=torch.bool, device=mask.device)
            if not mask.any(dim=-1).all():
                raise ValueError(
                    f"MMA roles and environment mask leave no legal action for group '{group}'."
                )
            tensordict.set((group, "action_mask"), mask)
        return tensordict


class MMAGoalRewardTransform(Transform):
    """Add MMA goal shaping to group rewards and log shaping diagnostics."""

    def __init__(
        self,
        model: OrganizationalModel,
        group_map: Mapping[str, Sequence[str]],
        reward_spec: Composite,
    ):
        super().__init__(in_keys=[], out_keys=[], in_keys_inv=[], out_keys_inv=[])
        self.model = model
        self.group_map = {group: tuple(agents) for group, agents in group_map.items()}
        self.reward_spec = reward_spec.clone()
        self.contexts: Dict[tuple, GoalContext] = {}
        self.reward_feature_shapes = {}

        for group in self.group_map:
            if (group, "reward") not in self.reward_spec.keys(True, True):
                raise ValueError(
                    "MMA goals require per-group rewards. "
                    f"Group '{group}' has no nested reward spec."
                )
            reward_spec = self.reward_spec[group, "reward"]
            self.reward_feature_shapes[group] = reward_spec.shape[
                len(self.reward_spec[group].shape) :
            ]

    def transform_observation_spec(self, observation_spec: Composite) -> Composite:
        observation_spec = observation_spec.clone()
        for group in self.group_map:
            reward_spec = self.reward_spec[group, "reward"]
            info_shape = (
                *observation_spec[group].shape,
                *self.reward_feature_shapes[group],
            )
            if "info" not in observation_spec[group].keys():
                observation_spec[group, "info"] = Composite(
                    shape=observation_spec[group].shape,
                    device=observation_spec.device,
                )
            observation_spec[group, "info", "mma_goal_bonus"] = Unbounded(
                shape=info_shape,
                device=reward_spec.device,
                dtype=reward_spec.dtype,
            )
            observation_spec[group, "info", "mma_raw_reward"] = Unbounded(
                shape=info_shape,
                device=reward_spec.device,
                dtype=reward_spec.dtype,
            )
        return observation_spec

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        self.contexts.clear()
        for group in self.group_map:
            observations = tensordict_reset.get((group, "observation"))
            zeros = torch.zeros(
                (*observations.shape[:-1], *self.reward_feature_shapes[group]),
                dtype=self.reward_spec[group, "reward"].dtype,
                device=observations.device,
            )
            tensordict_reset.set((group, "info", "mma_goal_bonus"), zeros.clone())
            tensordict_reset.set((group, "info", "mma_raw_reward"), zeros)
        return tensordict_reset

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        for group, agent_names in self.group_map.items():
            reward_key = (group, "reward")
            reward = next_tensordict.get(reward_key, None)
            if reward is None:
                raise ValueError(
                    f"MMA goals require step reward at key ('{group}', 'reward')."
                )

            raw_reward = reward.clone()
            bonus = torch.zeros_like(reward)
            observations = tensordict.get((group, "observation"))
            actions = tensordict.get((group, "action"))
            batch_shape = observations.shape[:-2]

            for batch_index in _batch_indices(batch_shape):
                for agent_index, agent_name in enumerate(agent_names):
                    goal_bonus = bonus[batch_index + (agent_index,)]
                    for goal_name, goal in self.model.goals_for(agent_name):
                        context_key = batch_index + (group, agent_name, goal_name)
                        context = self.contexts.setdefault(context_key, GoalContext())
                        shaped = goal.reward(
                            context,
                            _leaf_at(observations, batch_index, agent_index),
                            _leaf_at(actions, batch_index, agent_index),
                            agent_name,
                        )
                        goal_bonus += torch.as_tensor(
                            shaped, dtype=reward.dtype, device=reward.device
                        )

            next_tensordict.set(reward_key, raw_reward + bonus)
            next_tensordict.set((group, "info", "mma_goal_bonus"), bonus)
            next_tensordict.set((group, "info", "mma_raw_reward"), raw_reward)
        return next_tensordict
