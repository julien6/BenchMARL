#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

import copy
from typing import Callable, Dict, List, Optional

import torch
from torchrl.data import Composite
from torchrl.envs import EnvBase, PettingZooEnv, PettingZooWrapper

from benchmarl.environments.common import Task, TaskClass

from benchmarl.utils import DEVICE_TYPING


class PettingZooClass(TaskClass):
    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        config = copy.deepcopy(self.config)
        if self.supports_continuous_actions() and self.supports_discrete_actions():
            config.update({"continuous_actions": continuous_actions})
        if self.name == "ORBITAL":
            return lambda: PettingZooWrapper(
                categorical_actions=True,
                device=device,
                seed=seed,
                env=_get_orbital_env(config),
            )
        return lambda: PettingZooEnv(
            categorical_actions=True,
            device=device,
            seed=seed,
            parallel=True,
            return_state=self.has_state(),
            render_mode="rgb_array",
            **config
        )

    def supports_continuous_actions(self) -> bool:
        if self.name in {
            "MULTIWALKER",
            "WATERWORLD",
            "SIMPLE_ADVERSARY",
            "SIMPLE_CRYPTO",
            "SIMPLE_PUSH",
            "SIMPLE_REFERENCE",
            "SIMPLE_SPEAKER_LISTENER",
            "SIMPLE_SPREAD",
            "SIMPLE_TAG",
            "SIMPLE_WORLD_COMM",
        }:
            return True
        return False

    def supports_discrete_actions(self) -> bool:
        if self.name in {
            "SIMPLE_ADVERSARY",
            "SIMPLE_CRYPTO",
            "SIMPLE_PUSH",
            "SIMPLE_REFERENCE",
            "SIMPLE_SPEAKER_LISTENER",
            "SIMPLE_SPREAD",
            "SIMPLE_TAG",
            "SIMPLE_WORLD_COMM",
            "ORBITAL",
        }:
            return True
        return False

    def has_state(self) -> bool:
        if self.name in {
            "SIMPLE_ADVERSARY",
            "SIMPLE_CRYPTO",
            "SIMPLE_PUSH",
            "SIMPLE_REFERENCE",
            "SIMPLE_SPEAKER_LISTENER",
            "SIMPLE_SPREAD",
            "SIMPLE_TAG",
            "SIMPLE_WORLD_COMM",
        }:
            return True
        return False

    def has_render(self, env: EnvBase) -> bool:
        if self.name == "ORBITAL":
            return self.config["render_mode"] is not None
        return True

    def max_steps(self, env: EnvBase) -> int:
        if self.name == "ORBITAL":
            return self.config["max_steps"]
        return self.config["max_cycles"]

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        return env.group_map

    def state_spec(self, env: EnvBase) -> Optional[Composite]:
        if "state" in env.observation_spec:
            return Composite({"state": env.observation_spec["state"].clone()})
        return None

    def action_mask_spec(self, env: EnvBase) -> Optional[Composite]:
        observation_spec = env.observation_spec.clone()
        for group in self.group_map(env):
            group_obs_spec = observation_spec[group]
            for key in list(group_obs_spec.keys()):
                if key != "action_mask":
                    del group_obs_spec[key]
            if group_obs_spec.is_empty():
                del observation_spec[group]
        if "state" in observation_spec.keys():
            del observation_spec["state"]
        if observation_spec.is_empty():
            return None
        return observation_spec

    def observation_spec(self, env: EnvBase) -> Composite:
        observation_spec = env.observation_spec.clone()
        for group in self.group_map(env):
            group_obs_spec = observation_spec[group]
            for key in list(group_obs_spec.keys()):
                if key != "observation":
                    del group_obs_spec[key]
        if "state" in observation_spec.keys():
            del observation_spec["state"]
        return observation_spec

    def info_spec(self, env: EnvBase) -> Optional[Composite]:
        observation_spec = env.observation_spec.clone()
        for group in self.group_map(env):
            group_obs_spec = observation_spec[group]
            for key in list(group_obs_spec.keys()):
                if key != "info":
                    del group_obs_spec[key]
        if "state" in observation_spec.keys():
            del observation_spec["state"]
        return observation_spec

    def action_spec(self, env: EnvBase) -> Composite:
        return env.full_action_spec

    @staticmethod
    def env_name() -> str:
        return "pettingzoo"


class PettingZooTask(Task):
    """Enum for PettingZoo tasks."""

    MULTIWALKER = None
    WATERWORLD = None
    SIMPLE_ADVERSARY = None
    SIMPLE_CRYPTO = None
    SIMPLE_PUSH = None
    SIMPLE_REFERENCE = None
    SIMPLE_SPEAKER_LISTENER = None
    SIMPLE_SPREAD = None
    SIMPLE_TAG = None
    SIMPLE_WORLD_COMM = None
    ORBITAL = None

    @staticmethod
    def associated_class():
        return PettingZooClass


def _get_orbital_env(config):
    try:
        from orbital import parallel_env
    except ImportError:
        raise ImportError(
            "Module `orbital` not found, install ORBITAL with "
            "`pip install -e /path/to/ORBITAL`"
        )
    return _filter_info(parallel_env(**config))


def _filter_info(env):
    try:
        from pettingzoo.utils.wrappers import BaseParallelWrapper
    except ImportError:
        raise ImportError(
            "Module `pettingzoo` not found, install ORBITAL before using it "
            "with BenchMARL"
        )

    class TensorInfoWrapper(BaseParallelWrapper):
        @staticmethod
        def _filter(info_by_agent):
            return {
                agent: {
                    key: value
                    for key, value in info.items()
                    if _is_tensor_compatible(value)
                }
                for agent, info in info_by_agent.items()
            }

        def reset(self, seed=None, options=None):
            observation, info = super().reset(seed=seed, options=options)
            return observation, self._filter(info)

        def step(self, actions):
            observation, reward, terminated, truncated, info = super().step(actions)
            return (
                observation,
                reward,
                terminated,
                truncated,
                self._filter(info),
            )

    return TensorInfoWrapper(env)


def _is_tensor_compatible(value) -> bool:
    try:
        torch.as_tensor(value)
    except (RuntimeError, TypeError, ValueError):
        return False
    return True
