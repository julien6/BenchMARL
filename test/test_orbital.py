#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

import pytest

from benchmarl.algorithms import IppoConfig, MappoConfig, QmixConfig
from benchmarl.environments import PettingZooTask, Task
from benchmarl.experiment import Experiment

from utils import _has_orbital


@pytest.mark.skipif(not _has_orbital, reason="ORBITAL not found")
class TestOrbital:
    @pytest.mark.parametrize("algo_config", [IppoConfig, QmixConfig])
    @pytest.mark.parametrize("task", [PettingZooTask.ORBITAL])
    def test_discrete_algos(
        self,
        algo_config,
        task: Task,
        experiment_config,
        mlp_sequence_config,
    ):
        experiment_config.render = False
        experiment = Experiment(
            algorithm_config=algo_config.get_from_yaml(),
            model_config=mlp_sequence_config,
            seed=0,
            config=experiment_config,
            task=task.get_from_yaml(),
        )
        experiment.run()

    def test_mma_ppo_smoke(
        self,
        experiment_config,
        mlp_sequence_config,
    ):
        experiment_config.render = False
        experiment_config.organizational_model = "lb_moise_marl"
        experiment = Experiment(
            algorithm_config=MappoConfig.get_from_yaml(),
            model_config=mlp_sequence_config,
            seed=0,
            config=experiment_config,
            task=PettingZooTask.ORBITAL.get_from_yaml(),
        )

        assert experiment.action_mask_spec is not None
        assert ("sat", "action_mask") in experiment.action_mask_spec.keys(True, True)

        policy_td = experiment.policy(experiment.test_env.reset())
        chosen_action_is_allowed = policy_td["sat", "action_mask"].gather(
            -1, policy_td["sat", "action"].unsqueeze(-1)
        )
        assert chosen_action_is_allowed.all()

        experiment.run()
