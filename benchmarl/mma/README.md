# BenchMARL MMA

`benchmarl.mma` provides the first BenchMARL-native MOISE+MARL integration.
An `OrganizationalModel` contains Python `Role` and `Goal` objects plus
agent-name assignments. Roles write discrete action masks before BenchMARL
samples policy actions. Goals add per-agent reward shaping after each
environment step.

Register code-defined models with an id:

```python
from benchmarl.mma import OrganizationalModel, register_organizational_model


def my_model(task, group_map):
    return OrganizationalModel(...)


register_organizational_model("my_model", my_model)
```

Hydra experiments select that id through the experiment config:

```bash
python benchmarl/run.py \
  algorithm=mappo \
  task=pettingzoo/orbital \
  experiment.organizational_model=orbital_partial
```

ORBITAL ships prototype assignment ids `orbital_none`, `orbital_partial`, and
`orbital_all`. `orbital_none` is classic MARL without organizational guidance.

Article-oriented MAPPO baselines use these ids:

| Id | Roles | Goals | Meaning |
| --- | --- | --- | --- |
| `orbital_lb_reward_only` | no | yes | free actions with mission reward shaping |
| `orbital_lb_action_only` | yes | no | role action masks without mission shaping |
| `orbital_mma_full` | yes | yes | role action masks and mission goal shaping |

The article-oriented ids use the ORBITAL roles `orbital_observer_role`,
`orbital_relay_role`, and `orbital_safety_guard_role`, with the mission goals
`orbital_task_acquisition_goal`, `orbital_data_delivery_goal`, and
`orbital_fleet_resilience_goal`. Keep one id fixed when a W&B sweep tunes
hyperparameters for one baseline. Use a separate sweep with a different
`experiment.organizational_model` value when comparing organization choices.
