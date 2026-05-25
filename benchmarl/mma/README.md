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
  experiment.organizational_model=lb_moise_marl
```

ORBITAL baselines use these ids:

| Id | Roles | Goals | Meaning |
| --- | --- | --- | --- |
| `handcrafted` | `handcrafted_full` | no | fully scripted handcrafted policy |
| `lb_unconstrained` | no | no | classic MARL without organizational guidance |
| `lb_moise_marl` | partial `acquirer`, `deliverer`, `stabilizer` | yes | partial role shielding plus reward shaping |
| `lb_action_only` | partial `acquirer`, `deliverer`, `stabilizer` | no | partial role shielding only |
| `lb_reward_only` | no | yes | reward shaping only |
| `rb_deliverer` | full `deliverer`, partial `acquirer` and `stabilizer` | no | delivery-biased role shielding |
| `rb_dcop_like` | full `acquirer` and `deliverer`, partial `stabilizer` | no | stronger planning-like acquisition/delivery shielding |
| `rb_acquirer` | full `acquirer`, partial `deliverer` and `stabilizer` | no | acquisition-biased role shielding |

Partial roles use `constraint hardness = 0.3`: when sampled, the role imposes
its handcrafted action; otherwise the neural policy can choose freely. Full
roles use `constraint hardness = 1.0` and always impose the role action.

The goal catalog mirrors the three role logics: `acquirer_goal`,
`deliverer_goal`, and `stabilizer_goal`. Each goal gives a positive bonus when
the action follows the corresponding handcrafted role logic, and a small malus
when a relevant handcrafted action was available but not selected.
