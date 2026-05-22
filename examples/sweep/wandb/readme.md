# ORBITAL MAPPO HPO with W&B Sweeps

This folder now contains the constrained sweep used to tune `MAPPO` on the
default `pettingzoo/orbital` task. The ORBITAL scenario is kept fixed during
the sweep. Only training hyperparameters are searched, then shortlisted
configurations are validated across seeds and stress variants before they are
promoted to `fine_tuned/pettingzoo_orbital/conf/config.yaml`.

## Prerequisites

- Install BenchMARL plus W&B media support in the training environment:

```bash
pip install 'wandb[media]'
```
- Install ORBITAL with the 2D renderer from the local clone before launching
  PettingZoo runs:

```bash
pip install -e '/home/julien/Documents/ORBITAL[render]'
```

- ORBITAL videos require both `task.render_mode=rgb_array` and
  `experiment.render=true`. The fine-tuned config and committed sweep enable
  those switches so evaluation videos are sent to W&B.
- W&B parameters use dotted Hydra override names such as `experiment.lr`.

## Logged ORBITAL diagnostics

The ORBITAL PettingZoo wrapper exposes numeric mission info under keys such as:

- `collection/sat/info/delivered_total`
- `collection/sat/info/observed_total`
- `collection/sat/info/knowledge_shared`
- `collection/sat/info/ground_route`
- `collection/sat/info/reward_component_delivery`
- `collection/sat/info/reward_component_task`
- `collection/sat/info/reward_component_energy`
- `collection/sat/info/reward_component_debris_risk`

Use them with the BenchMARL timers and counters to reject apparently good runs
whose mission dynamics are unhealthy:

- `timers/collection_time`
- `timers/training_time`
- `timers/iteration_time`
- `counters/current_frames`
- `counters/total_frames`

MAPPO already logs PPO diagnostics during training:

- `train/sat/kl_approx`
- `train/sat/entropy`
- `train/sat/clip_fraction`
- `train/sat/explained_variance`
- `train/sat/grad_norm_loss_objective`
- `train/sat/grad_norm_loss_critic`

The W&B sweep objective remains the global evaluation metric:

```text
eval/reward/episode_reward_mean
```

## Phase A: baseline and CPU collection profile

Start with the balanced CPU-bound profile on the default ORBITAL task:

```bash
python benchmarl/run.py \
  algorithm=mappo \
  task=pettingzoo/orbital \
  seed=0 \
  task.render_mode=rgb_array \
  experiment.sampling_device=cpu \
  experiment.train_device=cuda \
  experiment.buffer_device=cpu \
  experiment.parallel_collection=true \
  experiment.prefer_continuous_actions=false \
  experiment.on_policy_n_envs_per_worker=18 \
  experiment.on_policy_collected_frames_per_batch=8208 \
  experiment.on_policy_minibatch_size=1024 \
  experiment.on_policy_n_minibatch_iters=20 \
  experiment.max_n_frames=500000 \
  experiment.evaluation_interval=41040 \
  experiment.evaluation_episodes=16 \
  experiment.evaluation_static=true \
  experiment.render=true
```

Compare one short profiling run against the safe collection profile by replacing:

```bash
experiment.on_policy_n_envs_per_worker=9 \
experiment.on_policy_collected_frames_per_batch=4104 \
experiment.on_policy_minibatch_size=512
```

Keep the balanced profile unless it loses clear stability or wall-clock
efficiency. It already occupies most of the 20 physical CPU cores, so run one
balanced W&B agent at a time on the target host.

Keep `on_policy_collected_frames_per_batch` divisible by
`on_policy_n_envs_per_worker`. BenchMARL evaluates only when collected
`total_frames` lands exactly on `evaluation_interval`, so the balanced profile
uses `8208 = 18 * 456` and evaluates every `41040 = 5 * 8208` frames.

For pure throughput profiling or reward-only sensitivity runs, disable videos
explicitly:

```bash
task.render_mode=null experiment.render=false
```

## Phase B: controlled 1D sensitivity runs

Run seed `0` for `750000` frames per point before relying on the Bayesian sweep.
Vary only one axis at a time from the baseline profile:

| Axis | Values |
| --- | --- |
| `experiment.lr` | `3e-5`, `1e-4`, `3e-4`, `6e-4` |
| `algorithm.entropy_coef` | `0.0`, `0.002`, `0.01`, `0.03` |
| `algorithm.clip_epsilon` | `0.1`, `0.2`, `0.3` |
| `experiment.on_policy_n_minibatch_iters` | `5`, `10`, `20`, `30` |
| `experiment.gamma` | `0.97`, `0.99`, `0.995` |

Drop unstable regions before widening the final search.

## Phase C: initialize the structured sweep

Review `sweepconfig.yaml`, then initialize the 18-run W&B sweep:

```bash
cd examples/sweep/wandb
wandb sweep sweepconfig.yaml
```

W&B prints the agent command for the new sweep:

```bash
wandb agent ENTITY/PROJECT/SWEEP_ID
```

The committed sweep uses:

- Bayesian search with Hyperband early termination.
- `MAPPO` and `pettingzoo/orbital`.
- A fixed balanced CPU collection profile.
- The W&B logger only, so W&B media support handles sweep videos.
- Search over learning rate, entropy, PPO clip, discounting, GAE lambda, and
  on-policy optimization pressure.

The sweep sets `experiment.organizational_model=orbital_none` explicitly. Keep
the model id fixed within a HPO sweep. For article-oriented MAPPO comparisons,
create sibling sweeps with `orbital_lb_reward_only`, `orbital_lb_action_only`,
and `orbital_mma_full`; each changes only the fixed organizational-model id.

Shortlist the two strongest non-collapsing configurations by the trailing
evaluation behavior, not by a single reward spike.

## Phase D: validate finalists

The completed default-MAPPO sweep synced from spark is post-processed locally
from BenchMARL JSON files, not from a single W&B summary point. Reproduce the
shortlist with:

```bash
cd /home/julien/Documents/BenchMARL
python examples/sweep/wandb/orbital_hpo_postprocess.py shortlist
```

The committed shortlist in `orbital_finalists.yaml` archives:

- finalist `A` from run `2026-05-22/15-07-27`
- finalist `B` from run `2026-05-22/13-14-48`
- fallback `C` from run `2026-05-22/14-15-09`

The shortlist score is the mean of the final five evaluation means. Before
launching long validation, inspect `A`, `B`, and fallback `C` in W&B with the
ORBITAL mission metrics, PPO diagnostics, and evaluation videos above.

Print the 10 default-task multi-seed commands for finalists `A` and `B`:

```bash
python examples/sweep/wandb/orbital_hpo_postprocess.py \
  validation-commands --mode default
```

Run those commands sequentially on the training host:

```bash
python examples/sweep/wandb/orbital_hpo_postprocess.py \
  validation-commands --mode default --run
```

Then print or execute the 18 stress commands:

```bash
python examples/sweep/wandb/orbital_hpo_postprocess.py \
  validation-commands --mode stress
python examples/sweep/wandb/orbital_hpo_postprocess.py \
  validation-commands --mode stress --run
```

Each generated command pins `experiment.organizational_model=orbital_none`,
uses 3M frames and 32 evaluation episodes, and adds W&B tags for candidate,
scenario, and seed. Stress validation keeps the same final hyperparameters and
changes only one ORBITAL task knob per scenario:

| Scenario | Override |
| --- | --- |
| `network_stress` | `task.p_link_drop=0.15` |
| `cyber_stress` | `task.adversarial_rate=0.10` |
| `resource_stress` | `task.energy_budget=32.0` |

After validation outputs are synced locally, summarize a validation output
folder with:

```bash
python examples/sweep/wandb/orbital_hpo_postprocess.py \
  validation-summary --outputs PATH_TO_SYNCED_VALIDATION_OUTPUTS
```

Promote the configuration with the best weighted rank: default ORBITAL counts
twice and each stress variant counts once. Break ties with default mean return,
then default variance, then delivery and fleet-health diagnostics. Only then
replace the placeholder hyperparameters in the fine-tuned config.

## Fine-tuned config

`fine_tuned/pettingzoo_orbital/conf/config.yaml` is the promotion target. It
ships with the balanced host profile and conservative placeholder MAPPO values
until a validated sweep winner replaces:

- `experiment.lr`
- `experiment.gamma`
- `experiment.on_policy_n_minibatch_iters`
- `algorithm.entropy_coef`
- `algorithm.clip_epsilon`
- `algorithm.lmbda`

## References

- https://docs.wandb.ai/guides/sweeps
- https://docs.wandb.ai/guides/sweeps/sweep-config-keys
