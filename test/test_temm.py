#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

import json
import sys

import torch
from tensordict import TensorDict

from benchmarl.temm import (
    TEMMConfig,
    TEMMResult,
    analyze_rollouts,
    extract_trajectories,
)
from benchmarl.temm_analyze import main as temm_main

GROUP_MAP = {"agents": ["agent_0", "agent_1"]}


def _rollout(success=True, offset=0.0):
    time = 5
    obs = torch.zeros(time, 2, 2)
    next_obs = torch.zeros(time, 2, 2)
    obs[:, 0, :] = torch.tensor([0.0 + offset, 0.1])
    obs[:, 1, :] = torch.tensor([2.0 + offset, 2.1])
    if success:
        obs[-2:, :, :] += 3.0
    else:
        obs[-2:, :, :] -= 3.0
    next_obs[:-1] = obs[1:]
    next_obs[-1] = obs[-1]
    actions = torch.zeros(time, 2, dtype=torch.long)
    actions[:, 1] = 1
    reward_value = 1.0 if success else 0.0
    reward = torch.full((time, 2, 1), reward_value)
    done = torch.zeros(time, 1, dtype=torch.bool)
    done[-1] = True
    group_done = done.unsqueeze(-2).expand(time, 2, 1)
    return TensorDict(
        {
            "agents": TensorDict(
                {"observation": obs, "action": actions},
                batch_size=(time, 2),
            ),
            "next": TensorDict(
                {
                    "agents": TensorDict(
                        {
                            "observation": next_obs,
                            "reward": reward,
                            "done": group_done,
                        },
                        batch_size=(time, 2),
                    ),
                    "done": done,
                },
                batch_size=(time,),
            ),
        },
        batch_size=(time,),
    )


def _config(**kwargs):
    params = {
        "episodes": 4,
        "success_quantile": 0.5,
        "role_distance_thresholds": (0.3,),
        "goal_distance_thresholds": (0.3,),
        "cluster_penalty": 0.0,
    }
    params.update(kwargs)
    return TEMMConfig(**params)


def test_trajectory_extraction_handles_group_agent_dimensions():
    dataset = extract_trajectories([_rollout(success=True)], GROUP_MAP)

    assert len(dataset.agent_trajectories) == 2
    assert len(dataset.joint_observations) == 5
    assert dataset.agent_trajectories[0].observations.shape == (5, 2)
    assert dataset.agent_trajectories[1].actions.tolist() == [1, 1, 1, 1, 1]
    assert dataset.episode_returns.tolist() == [5.0]


def test_temm_clusters_two_stable_behavioral_roles():
    rollouts = [_rollout(True, 0.01), _rollout(True, 0.02), _rollout(True, 0.03)]
    result = analyze_rollouts(rollouts, GROUP_MAP, _config(success_quantile=0.0))

    assert len(result.roles) == 2
    role_agents = {tuple(role.assigned_agents) for role in result.roles}
    assert ("agent_0",) in role_agents
    assert ("agent_1",) in role_agents


def test_goal_inference_uses_successful_trajectories_only():
    rollouts = [_rollout(True, 0.0), _rollout(True, 0.1), _rollout(False, 0.0)]
    result = analyze_rollouts(rollouts, GROUP_MAP, _config(success_quantile=0.67))

    assert result.goals
    assert {goal.medoid_episode for goal in result.goals}.issubset({0, 1})


def test_fit_scores_are_bounded_and_norm_thresholds_are_applied():
    rollouts = [_rollout(True, 0.01), _rollout(True, 0.02), _rollout(True, 0.03)]
    result = analyze_rollouts(
        rollouts,
        GROUP_MAP,
        _config(
            success_quantile=0.0,
            permission_min_support=0.5,
            obligation_min_support=0.5,
            obligation_exclusivity_threshold=0.5,
        ),
    )

    assert 0.0 <= result.fit.structural <= 1.0
    assert 0.0 <= result.fit.functional <= 1.0
    assert 0.0 <= result.fit.organizational <= 1.0
    assert result.permissions
    assert result.obligations


def test_temm_result_json_roundtrip(tmp_path):
    result = analyze_rollouts(
        [_rollout(True)], GROUP_MAP, _config(success_quantile=0.0)
    )
    path = tmp_path / "temm_result.json"

    result.to_json(path)
    loaded = TEMMResult.from_json(path)

    assert loaded.to_dict() == result.to_dict()


def test_temm_cli_smoke_writes_result_and_summary(monkeypatch, tmp_path):
    out = tmp_path / "temm_result.json"

    class FakeExperiment:
        group_map = GROUP_MAP

    import benchmarl.temm_analyze as cli

    monkeypatch.setattr(
        cli, "reload_experiment_for_temm", lambda checkpoint: FakeExperiment()
    )
    monkeypatch.setattr(
        cli,
        "collect_evaluation_rollouts",
        lambda experiment, config: [_rollout(True), _rollout(True)],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "temm_analyze.py",
            "checkpoint.pt",
            "--episodes",
            "2",
            "--out",
            str(out),
        ],
    )

    temm_main()

    assert out.exists()
    assert (tmp_path / "temm_summary.txt").exists()
    data = json.loads(out.read_text())
    assert "fit" in data
    assert data["metadata"]["episodes"] == 2


def test_temm_reload_falls_back_to_config_pickle(monkeypatch, tmp_path):
    checkpoint = tmp_path / "run" / "checkpoints" / "checkpoint_10.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("")
    (tmp_path / "run" / "config.pkl").write_text("")

    class FakeExperiment:
        pass

    import benchmarl.temm_analyze as cli

    monkeypatch.setattr(
        cli,
        "reload_experiment_from_file",
        lambda checkpoint: (_ for _ in ()).throw(RuntimeError("bad hydra")),
    )
    monkeypatch.setattr(
        cli.Experiment,
        "reload_from_file",
        staticmethod(lambda checkpoint: FakeExperiment()),
    )

    assert isinstance(cli.reload_experiment_for_temm(str(checkpoint)), FakeExperiment)
