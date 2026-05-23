#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

import json
import sys

import torch
from tensordict import TensorDict

import benchmarl.temm.wandb as temm_wandb
from benchmarl.temm import (
    TEMMConfig,
    TEMMVisualizer,
    TEMMResult,
    analyze_rollouts_with_diagnostics,
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
        "run_temm_for_experiment",
        lambda experiment, config, output_path, publish_wandb, create_wandb_section: (
            analyze_rollouts(
                [_rollout(True), _rollout(True)], GROUP_MAP, config
            ).to_json(output_path),
            output_path.with_name("temm_summary.txt").write_text("summary\n"),
        ),
    )
    monkeypatch.setattr(cli, "close_experiment_for_temm", lambda experiment: None)
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
        staticmethod(lambda checkpoint, experiment_patch=None: FakeExperiment()),
    )

    assert isinstance(cli.reload_experiment_for_temm(str(checkpoint)), FakeExperiment)


def test_wandb_tables_process_temm_result_for_chart_section():
    result = analyze_rollouts(
        [_rollout(True), _rollout(True)], GROUP_MAP, _config(success_quantile=0.0)
    )

    class FakeTable:
        def __init__(self, columns, data):
            self.columns = columns
            self.data = data

    class FakeWandb:
        Table = FakeTable

    fit_table = temm_wandb._fit_table(FakeWandb, result)
    roles_table = temm_wandb._temm_roles_table(FakeWandb, result)
    goals_table = temm_wandb._temm_goals_table(FakeWandb, result)
    missions_table = temm_wandb._temm_missions_table(FakeWandb, result)
    permissions_table = temm_wandb._temm_norms_table(FakeWandb, result.permissions)

    assert fit_table.columns == ["metric", "value"]
    assert ["organizational_fit", result.fit.organizational] in fit_table.data
    assert "representative_pattern" in roles_table.columns
    assert "representative_plan" in goals_table.columns
    assert missions_table.columns == ["mission", "goals", "support"]
    assert permissions_table.columns == [
        "kind",
        "role",
        "mission",
        "temporal_constraint",
        "support",
        "exclusivity",
    ]


def test_temm_diagnostics_and_visualizations_export_html(tmp_path):
    result, diagnostics = analyze_rollouts_with_diagnostics(
        [_rollout(True), _rollout(True)], GROUP_MAP, _config(success_quantile=0.0)
    )

    assert len(diagnostics.role_embeddings) == len(diagnostics.role_labels)
    assert diagnostics.role_action_histograms
    assert len(diagnostics.goal_embeddings) == len(diagnostics.goal_labels)
    assert diagnostics.trajectory_timelines
    assert diagnostics.role_prototypes
    assert diagnostics.goal_prototypes
    assert diagnostics.role_distance_matrix
    assert diagnostics.goal_distance_matrix
    assert diagnostics.role_members

    visualizer = TEMMVisualizer(result, diagnostics)
    figures = visualizer.build_figures()
    assert {
        "figure_fit_summary",
        "figure_role_projection_pca",
        "figure_goal_projection_pca",
        "figure_role_behavior_heatmap",
        "figure_role_mission_matrix",
        "figure_mission_graph",
        "figure_role_prototype_timelines",
        "figure_goal_prototype_timelines",
        "figure_symbolic_rule_table",
        "figure_agent_role_episode_map",
        "figure_role_hierarchy_tree",
        "figure_goal_mission_hierarchy_tree",
        "figure_role_distance_heatmap",
        "figure_goal_distance_heatmap",
    } <= set(figures)
    for figure in figures.values():
        assert figure.data

    paths = visualizer.write_html(tmp_path / "temm_figures")
    assert set(paths) == set(figures)
    assert all(path.exists() for path in paths.values())


def test_wandb_payload_includes_plotly_figures(monkeypatch, tmp_path):
    result, diagnostics = analyze_rollouts_with_diagnostics(
        [_rollout(True), _rollout(True)], GROUP_MAP, _config(success_quantile=0.0)
    )
    visualizer = TEMMVisualizer(result, diagnostics)
    figures = visualizer.build_figures()
    figure_paths = visualizer.write_html(tmp_path / "figures")
    result_path = tmp_path / "temm_result.json"
    summary_path = tmp_path / "temm_summary.txt"
    result.to_json(result_path)
    summary_path.write_text("summary\n")
    logged = {}
    artifacts = []

    class FakeRun:
        def __init__(self):
            self.summary = {}

        def log(self, payload):
            logged.update(payload)

        def log_artifact(self, artifact):
            artifacts.append(artifact)

    class FakeArtifact:
        def __init__(self, name, type):
            self.name = name
            self.type = type
            self.files = []

        def add_file(self, path, name=None):
            self.files.append((path, name))

    class FakeWandb:
        run = FakeRun()
        Artifact = FakeArtifact

        class Table:
            def __init__(self, columns, data):
                self.columns = columns
                self.data = data

        class Html:
            def __init__(self, html):
                self.html = html

        class Plotly:
            def __init__(self, figure):
                self.figure = figure

    monkeypatch.setitem(sys.modules, "wandb", FakeWandb)

    assert temm_wandb.publish_temm_to_wandb(
        result=result,
        result_path=result_path,
        summary_path=summary_path,
        project="benchmarl",
        entity=None,
        run_id="run",
        run_name="run",
        run_dir=tmp_path,
        figures=figures,
        figure_paths=figure_paths,
    )
    assert "TEMM & MOISE+MARL/figure_fit_summary" in logged
    assert "TEMM & MOISE+MARL/figure_mission_graph" in logged
    assert "TEMM & MOISE+MARL/figure_role_prototype_timelines" in logged
    assert "TEMM & MOISE+MARL/figure_role_hierarchy_tree" in logged
    assert artifacts
    assert any(name and name.startswith("figures/") for _, name in artifacts[0].files)


def test_wandb_run_id_resolution_prefers_original_training_run(tmp_path):
    wandb_dir = tmp_path / "wandb"
    training_run = wandb_dir / "run-20260523_010000-original123"
    temm_run = wandb_dir / "run-20260523_020000-display_name"
    training_files = training_run / "files"
    temm_files = temm_run / "files"
    training_files.mkdir(parents=True)
    temm_files.mkdir(parents=True)
    (training_files / "wandb-metadata.json").write_text(
        json.dumps(
            {
                "program": "/repo/benchmarl/run.py",
                "codePath": "benchmarl/run.py",
                "args": ["task=pettingzoo/orbital", "algorithm=mappo"],
            }
        )
    )
    (temm_files / "wandb-metadata.json").write_text(
        json.dumps(
            {
                "program": "/repo/benchmarl/temm_analyze.py",
                "codePath": "benchmarl/temm_analyze.py",
            }
        )
    )

    assert (
        temm_wandb._resolve_local_wandb_run_id(tmp_path, "display_name")
        == "original123"
    )


def test_temm_visualizations_handle_empty_result(tmp_path):
    from benchmarl.temm import TEMMDiagnostics
    from benchmarl.temm.types import FitScores

    result = TEMMResult(
        fit=FitScores(structural=0.0, functional=0.0, organizational=0.0),
        mean_return=0.0,
        reward_std=0.0,
    )
    visualizer = TEMMVisualizer(result, TEMMDiagnostics())
    figures = visualizer.build_figures()

    assert figures
    assert all(figure.data for figure in figures.values())
    paths = visualizer.write_html(tmp_path / "empty_figures")
    assert paths
