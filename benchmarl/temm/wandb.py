#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

import json
import re
from html import escape
from pathlib import Path
from typing import Any, Mapping, Optional

from .types import TEMMResult
from .visualization import figure_panel_html

WANDB_SECTION = "TEMM & MOISE+MARL"


def publish_temm_to_wandb(
    result: TEMMResult,
    result_path: Path,
    summary_path: Path,
    project: str,
    entity: Optional[str],
    run_id: str,
    run_name: str,
    run_dir: Path,
    organizational_model_id: Optional[str] = None,
    organizational_model: Optional[Any] = None,
    figures: Optional[Mapping[str, Any]] = None,
    explanations: Optional[Mapping[str, str]] = None,
    figure_paths: Optional[Mapping[str, Path]] = None,
    diagnostics: Optional[Any] = None,
) -> bool:
    """Upload TEMM files and panels to a W&B run Charts section."""

    try:
        import wandb
    except ImportError:
        return False

    run_id = _resolve_local_wandb_run_id(run_dir, run_id)
    active_run = wandb.run
    opened_run = False
    if active_run is None:
        active_run = wandb.init(
            project=project,
            entity=entity,
            id=run_id,
            name=run_name,
            resume="allow",
            dir=str(run_dir),
            job_type="temm",
        )
        opened_run = True

    try:
        active_run.summary["temm/organizational_fit"] = result.fit.organizational
        active_run.summary["temm/structural_fit"] = result.fit.structural
        active_run.summary["temm/functional_fit"] = result.fit.functional
        active_run.summary["temm/roles"] = len(result.roles)
        active_run.summary["temm/goals"] = len(result.goals)
        active_run.summary["temm/missions"] = len(result.missions)

        log_payload = {
            f"{WANDB_SECTION}/fit_scores": _fit_table(wandb, result),
            f"{WANDB_SECTION}/organizational_model": _model_table(
                wandb, organizational_model_id
            ),
            f"{WANDB_SECTION}/roles": _roles_table(wandb, organizational_model),
            f"{WANDB_SECTION}/goals": _goals_table(wandb, organizational_model),
            f"{WANDB_SECTION}/TEMM_roles": _temm_roles_table(wandb, result),
            f"{WANDB_SECTION}/TEMM_goals": _temm_goals_table(wandb, result),
            f"{WANDB_SECTION}/TEMM_missions": _temm_missions_table(wandb, result),
            f"{WANDB_SECTION}/TEMM_permissions": _temm_norms_table(
                wandb, result.permissions
            ),
            f"{WANDB_SECTION}/TEMM_obligations": _temm_norms_table(
                wandb, result.obligations
            ),
            f"{WANDB_SECTION}/semantic_action_map": _semantic_action_map_table(
                wandb, diagnostics
            ),
            f"{WANDB_SECTION}/TEMM_summary": wandb.Html(
                _pre_html(summary_path.read_text())
            ),
            f"{WANDB_SECTION}/TEMM_json": wandb.Html(
                _pre_html(json.dumps(result.to_dict(), indent=2))
            ),
            f"{WANDB_SECTION}/organizational_fit": result.fit.organizational,
            f"{WANDB_SECTION}/structural_fit": result.fit.structural,
            f"{WANDB_SECTION}/functional_fit": result.fit.functional,
        }
        if figures:
            for name, figure in figures.items():
                explanation = (explanations or {}).get(name, "")
                log_payload[f"{WANDB_SECTION}/{name}"] = wandb.Html(
                    figure_panel_html(figure, explanation)
                )
        active_run.log(log_payload)

        artifact_name = _sanitize_artifact_name(f"temm-{run_id}")
        artifact = wandb.Artifact(artifact_name, type="temm")
        artifact.add_file(str(result_path), name=result_path.name)
        artifact.add_file(str(summary_path), name=summary_path.name)
        for path in (figure_paths or {}).values():
            artifact.add_file(str(path), name=f"figures/{path.name}")
        active_run.log_artifact(artifact)
        return True
    finally:
        if opened_run:
            wandb.finish()


def _resolve_local_wandb_run_id(run_dir: Path, fallback_run_id: str) -> str:
    """Resolve the original W&B id from a BenchMARL run folder.

    In sweeps, W&B may assign a short run id that differs from the display name.
    Resuming by display name creates a second run with the same name, so TEMM
    prefers the local training run id recorded under ``wandb/run-*``.
    """

    wandb_dir = Path(run_dir) / "wandb"
    if not wandb_dir.exists():
        return fallback_run_id

    candidates = []
    for path in wandb_dir.iterdir():
        if not path.is_dir():
            continue
        match = re.match(r"(?:offline-)?run-(\d{8}_\d{6})-(.+)$", path.name)
        if match is None:
            continue

        timestamp, local_run_id = match.groups()
        metadata = _read_json(path / "files" / "wandb-metadata.json")
        config_path = path / "files" / "config.yaml"
        program = " ".join(
            str(metadata.get(key, "")) for key in ("program", "codePath")
        )
        args = metadata.get("args") or []
        args_text = "\n".join(str(arg) for arg in args)
        config_text = ""
        if config_path.exists():
            try:
                config_text = config_path.read_text(errors="ignore")
            except OSError:
                config_text = ""

        is_temm_run = "temm_analyze.py" in program
        looks_like_training = (
            "benchmarl/run.py" in program
            or "task=" in args_text
            or "algorithm_name:" in config_text
            or "experiment_config:" in config_text
        )
        score = 0
        if not is_temm_run:
            score += 100
        if looks_like_training:
            score += 50
        if local_run_id == fallback_run_id:
            score += 10
        candidates.append((score, timestamp, local_run_id))

    if not candidates:
        return fallback_run_id

    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][2]


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _fit_table(wandb, result: TEMMResult):
    return wandb.Table(
        columns=["metric", "value"],
        data=[
            ["structural_fit", result.fit.structural],
            ["functional_fit", result.fit.functional],
            ["organizational_fit", result.fit.organizational],
            ["mean_return", result.mean_return],
            ["reward_std", result.reward_std],
        ],
    )


def _model_table(wandb, model_id: Optional[str]):
    return wandb.Table(
        columns=["field", "value"],
        data=[
            ["organizational_model", model_id or "None"],
            ["description", "BenchMARL MOISE+MARL model selected for this run."],
        ],
    )


def _roles_table(wandb, organizational_model: Optional[Any]):
    rows = []
    if organizational_model is not None:
        role_agents = _assignment_index(
            getattr(organizational_model, "role_assignments", {})
        )
        for role_name, role in getattr(organizational_model, "roles", {}).items():
            rows.append(
                [
                    role_name,
                    _one_line_description(role),
                    ", ".join(role_agents.get(role_name, [])) or "-",
                ]
            )
    if not rows:
        rows = [["-", "-", "-"]]
    return wandb.Table(columns=["role", "description", "assigned_agents"], data=rows)


def _goals_table(wandb, organizational_model: Optional[Any]):
    rows = []
    if organizational_model is not None:
        goal_agents = {}
        for agent_name, goals in getattr(
            organizational_model, "goal_assignments", {}
        ).items():
            for goal_name in goals:
                goal_agents.setdefault(goal_name, []).append(agent_name)
        for goal_name, goal in getattr(organizational_model, "goals", {}).items():
            rows.append(
                [
                    goal_name,
                    _one_line_description(goal),
                    ", ".join(sorted(goal_agents.get(goal_name, []))) or "-",
                ]
            )
    if not rows:
        rows = [["-", "-", "-"]]
    return wandb.Table(columns=["goal", "description", "assigned_agents"], data=rows)


def _temm_roles_table(wandb, result: TEMMResult):
    rows = [
        [
            role.id,
            ", ".join(role.assigned_agents) or "-",
            role.support,
            role.variance,
            role.representativeness,
            role.medoid,
            _list_cell(role.representative_pattern),
            _semantic_cell(role.representative_pattern_semantic),
        ]
        for role in result.roles
    ]
    if not rows:
        rows = [["-", "-", 0, 0.0, 0.0, "-", "-", "-"]]
    return wandb.Table(
        columns=[
            "role",
            "assigned_agents",
            "support",
            "variance",
            "representativeness",
            "medoid",
            "representative_pattern",
            "representative_pattern_semantic",
        ],
        data=rows,
    )


def _temm_goals_table(wandb, result: TEMMResult):
    rows = [
        [
            goal.id,
            goal.support,
            goal.variance,
            goal.representativeness,
            goal.medoid_episode,
            goal.medoid_time,
            _list_cell(goal.representative_plan),
            _semantic_cell(goal.representative_plan_semantic),
            _list_cell([f"{value:.4f}" for value in goal.centroid[:12]]),
        ]
        for goal in result.goals
    ]
    if not rows:
        rows = [["-", 0, 0.0, 0.0, -1, -1, "-", "-", "-"]]
    return wandb.Table(
        columns=[
            "goal",
            "support",
            "variance",
            "representativeness",
            "medoid_episode",
            "medoid_time",
            "representative_plan",
            "representative_plan_semantic",
            "centroid_preview",
        ],
        data=rows,
    )


def _temm_missions_table(wandb, result: TEMMResult):
    rows = [
        [mission.id, _list_cell(mission.goals), mission.support]
        for mission in result.missions
    ]
    if not rows:
        rows = [["-", "-", 0]]
    return wandb.Table(columns=["mission", "goals", "support"], data=rows)


def _temm_norms_table(wandb, norms):
    rows = [
        [
            norm.kind,
            norm.role,
            norm.mission,
            norm.temporal_constraint,
            norm.support,
            "-" if norm.exclusivity is None else norm.exclusivity,
        ]
        for norm in norms
    ]
    if not rows:
        rows = [["-", "-", "-", "-", 0.0, "-"]]
    return wandb.Table(
        columns=[
            "kind",
            "role",
            "mission",
            "temporal_constraint",
            "support",
            "exclusivity",
        ],
        data=rows,
    )


def _semantic_action_map_table(wandb, diagnostics):
    rows = []
    if diagnostics is not None:
        for item in getattr(diagnostics, "semantic_action_map", []):
            rows.append(
                [
                    item.get("action_id", "-"),
                    item.get("action_label", "-"),
                    item.get("description", "-"),
                ]
            )
    if not rows:
        rows = [["-", "-", "-"]]
    return wandb.Table(
        columns=["action_id", "action_label", "description"], data=rows
    )


def _assignment_index(assignments: Mapping[str, str]) -> dict[str, list[str]]:
    indexed = {}
    for agent_name, item_name in assignments.items():
        indexed.setdefault(item_name, []).append(agent_name)
    return {item_name: sorted(agents) for item_name, agents in indexed.items()}


def _one_line_description(item: Any) -> str:
    description = getattr(item, "description", "")
    if description:
        return " ".join(description.split())
    docstring = getattr(type(item), "__doc__", None)
    if docstring:
        return " ".join(docstring.strip().split())
    return "-"


def _list_cell(values) -> str:
    values = list(values)
    if not values:
        return "-"
    return "\n".join(str(value) for value in values)


def _semantic_cell(values) -> str:
    rows = []
    for item in values or []:
        token = item.get("token", "-")
        action = item.get("action_label", "-")
        interpretation = item.get("interpretation", "-")
        tags = ", ".join(str(value) for value in item.get("state_tags", [])) or "-"
        rows.append(f"{token} | {action} | {tags} | {interpretation}")
    return "\n".join(rows) if rows else "-"


def _pre_html(text: str) -> str:
    return (
        "<pre style='white-space: pre-wrap; font-size: 12px; "
        "line-height: 1.35; max-height: 700px; overflow: auto;'>"
        f"{escape(text)}"
        "</pre>"
    )


def _sanitize_artifact_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-") or "temm"
