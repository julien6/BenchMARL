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
) -> bool:
    """Upload TEMM files and panels to a W&B run Charts section."""

    try:
        import wandb
    except ImportError:
        return False

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

        active_run.log(
            {
                f"{WANDB_SECTION}/organizational_model": _model_table(
                    wandb, organizational_model_id
                ),
                f"{WANDB_SECTION}/roles": _roles_table(
                    wandb, organizational_model
                ),
                f"{WANDB_SECTION}/goals": _goals_table(
                    wandb, organizational_model
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
        )

        artifact_name = _sanitize_artifact_name(f"temm-{run_id}")
        artifact = wandb.Artifact(artifact_name, type="temm")
        artifact.add_file(str(result_path), name=result_path.name)
        artifact.add_file(str(summary_path), name=summary_path.name)
        active_run.log_artifact(artifact)
        return True
    finally:
        if opened_run:
            wandb.finish()


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


def _pre_html(text: str) -> str:
    return (
        "<pre style='white-space: pre-wrap; font-size: 12px; "
        "line-height: 1.35; max-height: 700px; overflow: auto;'>"
        f"{escape(text)}"
        "</pre>"
    )


def _sanitize_artifact_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-") or "temm"
