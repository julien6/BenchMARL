#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from .types import TEMMResult


def publish_temm_to_wandb(
    result: TEMMResult,
    result_path: Path,
    summary_path: Path,
    project: str,
    entity: Optional[str],
    run_id: str,
    run_name: str,
    run_dir: Path,
    create_report: bool = True,
) -> bool:
    """Upload TEMM files to W&B and create a report when report APIs are present."""

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

        artifact_name = _sanitize_artifact_name(f"temm-{run_id}")
        artifact = wandb.Artifact(artifact_name, type="temm")
        artifact.add_file(str(result_path), name=result_path.name)
        artifact.add_file(str(summary_path), name=summary_path.name)
        active_run.log_artifact(artifact)

        if create_report:
            _try_create_report(
                project=project,
                entity=entity,
                title="TEMM",
                result_path=result_path,
                summary_path=summary_path,
            )
        return True
    finally:
        if opened_run:
            wandb.finish()


def _try_create_report(
    project: str,
    entity: Optional[str],
    title: str,
    result_path: Path,
    summary_path: Path,
) -> None:
    try:
        import wandb.apis.reports as wr
    except Exception:
        return

    report_cls = getattr(wr, "Report", None)
    if report_cls is None:
        return

    markdown = _report_markdown(summary_path, result_path)
    blocks = _markdown_blocks(wr, markdown)
    if not blocks:
        return

    try:
        kwargs = {"project": project, "title": title}
        if entity is not None:
            kwargs["entity"] = entity
        report = report_cls(**kwargs)
        report.blocks = blocks
        report.save()
    except Exception:
        return


def _markdown_blocks(wr, markdown: str):
    markdown_cls = getattr(wr, "MarkdownBlock", None)
    if markdown_cls is not None:
        try:
            return [markdown_cls(text=markdown)]
        except TypeError:
            return [markdown_cls(markdown)]

    paragraph_cls = getattr(wr, "P", None)
    if paragraph_cls is not None:
        try:
            return [paragraph_cls(markdown)]
        except TypeError:
            return [paragraph_cls(text=markdown)]
    return []


def _report_markdown(summary_path: Path, result_path: Path) -> str:
    summary = summary_path.read_text()
    result = json.loads(result_path.read_text())
    result_json = json.dumps(result, indent=2)
    return "\n".join(
        [
            "# TEMM report",
            "",
            "## Summary",
            "",
            "```text",
            summary.strip(),
            "```",
            "",
            "## JSON",
            "",
            "```json",
            result_json,
            "```",
        ]
    )


def _sanitize_artifact_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-") or "temm"
