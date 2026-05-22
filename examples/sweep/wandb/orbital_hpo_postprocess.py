#!/usr/bin/env python3
"""Post-sweep utilities for ORBITAL MAPPO HPO."""

from __future__ import annotations

import argparse
import json
import shlex
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_SWEEP_OUTPUTS = SCRIPT_DIR / "outputs" / "2026-05-22"
DEFAULT_CANDIDATES = SCRIPT_DIR / "orbital_finalists.yaml"
SWEEP_FILTER = {
    "max_n_frames": 1_500_000,
    "on_policy_collected_frames_per_batch": 8208,
    "evaluation_interval": 41040,
}
DEFAULT_SEEDS = range(5)
STRESS_SEEDS = range(3)
SCENARIOS = {
    "default": (),
    "network_stress": ("task.p_link_drop=0.15",),
    "cyber_stress": ("task.adversarial_rate=0.10",),
    "resource_stress": ("task.energy_budget=32.0",),
}


@dataclass
class RunScore:
    run_dir: Path
    config_path: Path
    json_path: Path
    config: dict[str, Any]
    step_means: list[tuple[int, float]]
    absolute_return: float | None

    def tail_mean(self, tail_evals: int) -> float:
        return statistics.mean(value for _, value in self.step_means[-tail_evals:])

    @property
    def last_mean(self) -> float:
        return self.step_means[-1][1]

    @property
    def last_step(self) -> int:
        return self.step_means[-1][0]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as file:
        data = yaml.safe_load(file)
    return data or {}


def benchmarl_json(run_dir: Path) -> Path | None:
    # This direct child pattern avoids W&B copies even though this example
    # directory itself is named `wandb`.
    jsons = list(run_dir.glob("mappo_orbital_mlp__*/mappo_orbital_mlp__*.json"))
    return jsons[0] if jsons else None


def load_run_score(config_path: Path) -> RunScore | None:
    run_dir = config_path.parent.parent
    json_path = benchmarl_json(run_dir)
    if json_path is None:
        return None

    config = load_yaml(config_path)
    with json_path.open() as file:
        json_data = json.load(file)
    try:
        run_data = json_data["pettingzoo"]["orbital"]["mappo"][f"seed_{config['seed']}"]
    except KeyError:
        return None

    step_means = []
    for key, value in run_data.items():
        if key.startswith("step_") and value.get("return"):
            step_means.append((value["step_count"], statistics.mean(value["return"])))
    step_means.sort()
    if not step_means:
        return None

    absolute_values = run_data.get("absolute_metrics", {}).get("return", [])
    absolute_return = statistics.mean(absolute_values) if absolute_values else None
    return RunScore(run_dir, config_path, json_path, config, step_means, absolute_return)


def scan_runs(outputs: Path) -> list[RunScore]:
    runs = []
    for config_path in sorted(outputs.rglob(".hydra/config.yaml")):
        score = load_run_score(config_path)
        if score is not None:
            runs.append(score)
    return runs


def matches_sweep_filter(score: RunScore) -> bool:
    experiment = score.config.get("experiment", {})
    return all(experiment.get(key) == value for key, value in SWEEP_FILTER.items())


def config_override_values(config: dict[str, Any]) -> list[str]:
    experiment = config["experiment"]
    algorithm = config["algorithm"]
    return [
        f"experiment.lr={experiment['lr']}",
        f"experiment.gamma={experiment['gamma']}",
        (
            "experiment.on_policy_n_minibatch_iters="
            f"{experiment['on_policy_n_minibatch_iters']}"
        ),
        f"algorithm.entropy_coef={algorithm['entropy_coef']}",
        f"algorithm.clip_epsilon={algorithm['clip_epsilon']}",
        f"algorithm.lmbda={algorithm['lmbda']}",
    ]


def command_shortlist(args: argparse.Namespace) -> int:
    scores = [score for score in scan_runs(args.outputs) if matches_sweep_filter(score)]
    scores.sort(key=lambda score: score.tail_mean(args.tail_evals), reverse=True)
    print(
        "ORBITAL sweep runs matching "
        f"{SWEEP_FILTER}: {len(scores)} found under {args.outputs}"
    )
    print(f"Ranking metric: mean of final {args.tail_evals} evaluation means\n")
    print("| Rank | Run | Evals | Last frames | Tail mean | Last eval | Absolute | Overrides |")
    print("| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for rank, score in enumerate(scores[: args.top_k], 1):
        absolute = "n/a" if score.absolute_return is None else f"{score.absolute_return:.1f}"
        overrides = "<br>".join(config_override_values(score.config))
        print(
            f"| {rank} | `{score.run_dir.name}` | {len(score.step_means)} | "
            f"{score.last_step} | {score.tail_mean(args.tail_evals):.1f} | "
            f"{score.last_mean:.1f} | {absolute} | `{overrides}` |"
        )
    return 0


def candidate_sections(path: Path, include_fallbacks: bool) -> dict[str, dict[str, Any]]:
    data = load_yaml(path)
    candidates = dict(data.get("finalists", {}))
    if include_fallbacks:
        candidates.update(data.get("fallbacks", {}))
    if not candidates:
        raise ValueError(f"No candidates found in {path}")
    return candidates


def candidate_overrides(candidate: dict[str, Any]) -> list[str]:
    experiment = candidate["experiment"]
    algorithm = candidate["algorithm"]
    return [
        f"experiment.lr={experiment['lr']}",
        f"experiment.gamma={experiment['gamma']}",
        (
            "experiment.on_policy_n_minibatch_iters="
            f"{experiment['on_policy_n_minibatch_iters']}"
        ),
        f"algorithm.entropy_coef={algorithm['entropy_coef']}",
        f"algorithm.clip_epsilon={algorithm['clip_epsilon']}",
        f"algorithm.lmbda={algorithm['lmbda']}",
    ]


def validation_specs(
    candidates: dict[str, dict[str, Any]], scenarios: Iterable[str]
) -> Iterable[tuple[str, dict[str, Any], str, int]]:
    for candidate_id, candidate in candidates.items():
        for scenario in scenarios:
            seeds = DEFAULT_SEEDS if scenario == "default" else STRESS_SEEDS
            for seed in seeds:
                yield candidate_id, candidate, scenario, seed


def validation_command(
    python_bin: str, candidate_id: str, candidate: dict[str, Any], scenario: str, seed: int
) -> list[str]:
    run_name = f"orbital-hpo-{candidate_id}-{scenario}-seed{seed}"
    tags = (
        "[orbital_hpo_validation,"
        f"candidate_{candidate_id},scenario_{scenario},seed_{seed}]"
    )
    return [
        python_bin,
        "fine_tuned/pettingzoo_orbital/pettingzoo_orbital_run.py",
        f"seed={seed}",
        "++experiment.organizational_model=orbital_none",
        "experiment.max_n_frames=3000000",
        "experiment.evaluation_episodes=32",
        *candidate_overrides(candidate),
        *SCENARIOS[scenario],
        f"+experiment.wandb_extra_kwargs.tags={tags}",
        f"+experiment.wandb_extra_kwargs.name={run_name}",
    ]


def chosen_scenarios(mode: str) -> tuple[str, ...]:
    if mode == "default":
        return ("default",)
    if mode == "stress":
        return tuple(scenario for scenario in SCENARIOS if scenario != "default")
    return tuple(SCENARIOS)


def command_validation_commands(args: argparse.Namespace) -> int:
    candidates = candidate_sections(args.candidates, args.include_fallbacks)
    specs = list(validation_specs(candidates, chosen_scenarios(args.mode)))
    print(f"# {len(specs)} ORBITAL validation commands")
    for candidate_id, candidate, scenario, seed in specs:
        command = validation_command(args.python, candidate_id, candidate, scenario, seed)
        print(shlex.join(command))
        if args.run:
            subprocess.run(command, cwd=REPO_ROOT, check=True)
    return 0


def validation_tags(score: RunScore) -> set[str]:
    wandb_kwargs = score.config.get("experiment", {}).get("wandb_extra_kwargs", {})
    return set(wandb_kwargs.get("tags", []))


def tag_value(tags: set[str], prefix: str) -> str | None:
    values = sorted(tag.removeprefix(prefix) for tag in tags if tag.startswith(prefix))
    return values[0] if values else None


def rank(values: dict[str, float]) -> dict[str, int]:
    sorted_values = sorted(values.items(), key=lambda item: item[1], reverse=True)
    return {candidate: index for index, (candidate, _) in enumerate(sorted_values, 1)}


def command_validation_summary(args: argparse.Namespace) -> int:
    records: dict[str, dict[str, list[float]]] = {}
    for score in scan_runs(args.outputs):
        tags = validation_tags(score)
        if "orbital_hpo_validation" not in tags:
            continue
        candidate = tag_value(tags, "candidate_")
        scenario = tag_value(tags, "scenario_")
        if candidate is None or scenario is None:
            continue
        records.setdefault(scenario, {}).setdefault(candidate, []).append(
            score.tail_mean(args.tail_evals)
        )

    if not records:
        print(f"No tagged ORBITAL validation runs found under {args.outputs}", file=sys.stderr)
        return 1

    print(f"Validation tail metric: mean of final {args.tail_evals} evaluation means\n")
    print("| Scenario | Candidate | Runs | Mean | Std |")
    print("| --- | --- | ---: | ---: | ---: |")
    means: dict[str, dict[str, float]] = {}
    for scenario in SCENARIOS:
        for candidate, values in sorted(records.get(scenario, {}).items()):
            mean = statistics.mean(values)
            std = statistics.pstdev(values)
            means.setdefault(scenario, {})[candidate] = mean
            print(f"| `{scenario}` | `{candidate}` | {len(values)} | {mean:.1f} | {std:.1f} |")

    weighted_ranks: dict[str, float] = {}
    weights = {"default": 2, "network_stress": 1, "cyber_stress": 1, "resource_stress": 1}
    for scenario, scenario_means in means.items():
        for candidate, scenario_rank in rank(scenario_means).items():
            weighted_ranks[candidate] = weighted_ranks.get(candidate, 0.0) + (
                weights.get(scenario, 1) * scenario_rank
            )
    print("\nWeighted rank totals; lower is better:")
    for candidate, total in sorted(weighted_ranks.items(), key=lambda item: item[1]):
        print(f"- {candidate}: {total:.1f}")
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    subparsers = root.add_subparsers(dest="command", required=True)

    shortlist = subparsers.add_parser("shortlist", help="Rank completed sweep JSON files.")
    shortlist.add_argument("--outputs", type=Path, default=DEFAULT_SWEEP_OUTPUTS)
    shortlist.add_argument("--tail-evals", type=int, default=5)
    shortlist.add_argument("--top-k", type=int, default=18)
    shortlist.set_defaults(func=command_shortlist)

    commands = subparsers.add_parser(
        "validation-commands", help="Print or run default/stress validation commands."
    )
    commands.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    commands.add_argument("--mode", choices=("default", "stress", "all"), default="all")
    commands.add_argument("--include-fallbacks", action="store_true")
    commands.add_argument("--python", default="python")
    commands.add_argument("--run", action="store_true")
    commands.set_defaults(func=command_validation_commands)

    summary = subparsers.add_parser(
        "validation-summary", help="Summarize tagged validation JSON files."
    )
    summary.add_argument("--outputs", type=Path, required=True)
    summary.add_argument("--tail-evals", type=int, default=5)
    summary.set_defaults(func=command_validation_summary)
    return root


def main() -> int:
    args = parser().parse_args()
    try:
        return args.func(args)
    except (KeyError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
