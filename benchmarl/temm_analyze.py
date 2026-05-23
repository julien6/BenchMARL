#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarl.experiment import Experiment
from benchmarl.hydra_config import reload_experiment_from_file
from benchmarl.temm import TEMMConfig, run_temm_for_experiment


def reload_experiment_for_temm(checkpoint: str):
    """Reload a checkpoint, falling back to config.pkl if Hydra metadata is stale."""

    try:
        return reload_experiment_from_file(checkpoint)
    except Exception:
        config_file = Path(checkpoint).parent.parent / "config.pkl"
        if not config_file.exists():
            raise
        return Experiment.reload_from_file(
            checkpoint,
            experiment_patch={
                "loggers": [],
                "create_json": False,
                "render": False,
            },
        )


def close_experiment_for_temm(experiment) -> None:
    close = getattr(experiment, "close", None)
    if close is not None:
        close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run post-hoc TEMM analysis on a BenchMARL checkpoint."
    )
    parser.add_argument("checkpoint", type=str, help="BenchMARL checkpoint file")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--out", type=str, default="temm_result.json")
    parser.add_argument("--success-quantile", type=float, default=0.75)
    parser.add_argument(
        "--distance-metric", choices=("euclidean", "cosine"), default="euclidean"
    )
    parser.add_argument("--permission-min-support", type=float, default=0.25)
    parser.add_argument("--obligation-min-support", type=float, default=0.75)
    parser.add_argument("--obligation-exclusivity-threshold", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--semantic-adapter",
        choices=("auto", "none", "orbital"),
        default="auto",
        help="Semantic grounding used for TEMM trajectory tokens.",
    )
    parser.add_argument(
        "--wandb-section",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Upload TEMM and MOISE+MARL panels to the W&B Charts section.",
    )
    parser.add_argument(
        "--wandb-report",
        action=argparse.BooleanOptionalAction,
        dest="wandb_section",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    config = TEMMConfig(
        episodes=args.episodes,
        success_quantile=args.success_quantile,
        distance_metric=args.distance_metric,
        permission_min_support=args.permission_min_support,
        obligation_min_support=args.obligation_min_support,
        obligation_exclusivity_threshold=args.obligation_exclusivity_threshold,
        output_path=args.out,
        seed=args.seed,
        semantic_adapter=args.semantic_adapter,
    )

    experiment = reload_experiment_for_temm(str(Path(args.checkpoint).resolve()))
    try:
        run_temm_for_experiment(
            experiment=experiment,
            config=config,
            output_path=Path(config.output_path),
            publish_wandb=args.wandb_section,
            create_wandb_section=args.wandb_section,
        )
    finally:
        close_experiment_for_temm(experiment)


if __name__ == "__main__":
    main()
