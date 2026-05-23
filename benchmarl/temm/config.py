#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence


@dataclass
class TEMMConfig:
    """Configuration for post-hoc TEMM trajectory analysis."""

    episodes: int = 10
    success_quantile: float = 0.75
    distance_metric: str = "euclidean"
    role_distance_thresholds: Sequence[float] = field(
        default_factory=lambda: (0.20, 0.35, 0.50)
    )
    goal_distance_thresholds: Sequence[float] = field(
        default_factory=lambda: (0.20, 0.35, 0.50)
    )
    representativeness_thresholds: Sequence[float] = field(
        default_factory=lambda: (0.0, 0.25, 0.50)
    )
    max_action_bins: int = 32
    plan_window: int = 3
    permission_min_support: float = 0.25
    obligation_min_support: float = 0.75
    obligation_exclusivity_threshold: float = 0.75
    cluster_penalty: float = 0.03
    output_path: str = "temm_result.json"
    seed: Optional[int] = None
    static_evaluation: Optional[bool] = None

    def __post_init__(self) -> None:
        if self.episodes <= 0:
            raise ValueError("TEMM episodes must be positive.")
        if not 0.0 <= self.success_quantile <= 1.0:
            raise ValueError("success_quantile must be in [0, 1].")
        for name in (
            "permission_min_support",
            "obligation_min_support",
            "obligation_exclusivity_threshold",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")
        if self.distance_metric not in {"euclidean", "cosine"}:
            raise ValueError("distance_metric must be 'euclidean' or 'cosine'.")
