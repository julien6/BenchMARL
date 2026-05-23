#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class FitScores:
    structural: float
    functional: float
    organizational: float


@dataclass
class InferredRole:
    id: str
    assigned_agents: List[str]
    support: int
    variance: float
    representativeness: float
    medoid: str
    representative_pattern: List[str]


@dataclass
class InferredGoal:
    id: str
    support: int
    variance: float
    representativeness: float
    medoid_episode: int
    medoid_time: int
    centroid: List[float]
    representative_plan: List[str]


@dataclass
class InferredMission:
    id: str
    goals: List[str]
    support: int


@dataclass
class InferredNorm:
    kind: str
    role: str
    mission: str
    temporal_constraint: str
    support: float
    exclusivity: Optional[float] = None


@dataclass
class TEMMResult:
    fit: FitScores
    mean_return: float
    reward_std: float
    roles: List[InferredRole] = field(default_factory=list)
    goals: List[InferredGoal] = field(default_factory=list)
    missions: List[InferredMission] = field(default_factory=list)
    permissions: List[InferredNorm] = field(default_factory=list)
    obligations: List[InferredNorm] = field(default_factory=list)
    selected_parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TEMMResult":
        return cls(
            fit=FitScores(**data["fit"]),
            mean_return=data["mean_return"],
            reward_std=data["reward_std"],
            roles=[InferredRole(**item) for item in data.get("roles", [])],
            goals=[InferredGoal(**item) for item in data.get("goals", [])],
            missions=[InferredMission(**item) for item in data.get("missions", [])],
            permissions=[
                InferredNorm(**item) for item in data.get("permissions", [])
            ],
            obligations=[
                InferredNorm(**item) for item in data.get("obligations", [])
            ],
            selected_parameters=data.get("selected_parameters", {}),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "TEMMResult":
        with open(path) as f:
            return cls.from_dict(json.load(f))


@dataclass
class TEMMDiagnostics:
    role_embeddings: List[List[float]] = field(default_factory=list)
    role_labels: List[str] = field(default_factory=list)
    role_trajectory_ids: List[str] = field(default_factory=list)
    role_agent_names: List[str] = field(default_factory=list)
    role_episode_indices: List[int] = field(default_factory=list)
    role_action_histograms: Dict[str, List[float]] = field(default_factory=dict)
    action_bin_labels: List[str] = field(default_factory=list)
    goal_embeddings: List[List[float]] = field(default_factory=list)
    goal_labels: List[str] = field(default_factory=list)
    goal_episode_indices: List[int] = field(default_factory=list)
    goal_time_indices: List[int] = field(default_factory=list)
    role_mission_matrix: List[List[float]] = field(default_factory=list)
    role_mission_roles: List[str] = field(default_factory=list)
    role_mission_missions: List[str] = field(default_factory=list)
