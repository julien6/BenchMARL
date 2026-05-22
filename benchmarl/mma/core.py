#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from torch import Tensor

from benchmarl.environments.common import TaskClass


def _one_line_description(item: Any) -> str:
    description = getattr(item, "description", "")
    if description:
        return " ".join(description.split())

    docstring = getattr(type(item), "__doc__", None)
    if docstring:
        return " ".join(docstring.strip().split())
    return "-"


def _ascii_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    if not rows:
        rows = [tuple("-" for _ in headers)]

    widths = [
        max(len(str(row[index])) for row in (headers, *rows))
        for index in range(len(headers))
    ]
    separator = "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    def render_row(row: Sequence[str]) -> str:
        cells = [
            f" {str(value).ljust(widths[index])} " for index, value in enumerate(row)
        ]
        return "|" + "|".join(cells) + "|"

    lines = [separator, render_row(headers), separator]
    lines.extend(render_row(row) for row in rows)
    lines.append(separator)
    return "\n".join(lines)


@dataclass
class GoalContext:
    """Mutable state owned by one goal assignment for one episode."""

    state: Dict[str, Any] = field(default_factory=dict)


class Role(ABC):
    """A structural MMA rule that constrains a discrete policy action set."""

    @abstractmethod
    def allowed_actions(self, observation: Tensor, agent_name: str) -> Iterable[int]:
        """Return discrete action ids that the agent may sample."""
        raise NotImplementedError


class Goal(ABC):
    """A functional MMA rule that shapes one agent reward."""

    @abstractmethod
    def reward(
        self,
        context: GoalContext,
        observation: Tensor,
        action: Tensor,
        agent_name: str,
    ) -> float | Tensor:
        """Return a bonus or malus for the current agent transition."""
        raise NotImplementedError


@dataclass
class ScriptedRole(Role):
    """Role backed by a Python callable."""

    rule: Callable[[Tensor, str], Iterable[int]]
    description: str = ""

    def allowed_actions(self, observation: Tensor, agent_name: str) -> Iterable[int]:
        return self.rule(observation, agent_name)


@dataclass
class ScriptedGoal(Goal):
    """Goal backed by a Python callable."""

    rule: Callable[[GoalContext, Tensor, Tensor, str], float | Tensor]
    description: str = ""

    def reward(
        self,
        context: GoalContext,
        observation: Tensor,
        action: Tensor,
        agent_name: str,
    ) -> float | Tensor:
        return self.rule(context, observation, action, agent_name)


@dataclass
class OrganizationalModel:
    """BenchMARL-native MMA model for role and goal assignments."""

    roles: Mapping[str, Role] = field(default_factory=dict)
    goals: Mapping[str, Goal] = field(default_factory=dict)
    role_assignments: Mapping[str, str] = field(default_factory=dict)
    goal_assignments: Mapping[str, Sequence[str]] = field(default_factory=dict)

    def validate(self, group_map: Mapping[str, Sequence[str]]) -> None:
        """Validate that assignments target known agents and specifications."""
        agents = {
            agent_name
            for agent_names in group_map.values()
            for agent_name in agent_names
        }
        unknown_agents = (
            set(self.role_assignments).union(self.goal_assignments) - agents
        )
        if unknown_agents:
            unknown = ", ".join(sorted(unknown_agents))
            raise ValueError(f"MMA assignments target unknown agents: {unknown}")

        missing_roles = set(self.role_assignments.values()) - set(self.roles)
        if missing_roles:
            missing = ", ".join(sorted(missing_roles))
            raise ValueError(f"MMA role assignments reference unknown roles: {missing}")

        missing_goals = {
            goal_name
            for goal_names in self.goal_assignments.values()
            for goal_name in goal_names
            if goal_name not in self.goals
        }
        if missing_goals:
            missing = ", ".join(sorted(missing_goals))
            raise ValueError(f"MMA goal assignments reference unknown goals: {missing}")

    def role_for(self, agent_name: str) -> Optional[Role]:
        role_name = self.role_assignments.get(agent_name)
        return None if role_name is None else self.roles[role_name]

    def goals_for(self, agent_name: str) -> List[tuple[str, Goal]]:
        return [
            (goal_name, self.goals[goal_name])
            for goal_name in self.goal_assignments.get(agent_name, ())
        ]

    def ascii_summary(self, model_id: Optional[str] = None) -> str:
        """Render a concise table of MMA specifications and assignments."""
        role_agents: Dict[str, List[str]] = {role_name: [] for role_name in self.roles}
        for agent_name, role_name in self.role_assignments.items():
            role_agents.setdefault(role_name, []).append(agent_name)

        goal_agents: Dict[str, List[str]] = {goal_name: [] for goal_name in self.goals}
        for agent_name, goal_names in self.goal_assignments.items():
            for goal_name in goal_names:
                goal_agents.setdefault(goal_name, []).append(agent_name)

        def assignment_cell(agents: Sequence[str]) -> str:
            return ", ".join(sorted(agents)) or "-"

        role_rows = [
            (
                role_name,
                _one_line_description(role),
                assignment_cell(role_agents[role_name]),
            )
            for role_name, role in self.roles.items()
        ]
        goal_rows = [
            (
                goal_name,
                _one_line_description(goal),
                assignment_cell(goal_agents[goal_name]),
            )
            for goal_name, goal in self.goals.items()
        ]
        title = "MMA organizational model"
        if model_id is not None:
            title += f": {model_id}"
        return "\n".join(
            (
                title,
                "",
                "Roles",
                _ascii_table(("Role", "Description", "Assigned agents"), role_rows),
                "",
                "Goals",
                _ascii_table(("Goal", "Description", "Assigned agents"), goal_rows),
            )
        )


OrganizationalModelFactory = Callable[
    [TaskClass, Mapping[str, Sequence[str]]], OrganizationalModel
]

organizational_model_registry: Dict[str, OrganizationalModelFactory] = {}


def register_organizational_model(
    model_id: str, factory: OrganizationalModelFactory
) -> None:
    """Register an organizational model factory under an experiment id."""
    if model_id in organizational_model_registry:
        raise ValueError(f"MMA organizational model id already registered: {model_id}")
    organizational_model_registry[model_id] = factory


def make_organizational_model(
    model_id: str, task: TaskClass, group_map: Mapping[str, Sequence[str]]
) -> OrganizationalModel:
    """Build and validate an organizational model selected from config."""
    try:
        factory = organizational_model_registry[model_id]
    except KeyError as err:
        known_ids = ", ".join(sorted(organizational_model_registry)) or "<none>"
        raise ValueError(
            f"Unknown MMA organizational model '{model_id}'. Known ids: {known_ids}"
        ) from err

    model = factory(task, group_map)
    model.validate(group_map)
    return model
