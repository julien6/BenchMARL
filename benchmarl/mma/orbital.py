#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from typing import Mapping, Sequence

from torch import Tensor

from benchmarl.environments.common import TaskClass

from .core import GoalContext, OrganizationalModel, ScriptedGoal, ScriptedRole

OBS = 0
REL_GRN = 1
REL_SAT = 2
DN = 3
UP = 4
PWR = 5
SCAN = 6
IDLE = 7

ENERGY = 0
SUNLIGHT = 5
GROUND_CONTACT = 6
GROUND_ROUTE = 7
LOCAL_DEGREE = 8
BUFFERED_DATA = 9
BUFFER_REMAINING = 10
KNOWN_NEARBY_TASKS = 11
LOCAL_PC = 14
COMPROMISED = 15

LOW_ENERGY = 0.25
LOW_SAFETY_ENERGY = 0.20
USEFUL_BUFFER_SPACE = 0.30
BUFFERED_MISSION_DATA = 0.05
DEBRIS_ALERT = 0.35


def _check_orbital(task: TaskClass) -> None:
    if task.env_name() != "pettingzoo" or task.name != "ORBITAL":
        raise ValueError(
            "ORBITAL MMA models are only compatible with task=pettingzoo/orbital."
        )


def _safe_fallback(observation: Tensor) -> set[int]:
    allowed = {IDLE}
    if observation[ENERGY] < LOW_ENERGY and observation[SUNLIGHT] > 0.5:
        allowed.add(PWR)
    if observation[COMPROMISED] > 0.5:
        allowed.add(SCAN)
    if observation[LOCAL_PC] > DEBRIS_ALERT:
        allowed.update((DN, UP))
    return allowed


def _acquisition_actions(observation: Tensor, agent_name: str) -> set[int]:
    allowed = _safe_fallback(observation)
    if observation[GROUND_CONTACT] > 0.5:
        allowed.add(REL_GRN)
    if (
        observation[KNOWN_NEARBY_TASKS] > 0.0
        and observation[BUFFER_REMAINING] > USEFUL_BUFFER_SPACE
    ):
        allowed.add(OBS)
    if observation[KNOWN_NEARBY_TASKS] > 0.0 and observation[LOCAL_DEGREE] > 0.0:
        allowed.add(REL_SAT)
    return allowed


def _relay_actions(observation: Tensor, agent_name: str) -> set[int]:
    allowed = _safe_fallback(observation)
    if observation[GROUND_CONTACT] > 0.5:
        allowed.add(REL_GRN)
    if observation[BUFFERED_DATA] > BUFFERED_MISSION_DATA and (
        observation[GROUND_ROUTE] > 0.0 or observation[LOCAL_DEGREE] > 0.0
    ):
        allowed.add(REL_SAT)
    if observation[KNOWN_NEARBY_TASKS] > 0.0 and observation[LOCAL_DEGREE] > 0.0:
        allowed.add(REL_SAT)
    return allowed


def _safety_actions(observation: Tensor, agent_name: str) -> set[int]:
    allowed = _safe_fallback(observation)
    allowed.update((PWR, SCAN))
    return allowed


def _ground_intake_goal(
    context: GoalContext, observation: Tensor, action: Tensor, agent_name: str
) -> float:
    useful_contact = observation[GROUND_CONTACT] > 0.5 and (
        observation[KNOWN_NEARBY_TASKS] > 0.0
        or observation[BUFFERED_DATA] > BUFFERED_MISSION_DATA
    )
    if useful_contact and int(action.item()) == REL_GRN:
        context.state["ground_relays"] = context.state.get("ground_relays", 0) + 1
        return 1.0
    return 0.0


def _observation_goal(
    context: GoalContext, observation: Tensor, action: Tensor, agent_name: str
) -> float:
    useful_observation = (
        observation[KNOWN_NEARBY_TASKS] > 0.0
        and observation[BUFFER_REMAINING] > USEFUL_BUFFER_SPACE
    )
    if useful_observation and int(action.item()) == OBS:
        context.state["observations"] = context.state.get("observations", 0) + 1
        return 1.0
    return 0.0


def _delivery_goal(
    context: GoalContext, observation: Tensor, action: Tensor, agent_name: str
) -> float:
    if observation[BUFFERED_DATA] <= BUFFERED_MISSION_DATA:
        return 0.0
    act = int(action.item())
    direct_delivery = observation[GROUND_CONTACT] > 0.5 and act == REL_GRN
    route_delivery = (
        observation[GROUND_ROUTE] > 0.0 or observation[LOCAL_DEGREE] > 0.0
    ) and act == REL_SAT
    if direct_delivery or route_delivery:
        context.state["relays"] = context.state.get("relays", 0) + 1
        return 1.0
    return 0.0


def _task_acquisition_goal(
    context: GoalContext, observation: Tensor, action: Tensor, agent_name: str
) -> float:
    act = int(action.item())
    ground_catalog_intake = (
        act == REL_GRN
        and observation[GROUND_CONTACT] > 0.5
        and observation[BUFFERED_DATA] <= BUFFERED_MISSION_DATA
    )
    useful_observation = (
        act == OBS
        and observation[KNOWN_NEARBY_TASKS] > 0.0
        and observation[BUFFER_REMAINING] > USEFUL_BUFFER_SPACE
    )
    if ground_catalog_intake or useful_observation:
        context.state["acquisition_steps"] = (
            context.state.get("acquisition_steps", 0) + 1
        )
        return 1.0
    return 0.0


def _fleet_resilience_goal(
    context: GoalContext, observation: Tensor, action: Tensor, agent_name: str
) -> float:
    act = int(action.item())
    cyber_response = act == SCAN and observation[COMPROMISED] > 0.5
    energy_response = (
        act == PWR
        and observation[ENERGY] < LOW_SAFETY_ENERGY
        and observation[SUNLIGHT] > 0.5
    )
    debris_response = act in (DN, UP) and observation[LOCAL_PC] > DEBRIS_ALERT
    if cyber_response or energy_response or debris_response:
        context.state["resilience_steps"] = context.state.get("resilience_steps", 0) + 1
        return 1.0
    return 0.0


def _orbital_specs() -> tuple[dict, dict]:
    roles = {
        "orbital_acquisition_role": ScriptedRole(
            _acquisition_actions,
            "Allows data acquisition and useful ground intake when local state permits.",
        ),
        "orbital_relay_role": ScriptedRole(
            _relay_actions,
            "Allows buffered data to reach ground or a relevant satellite route.",
        ),
        "orbital_safety_role": ScriptedRole(
            _safety_actions,
            "Keeps safety actions available for energy, cyber, and orbital stress.",
        ),
    }
    goals = {
        "orbital_ground_intake_goal": ScriptedGoal(
            _ground_intake_goal,
            "Rewards use of a relevant ground contact through REL_GRN.",
        ),
        "orbital_observation_goal": ScriptedGoal(
            _observation_goal,
            "Rewards OBS when a nearby known task and buffer space make it useful.",
        ),
        "orbital_delivery_goal": ScriptedGoal(
            _delivery_goal,
            "Rewards direct or routed delivery of data already in the buffer.",
        ),
    }
    return roles, goals


def _article_specs() -> tuple[dict, dict]:
    roles = {
        "orbital_observer_role": ScriptedRole(
            _acquisition_actions,
            "Prioritizes catalog intake and observation when acquisition is feasible.",
        ),
        "orbital_relay_role": ScriptedRole(
            _relay_actions,
            "Prioritizes useful ground and satellite relays for mission continuity.",
        ),
        "orbital_safety_guard_role": ScriptedRole(
            _safety_actions,
            "Prioritizes low-power, cyber-scan, and debris mitigation actions.",
        ),
    }
    goals = {
        "orbital_task_acquisition_goal": ScriptedGoal(
            _task_acquisition_goal,
            "Rewards ground catalog intake and feasible observation of known tasks.",
        ),
        "orbital_data_delivery_goal": ScriptedGoal(
            _delivery_goal,
            "Rewards direct or routed delivery of data already in the buffer.",
        ),
        "orbital_fleet_resilience_goal": ScriptedGoal(
            _fleet_resilience_goal,
            "Rewards energy, cyber, and debris responses when the fleet is stressed.",
        ),
    }
    return roles, goals


def _orbital_agents(group_map: Mapping[str, Sequence[str]]) -> list[str]:
    if set(group_map) != {"sat"}:
        raise ValueError("ORBITAL MMA models expect the single BenchMARL 'sat' group.")

    def sat_index(agent_name: str) -> tuple[int, str]:
        try:
            return int(agent_name.rsplit("_", 1)[1]), agent_name
        except (IndexError, ValueError):
            return 10**9, agent_name

    return sorted(group_map["sat"], key=sat_index)


def _role_goals(role_name: str) -> list[str]:
    if role_name == "orbital_acquisition_role":
        return ["orbital_ground_intake_goal", "orbital_observation_goal"]
    if role_name == "orbital_relay_role":
        return ["orbital_delivery_goal"]
    return []


def _article_role_goals(role_name: str) -> list[str]:
    if role_name == "orbital_observer_role":
        return ["orbital_task_acquisition_goal"]
    if role_name == "orbital_relay_role":
        return ["orbital_data_delivery_goal"]
    if role_name == "orbital_safety_guard_role":
        return ["orbital_fleet_resilience_goal"]
    return []


def orbital_none(
    task: TaskClass, group_map: Mapping[str, Sequence[str]]
) -> OrganizationalModel:
    _check_orbital(task)
    _orbital_agents(group_map)
    return OrganizationalModel()


def _orbital_assigned(
    task: TaskClass,
    group_map: Mapping[str, Sequence[str]],
    agents: Sequence[str],
) -> OrganizationalModel:
    _check_orbital(task)
    roles, goals = _orbital_specs()
    role_names = tuple(roles)
    role_assignments = {
        agent_name: role_names[index % len(role_names)]
        for index, agent_name in enumerate(agents)
    }
    goal_assignments = {
        agent_name: _role_goals(role_name)
        for agent_name, role_name in role_assignments.items()
        if _role_goals(role_name)
    }
    return OrganizationalModel(
        roles=roles,
        goals=goals,
        role_assignments=role_assignments,
        goal_assignments=goal_assignments,
    )


def _article_role_assignments(agents: Sequence[str], roles: Mapping[str, object]):
    role_names = tuple(roles)
    return {
        agent_name: role_names[index % len(role_names)]
        for index, agent_name in enumerate(agents)
    }


def orbital_lb_reward_only(
    task: TaskClass, group_map: Mapping[str, Sequence[str]]
) -> OrganizationalModel:
    """Learning baseline with ORBITAL mission shaping and free actions."""
    _check_orbital(task)
    agents = _orbital_agents(group_map)
    _, goals = _article_specs()
    return OrganizationalModel(
        goals=goals,
        goal_assignments={agent_name: tuple(goals) for agent_name in agents},
    )


def orbital_lb_action_only(
    task: TaskClass, group_map: Mapping[str, Sequence[str]]
) -> OrganizationalModel:
    """Learning baseline with ORBITAL action masks and no mission shaping."""
    _check_orbital(task)
    agents = _orbital_agents(group_map)
    roles, _ = _article_specs()
    return OrganizationalModel(
        roles=roles,
        role_assignments=_article_role_assignments(agents, roles),
    )


def orbital_mma_full(
    task: TaskClass, group_map: Mapping[str, Sequence[str]]
) -> OrganizationalModel:
    """ORBITAL role-action and mission-goal MMA model."""
    _check_orbital(task)
    agents = _orbital_agents(group_map)
    roles, goals = _article_specs()
    role_assignments = _article_role_assignments(agents, roles)
    return OrganizationalModel(
        roles=roles,
        goals=goals,
        role_assignments=role_assignments,
        goal_assignments={
            agent_name: _article_role_goals(role_name)
            for agent_name, role_name in role_assignments.items()
        },
    )


def orbital_partial(
    task: TaskClass, group_map: Mapping[str, Sequence[str]]
) -> OrganizationalModel:
    agents = _orbital_agents(group_map)
    n_assigned = max(1, (len(agents) + 1) // 2)
    return _orbital_assigned(task, group_map, agents[:n_assigned])


def orbital_all(
    task: TaskClass, group_map: Mapping[str, Sequence[str]]
) -> OrganizationalModel:
    return _orbital_assigned(task, group_map, _orbital_agents(group_map))
