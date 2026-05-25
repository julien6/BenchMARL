#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

import random
from typing import Mapping, Sequence

from torch import Tensor

from benchmarl.environments.common import TaskClass

from .core import GoalContext, OrganizationalModel, Role, ScriptedGoal

OBS = 0
REL_GRN = 1
REL_SAT = 2
DN = 3
UP = 4
PWR = 5
SCAN = 6
IDLE = 7

ENERGY = 0
RADIUS = 3
SUNLIGHT = 5
GROUND_CONTACT = 6
GROUND_ROUTE = 7
LOCAL_DEGREE = 8
BUFFERED_DATA = 9
BUFFER_REMAINING = 10
KNOWN_NEARBY_TASKS = 11
LOCAL_PC = 14
COMPROMISED = 15
JAMMED = 17

LOW_ENERGY = 0.25
LOW_SAFETY_ENERGY = 0.20
USEFUL_BUFFER_SPACE = 0.30
BUFFERED_MISSION_DATA = 0.05
DEBRIS_ALERT = 0.35
PARTIAL_CONSTRAINT_HARDNESS = 0.30
FULL_CONSTRAINT_HARDNESS = 1.00
ORBITAL_ACTIONS = (OBS, REL_GRN, REL_SAT, DN, UP, PWR, SCAN, IDLE)


def _check_orbital(task: TaskClass) -> None:
    if task.env_name() != "pettingzoo" or task.name != "ORBITAL":
        raise ValueError(
            "ORBITAL MMA models are only compatible with task=pettingzoo/orbital."
        )


class ConstraintHardnessRole(Role):
    """Role that sometimes yields control back to the neural policy."""

    def __init__(self, rule, hardness: float, description: str = ""):
        self.rule = rule
        self.hardness = hardness
        self.description = description

    def allowed_actions(self, observation: Tensor, agent_name: str) -> Sequence[int]:
        if self.hardness < FULL_CONSTRAINT_HARDNESS and random.random() > self.hardness:
            return ORBITAL_ACTIONS
        return (int(self.rule(observation, agent_name)),)


def _known_tasks(observation: Tensor) -> int:
    return 1 if observation[KNOWN_NEARBY_TASKS] > 0.0 else 0


def _has_data(observation: Tensor) -> bool:
    return bool(observation[BUFFERED_DATA] > BUFFERED_MISSION_DATA)


def _can_observe(observation: Tensor) -> bool:
    return bool(
        observation[KNOWN_NEARBY_TASKS] > 0.0
        and observation[BUFFER_REMAINING] > 0.25
        and observation[BUFFERED_DATA] < 0.85
    )


def _has_relay_path(observation: Tensor) -> bool:
    return bool(observation[GROUND_ROUTE] > 0.0 or observation[LOCAL_DEGREE] > 0.0)


def _acquirer_action(observation: Tensor, agent_name: str) -> int:
    """Acquire catalog knowledge or observe known nearby tasks."""
    if _can_observe(observation):
        return OBS
    if observation[GROUND_CONTACT] > 0.5 and _known_tasks(observation) == 0:
        return REL_GRN
    return IDLE


def _deliverer_action(observation: Tensor, agent_name: str) -> int:
    """Deliver buffered data directly or through peer relays."""
    if observation[GROUND_CONTACT] > 0.5 and _has_data(observation):
        return REL_GRN
    if _has_data(observation) and _has_relay_path(observation):
        return REL_SAT
    if observation[LOCAL_DEGREE] > 0.0 and _known_tasks(observation) > 0:
        return REL_SAT
    return IDLE


def _stabilizer_action(observation: Tensor, agent_name: str) -> int:
    """Stabilize cyber, energy, and local orbital safety state."""
    if observation[COMPROMISED] > 0.5:
        return SCAN
    if observation[JAMMED] > 0.5:
        if observation[ENERGY] < 0.25 and observation[SUNLIGHT] > 0.5:
            return PWR
        return IDLE
    if observation[ENERGY] < LOW_SAFETY_ENERGY and observation[SUNLIGHT] > 0.5:
        return PWR
    if observation[LOCAL_PC] > DEBRIS_ALERT:
        return UP if observation[RADIUS] < 0.5 else DN
    return IDLE


def _handcrafted_full_action(observation: Tensor, agent_name: str) -> int:
    """Coordinate catalog intake, observations, delivery, and peer relays."""
    known_tasks = 1 if observation[KNOWN_NEARBY_TASKS] > 0.0 else 0
    has_data = observation[BUFFERED_DATA] > BUFFERED_MISSION_DATA
    can_observe = (
        observation[KNOWN_NEARBY_TASKS] > 0.0
        and observation[BUFFER_REMAINING] > 0.25
        and observation[BUFFERED_DATA] < 0.85
    )

    if observation[COMPROMISED] > 0.5:
        return SCAN

    if observation[JAMMED] > 0.5:
        if observation[ENERGY] < 0.25 and observation[SUNLIGHT] > 0.5:
            return PWR
        return IDLE

    if observation[GROUND_CONTACT] > 0.5 and has_data:
        return REL_GRN

    if can_observe:
        return OBS

    if observation[GROUND_CONTACT] > 0.5 and known_tasks == 0:
        return REL_GRN

    if has_data and (
        observation[GROUND_ROUTE] > 0.0 or observation[LOCAL_DEGREE] > 0.0
    ):
        return REL_SAT

    if observation[LOCAL_DEGREE] > 0.0 and known_tasks > 0:
        return REL_SAT

    if observation[ENERGY] < 0.20 and observation[SUNLIGHT] > 0.5:
        return PWR

    return IDLE


def _role_logic_goal(
    context: GoalContext,
    observation: Tensor,
    action: Tensor,
    agent_name: str,
    rule,
    state_key: str,
) -> float:
    expected_action = int(rule(observation, agent_name))
    if expected_action == IDLE:
        return 0.0
    if int(action.item()) == expected_action:
        context.state[state_key] = context.state.get(state_key, 0) + 1
        return 1.0
    return -0.25


def _acquisition_goal(
    context: GoalContext, observation: Tensor, action: Tensor, agent_name: str
) -> float:
    return _role_logic_goal(
        context, observation, action, agent_name, _acquirer_action, "acquisitions"
    )


def _delivery_goal(
    context: GoalContext, observation: Tensor, action: Tensor, agent_name: str
) -> float:
    return _role_logic_goal(
        context, observation, action, agent_name, _deliverer_action, "deliveries"
    )


def _stability_goal(
    context: GoalContext, observation: Tensor, action: Tensor, agent_name: str
) -> float:
    return _role_logic_goal(
        context, observation, action, agent_name, _stabilizer_action, "stabilizations"
    )


def _role_specs(
    acquirer_hardness: float,
    deliverer_hardness: float,
    stabilizer_hardness: float,
) -> dict[str, Role]:
    return {
        "acquirer": ConstraintHardnessRole(
            _acquirer_action,
            acquirer_hardness,
            "Acquires task catalog entries and observes known nearby tasks.",
        ),
        "deliverer": ConstraintHardnessRole(
            _deliverer_action,
            deliverer_hardness,
            "Delivers buffered data to ground or through peer relays.",
        ),
        "stabilizer": ConstraintHardnessRole(
            _stabilizer_action,
            stabilizer_hardness,
            "Handles cyber, energy, and local orbital safety responses.",
        ),
    }


def _partial_roles() -> dict[str, Role]:
    return _role_specs(
        PARTIAL_CONSTRAINT_HARDNESS,
        PARTIAL_CONSTRAINT_HARDNESS,
        PARTIAL_CONSTRAINT_HARDNESS,
    )


def _goal_specs() -> dict[str, ScriptedGoal]:
    return {
        "acquirer_goal": ScriptedGoal(
            _acquisition_goal,
            "Rewards acquirer-compatible catalog intake and observation decisions.",
        ),
        "deliverer_goal": ScriptedGoal(
            _delivery_goal,
            "Rewards deliverer-compatible ground and satellite relay decisions.",
        ),
        "stabilizer_goal": ScriptedGoal(
            _stability_goal,
            "Rewards stabilizer-compatible cyber, energy, and debris responses.",
        ),
    }


def _orbital_agents(group_map: Mapping[str, Sequence[str]]) -> list[str]:
    if set(group_map) != {"sat"}:
        raise ValueError("ORBITAL MMA models expect the single BenchMARL 'sat' group.")

    def sat_index(agent_name: str) -> tuple[int, str]:
        try:
            return int(agent_name.rsplit("_", 1)[1]), agent_name
        except (IndexError, ValueError):
            return 10**9, agent_name

    return sorted(group_map["sat"], key=sat_index)


def _role_assignments(agents: Sequence[str], roles: Mapping[str, object]):
    role_names = tuple(roles)
    return {
        agent_name: role_names[index % len(role_names)]
        for index, agent_name in enumerate(agents)
    }


def _all_goal_assignments(agents: Sequence[str], goals: Mapping[str, object]):
    return {agent_name: tuple(goals) for agent_name in agents}


def handcrafted(
    task: TaskClass, group_map: Mapping[str, Sequence[str]]
) -> OrganizationalModel:
    """Fully scripted handcrafted baseline."""
    _check_orbital(task)
    agents = _orbital_agents(group_map)
    roles = {
        "handcrafted_full": ConstraintHardnessRole(
            _handcrafted_full_action,
            FULL_CONSTRAINT_HARDNESS,
            "Fully constrains agents to the handcrafted ORBITAL policy.",
        )
    }
    return OrganizationalModel(
        roles=roles,
        role_assignments={agent_name: "handcrafted_full" for agent_name in agents},
    )


def lb_unconstrained(
    task: TaskClass, group_map: Mapping[str, Sequence[str]]
) -> OrganizationalModel:
    _check_orbital(task)
    _orbital_agents(group_map)
    return OrganizationalModel()


def lb_moise_marl(
    task: TaskClass, group_map: Mapping[str, Sequence[str]]
) -> OrganizationalModel:
    """MOISE+MARL baseline with partial roles and reward-shaping goals."""
    _check_orbital(task)
    agents = _orbital_agents(group_map)
    roles = _partial_roles()
    goals = _goal_specs()
    return OrganizationalModel(
        roles=roles,
        goals=goals,
        role_assignments=_role_assignments(agents, roles),
        goal_assignments=_all_goal_assignments(agents, goals),
    )


def lb_action_only(
    task: TaskClass, group_map: Mapping[str, Sequence[str]]
) -> OrganizationalModel:
    """Learning baseline with partial role shielding and no reward shaping."""
    _check_orbital(task)
    agents = _orbital_agents(group_map)
    roles = _partial_roles()
    return OrganizationalModel(
        roles=roles,
        role_assignments=_role_assignments(agents, roles),
    )


def lb_reward_only(
    task: TaskClass, group_map: Mapping[str, Sequence[str]]
) -> OrganizationalModel:
    """Learning baseline with reward shaping and unconstrained actions."""
    _check_orbital(task)
    agents = _orbital_agents(group_map)
    goals = _goal_specs()
    return OrganizationalModel(
        goals=goals,
        goal_assignments=_all_goal_assignments(agents, goals),
    )


def _mixed_role_baseline(
    task: TaskClass,
    group_map: Mapping[str, Sequence[str]],
    acquirer_hardness: float,
    deliverer_hardness: float,
    stabilizer_hardness: float,
) -> OrganizationalModel:
    _check_orbital(task)
    agents = _orbital_agents(group_map)
    roles = _role_specs(acquirer_hardness, deliverer_hardness, stabilizer_hardness)
    return OrganizationalModel(
        roles=roles,
        role_assignments=_role_assignments(agents, roles),
    )


def rb_deliverer(
    task: TaskClass, group_map: Mapping[str, Sequence[str]]
) -> OrganizationalModel:
    """Role baseline with a fully constrained deliverer role."""
    return _mixed_role_baseline(
        task=task,
        group_map=group_map,
        acquirer_hardness=PARTIAL_CONSTRAINT_HARDNESS,
        deliverer_hardness=FULL_CONSTRAINT_HARDNESS,
        stabilizer_hardness=PARTIAL_CONSTRAINT_HARDNESS,
    )


def rb_dcop_like(
    task: TaskClass, group_map: Mapping[str, Sequence[str]]
) -> OrganizationalModel:
    """Role baseline with fully constrained acquirer and deliverer roles."""
    return _mixed_role_baseline(
        task=task,
        group_map=group_map,
        acquirer_hardness=FULL_CONSTRAINT_HARDNESS,
        deliverer_hardness=FULL_CONSTRAINT_HARDNESS,
        stabilizer_hardness=PARTIAL_CONSTRAINT_HARDNESS,
    )


def rb_acquirer(
    task: TaskClass, group_map: Mapping[str, Sequence[str]]
) -> OrganizationalModel:
    """Role baseline with a fully constrained acquirer role."""
    return _mixed_role_baseline(
        task=task,
        group_map=group_map,
        acquirer_hardness=FULL_CONSTRAINT_HARDNESS,
        deliverer_hardness=PARTIAL_CONSTRAINT_HARDNESS,
        stabilizer_hardness=PARTIAL_CONSTRAINT_HARDNESS,
    )


# Compatibility aliases for configs produced before the baseline rename.
orbital_none = lb_unconstrained
orbital_lb_reward_only = lb_reward_only
orbital_lb_action_only = lb_action_only
orbital_mma_full = lb_moise_marl
orbital_all = lb_moise_marl
orbital_partial = lb_action_only
moise_marl = handcrafted
