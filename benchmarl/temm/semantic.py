#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Protocol

from torch import Tensor


class TEMMSemanticAdapter(Protocol):
    name: str

    def describe_step(
        self,
        observation: Tensor,
        action: Tensor,
        next_observation: Tensor,
        agent_name: str,
    ) -> Dict[str, Any]:
        ...

    def action_map(self) -> List[Dict[str, Any]]:
        ...


@dataclass
class GenericSemanticAdapter:
    name: str = "generic"

    def describe_step(
        self,
        observation: Tensor,
        action: Tensor,
        next_observation: Tensor,
        agent_name: str,
    ) -> Dict[str, Any]:
        action_label = _action_label(action)
        return {
            "adapter": self.name,
            "action_label": action_label,
            "state_tags": [],
            "transition_tags": [],
            "interpretation": f"takes action {action_label}",
        }

    def action_map(self) -> List[Dict[str, Any]]:
        return []


class OrbitalSemanticAdapter(GenericSemanticAdapter):
    def __init__(self):
        self.name = "orbital"

    def describe_step(
        self,
        observation: Tensor,
        action: Tensor,
        next_observation: Tensor,
        agent_name: str,
    ) -> Dict[str, Any]:
        from benchmarl.mma import orbital

        obs = observation.flatten().float()
        next_obs = next_observation.flatten().float()
        action_id = _discrete_action_id(action)
        action_label = ORBITAL_ACTION_LABELS.get(action_id, f"action_{action_id}")
        state_tags = _orbital_state_tags(obs)
        transition_tags = _orbital_transition_tags(obs, next_obs)
        interpretation = _orbital_interpretation(action_id, state_tags)
        return {
            "adapter": self.name,
            "action_label": action_label,
            "state_tags": state_tags,
            "transition_tags": transition_tags,
            "interpretation": interpretation,
            "action_id": action_id,
            "agent": agent_name,
            "thresholds": {
                "low_energy": orbital.LOW_ENERGY,
                "buffered_mission_data": orbital.BUFFERED_MISSION_DATA,
                "debris_alert": orbital.DEBRIS_ALERT,
            },
        }

    def action_map(self) -> List[Dict[str, Any]]:
        return [
            {
                "action_id": action_id,
                "action_label": label,
                "description": ORBITAL_ACTION_DESCRIPTIONS.get(label, "-"),
            }
            for action_id, label in sorted(ORBITAL_ACTION_LABELS.items())
        ]


ORBITAL_ACTION_LABELS = {
    0: "observe",
    1: "relay_ground",
    2: "relay_satellite",
    3: "move_down",
    4: "move_up",
    5: "power_save",
    6: "scan",
    7: "idle",
}

ORBITAL_ACTION_DESCRIPTIONS = {
    "observe": "Acquire useful local task data when nearby work and buffer space are available.",
    "relay_ground": "Relay data or task information through direct ground contact.",
    "relay_satellite": "Relay data or task information through a satellite route or neighbor.",
    "move_down": "Adjust orbit downward, often useful under debris or collision pressure.",
    "move_up": "Adjust orbit upward, often useful under debris or collision pressure.",
    "power_save": "Recharge or conserve power when energy is low and sunlight is available.",
    "scan": "Perform a safety/cyber scan when the satellite is compromised.",
    "idle": "Keep the current behavior when no stronger local trigger is detected.",
}


def resolve_semantic_adapter(config, experiment_or_task=None) -> TEMMSemanticAdapter:
    adapter_name = getattr(config, "semantic_adapter", "auto")
    if adapter_name == "none":
        return GenericSemanticAdapter()
    if adapter_name == "orbital":
        return OrbitalSemanticAdapter()
    if _is_orbital(experiment_or_task):
        return OrbitalSemanticAdapter()
    return GenericSemanticAdapter()


def _is_orbital(experiment_or_task) -> bool:
    task = getattr(experiment_or_task, "task", experiment_or_task)
    task_name = getattr(task, "name", None)
    env_name = None
    env_name_fn = getattr(task, "env_name", None)
    if callable(env_name_fn):
        try:
            env_name = env_name_fn()
        except TypeError:
            env_name = None
    return task_name == "ORBITAL" or (
        str(task_name).lower() == "orbital" and str(env_name).lower() == "pettingzoo"
    )


def _orbital_state_tags(observation: Tensor) -> List[str]:
    from benchmarl.mma import orbital

    tags = []
    if _value(observation, orbital.ENERGY) < orbital.LOW_ENERGY:
        tags.append("low_energy")
    if _value(observation, orbital.SUNLIGHT) > 0.5:
        tags.append("sunlight_available")
    if _value(observation, orbital.GROUND_CONTACT) > 0.5:
        tags.append("ground_contact")
    if _value(observation, orbital.BUFFERED_DATA) > orbital.BUFFERED_MISSION_DATA:
        tags.append("has_buffered_data")
    if _value(observation, orbital.BUFFER_REMAINING) > orbital.USEFUL_BUFFER_SPACE:
        tags.append("buffer_space_available")
    if _value(observation, orbital.KNOWN_NEARBY_TASKS) > 0.0:
        tags.append("known_nearby_task")
    if (
        _value(observation, orbital.GROUND_ROUTE) > 0.0
        or _value(observation, orbital.LOCAL_DEGREE) > 0.0
    ):
        tags.append("relay_route_available")
    if _value(observation, orbital.LOCAL_PC) > orbital.DEBRIS_ALERT:
        tags.append("collision_risk")
    if _value(observation, orbital.COMPROMISED) > 0.5:
        tags.append("compromised")
    return tags


def _orbital_transition_tags(observation: Tensor, next_observation: Tensor) -> List[str]:
    from benchmarl.mma import orbital

    tags = []
    energy_delta = _value(next_observation, orbital.ENERGY) - _value(
        observation, orbital.ENERGY
    )
    data_delta = _value(next_observation, orbital.BUFFERED_DATA) - _value(
        observation, orbital.BUFFERED_DATA
    )
    risk_delta = _value(next_observation, orbital.LOCAL_PC) - _value(
        observation, orbital.LOCAL_PC
    )
    if energy_delta > 0.03:
        tags.append("energy_increased")
    elif energy_delta < -0.03:
        tags.append("energy_decreased")
    if data_delta > 0.03:
        tags.append("buffer_increased")
    elif data_delta < -0.03:
        tags.append("buffer_decreased")
    if risk_delta > 0.03:
        tags.append("risk_increased")
    elif risk_delta < -0.03:
        tags.append("risk_decreased")
    if not tags:
        tags.append("stable_transition")
    return tags


def _orbital_interpretation(action_id: int, state_tags: List[str]) -> str:
    from benchmarl.mma import orbital

    if action_id == orbital.OBS:
        if "known_nearby_task" in state_tags and "buffer_space_available" in state_tags:
            return "observes available task"
        return "attempts observation"
    if action_id == orbital.REL_GRN:
        if "ground_contact" in state_tags:
            return "relays through ground contact"
        return "attempts ground relay"
    if action_id == orbital.REL_SAT:
        if "relay_route_available" in state_tags:
            return "relays through satellite route"
        return "attempts satellite relay"
    if action_id == orbital.PWR:
        if "low_energy" in state_tags and "sunlight_available" in state_tags:
            return "recharges under low energy"
        return "uses power-save behavior"
    if action_id == orbital.SCAN:
        if "compromised" in state_tags:
            return "performs safety scan"
        return "scans for safety"
    if action_id in (orbital.DN, orbital.UP):
        if "collision_risk" in state_tags:
            return "maneuvers under collision risk"
        return "adjusts orbital position"
    if action_id == orbital.IDLE:
        return "idle while stable"
    return f"takes action {action_id}"


def _value(observation: Tensor, index: int) -> float:
    if observation.numel() <= index:
        return 0.0
    return float(observation[index].item())


def _discrete_action_id(action: Tensor) -> int:
    flat = action.flatten()
    if flat.numel() == 0:
        return -1
    return int(float(flat[0].item()))


def _action_label(action: Tensor) -> str:
    flat = action.flatten()
    if flat.numel() == 1 and float(flat[0].item()).is_integer():
        return f"act:{int(flat[0].item())}"
    return ",".join(f"{float(value):.2f}" for value in flat[:4])
