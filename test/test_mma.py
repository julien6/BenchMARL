#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

import pytest
import torch
from tensordict import TensorDict
from torchrl.data import Categorical, Composite, Unbounded

from benchmarl.environments import PettingZooTask
from benchmarl.mma import (
    GoalContext,
    MMAGoalRewardTransform,
    MMARoleMaskTransform,
    OrganizationalModel,
    ScriptedGoal,
    ScriptedRole,
    make_organizational_model,
)

GROUP_MAP = {"agents": ["agent_0", "agent_1"]}


def _action_spec():
    return Composite(
        {
            "agents": Composite(
                {"action": Categorical(3, shape=(2,))},
                shape=(2,),
            )
        }
    )


def _reward_spec():
    return Composite(
        {
            "agents": Composite(
                {"reward": Unbounded(shape=(2, 1))},
                shape=(2,),
            )
        }
    )


def _observation_tensordict():
    return TensorDict(
        {
            "agents": TensorDict(
                {"observation": torch.tensor([[0.0, 1.0], [1.0, 0.0]])},
                batch_size=(2,),
            )
        },
        batch_size=(),
    )


def _step_tensordicts(actions=None):
    current = _observation_tensordict()
    current["agents", "action"] = torch.tensor([1, 2]) if actions is None else actions
    next_tensordict = TensorDict(
        {
            "agents": TensorDict(
                {"reward": torch.tensor([[0.25], [0.50]])},
                batch_size=(2,),
            )
        },
        batch_size=(),
    )
    return current, next_tensordict


def test_role_mask_keeps_unassigned_agent_permissive():
    model = OrganizationalModel(
        roles={"only_action_1": ScriptedRole(lambda observation, agent_name: [1])},
        role_assignments={"agent_0": "only_action_1"},
    )
    transform = MMARoleMaskTransform(model, GROUP_MAP, _action_spec())

    reset = transform._reset(TensorDict(), _observation_tensordict())

    assert reset["agents", "action_mask"].tolist() == [
        [False, True, False],
        [True, True, True],
    ]


def test_role_mask_intersects_environment_mask():
    model = OrganizationalModel(
        roles={"actions_0_1": ScriptedRole(lambda observation, agent_name: [0, 1])},
        role_assignments={"agent_0": "actions_0_1"},
    )
    transform = MMARoleMaskTransform(model, GROUP_MAP, _action_spec())
    reset = _observation_tensordict()
    reset["agents", "action_mask"] = torch.tensor(
        [[True, False, True], [False, True, True]]
    )

    reset = transform._reset(TensorDict(), reset)

    assert reset["agents", "action_mask"].tolist() == [
        [True, False, False],
        [False, True, True],
    ]


def test_role_mask_rejects_empty_allowed_action_set():
    model = OrganizationalModel(
        roles={"none": ScriptedRole(lambda observation, agent_name: [])},
        role_assignments={"agent_0": "none"},
    )
    transform = MMARoleMaskTransform(model, GROUP_MAP, _action_spec())

    with pytest.raises(ValueError, match="allowed no actions"):
        transform._reset(TensorDict(), _observation_tensordict())


def test_goal_rewards_sum_assigned_goals_and_keep_raw_reward():
    model = OrganizationalModel(
        goals={
            "one": ScriptedGoal(lambda context, observation, action, agent_name: 1.0),
            "action": ScriptedGoal(
                lambda context, observation, action, agent_name: action.float()
            ),
        },
        goal_assignments={"agent_0": ["one", "action"]},
    )
    transform = MMAGoalRewardTransform(model, GROUP_MAP, _reward_spec())
    current, next_tensordict = _step_tensordicts()

    next_tensordict = transform._step(current, next_tensordict)

    assert next_tensordict["agents", "reward"].tolist() == [[2.25], [0.50]]
    assert next_tensordict["agents", "info", "mma_goal_bonus"].tolist() == [
        [2.0],
        [0.0],
    ]
    assert next_tensordict["agents", "info", "mma_raw_reward"].tolist() == [
        [0.25],
        [0.50],
    ]


def test_goal_context_is_stateful_until_reset():
    def reward(context: GoalContext, observation, action, agent_name):
        context.state["count"] = context.state.get("count", 0) + 1
        return context.state["count"]

    model = OrganizationalModel(
        goals={"counter": ScriptedGoal(reward)},
        goal_assignments={"agent_0": ["counter"]},
    )
    transform = MMAGoalRewardTransform(model, GROUP_MAP, _reward_spec())

    current, next_tensordict = _step_tensordicts()
    first = transform._step(current, next_tensordict)
    current, next_tensordict = _step_tensordicts()
    second = transform._step(current, next_tensordict)
    transform._reset(TensorDict(), _observation_tensordict())
    current, next_tensordict = _step_tensordicts()
    after_reset = transform._step(current, next_tensordict)

    assert first["agents", "info", "mma_goal_bonus"][0].item() == 1.0
    assert second["agents", "info", "mma_goal_bonus"][0].item() == 2.0
    assert after_reset["agents", "info", "mma_goal_bonus"][0].item() == 1.0


@pytest.mark.parametrize(
    "model_id",
    [
        "orbital_none",
        "orbital_partial",
        "orbital_all",
        "orbital_lb_reward_only",
        "orbital_lb_action_only",
        "orbital_mma_full",
        "orbital_rb_rule",
        "orbital_rb_relay_heavy",
        "orbital_pb_dcop_lite",
        "moise-marl",
    ],
)
def test_orbital_organizational_models_build_from_registry(model_id):
    model = make_organizational_model(
        model_id,
        task=PettingZooTask.ORBITAL.get_from_yaml(),
        group_map={"sat": [f"sat_{index}" for index in range(6)]},
    )

    assert isinstance(model, OrganizationalModel)


def test_article_orbital_baselines_split_roles_and_goals():
    group_map = {"sat": [f"sat_{index}" for index in range(6)]}
    task = PettingZooTask.ORBITAL.get_from_yaml()

    reward_only = make_organizational_model(
        "orbital_lb_reward_only", task=task, group_map=group_map
    )
    action_only = make_organizational_model(
        "orbital_lb_action_only", task=task, group_map=group_map
    )
    full = make_organizational_model("orbital_mma_full", task=task, group_map=group_map)

    assert not reward_only.roles
    assert set(reward_only.goals) == {
        "orbital_task_acquisition_goal",
        "orbital_data_delivery_goal",
        "orbital_fleet_resilience_goal",
    }
    assert set(reward_only.goal_assignments) == set(group_map["sat"])

    assert set(action_only.roles) == {
        "orbital_observer_role",
        "orbital_relay_role",
        "orbital_safety_guard_role",
    }
    assert set(action_only.role_assignments) == set(group_map["sat"])
    assert not action_only.goals

    assert full.role_assignments["sat_0"] == "orbital_observer_role"
    assert full.role_assignments["sat_1"] == "orbital_relay_role"
    assert full.role_assignments["sat_2"] == "orbital_safety_guard_role"
    assert full.goal_assignments["sat_0"] == ["orbital_task_acquisition_goal"]
    assert full.goal_assignments["sat_1"] == ["orbital_data_delivery_goal"]
    assert full.goal_assignments["sat_2"] == ["orbital_fleet_resilience_goal"]


@pytest.mark.parametrize(
    "model_id",
    ["orbital_rb_rule", "orbital_rb_relay_heavy", "orbital_pb_dcop_lite", "moise-marl"],
)
def test_handcrafted_orbital_baselines_are_role_only_single_action_policies(model_id):
    group_map = {"sat": [f"sat_{index}" for index in range(6)]}
    model = make_organizational_model(
        model_id, task=PettingZooTask.ORBITAL.get_from_yaml(), group_map=group_map
    )
    observations = [
        torch.zeros(20),
        torch.tensor([0.1, 1, 0.5, 0.4, 0, 1, 1, 1, 0.5, 0.7, 0.3, 0.2, 0.1] + [0] * 7),
    ]

    assert model.roles
    assert model.role_assignments == {
        agent_name: next(iter(model.roles)) for agent_name in group_map["sat"]
    }
    assert not model.goals
    assert not model.goal_assignments
    for agent_name in group_map["sat"]:
        for observation in observations:
            assert (
                len(
                    tuple(
                        model.role_for(agent_name).allowed_actions(
                            observation, agent_name
                        )
                    )
                )
                == 1
            )


def test_handcrafted_orbital_policy_priorities():
    group_map = {"sat": ["sat_0"]}
    task = PettingZooTask.ORBITAL.get_from_yaml()
    mission_observation = torch.zeros(20)
    mission_observation[6] = 1.0  # ground contact
    mission_observation[9] = 0.7  # buffered data
    mission_observation[10] = 0.7  # buffer remaining
    mission_observation[11] = 0.2  # local known tasks
    mission_observation[12] = 0.1  # local known task priority

    rule = make_organizational_model("orbital_rb_rule", task, group_map)
    relay_heavy = make_organizational_model("orbital_rb_relay_heavy", task, group_map)
    assert tuple(
        rule.role_for("sat_0").allowed_actions(mission_observation, "sat_0")
    ) == (0,)
    assert tuple(
        relay_heavy.role_for("sat_0").allowed_actions(mission_observation, "sat_0")
    ) == (1,)

    recharge_observation = torch.zeros(20)
    recharge_observation[0] = 0.1  # energy
    recharge_observation[5] = 1.0  # sunlight
    dcop_lite = make_organizational_model("orbital_pb_dcop_lite", task, group_map)
    assert tuple(
        dcop_lite.role_for("sat_0").allowed_actions(recharge_observation, "sat_0")
    ) == (5,)


def test_moise_marl_manual_policy_priorities():
    group_map = {"sat": ["sat_0"]}
    task = PettingZooTask.ORBITAL.get_from_yaml()
    model = make_organizational_model("moise-marl", task, group_map)
    role = model.role_for("sat_0")

    compromised = torch.zeros(20)
    compromised[15] = 1.0
    assert tuple(role.allowed_actions(compromised, "sat_0")) == (6,)

    ground_relay = torch.zeros(20)
    ground_relay[6] = 1.0
    ground_relay[9] = 0.2
    assert tuple(role.allowed_actions(ground_relay, "sat_0")) == (1,)

    observe = torch.zeros(20)
    observe[10] = 0.5
    observe[11] = 0.1
    assert tuple(role.allowed_actions(observe, "sat_0")) == (0,)


def test_unknown_organizational_model_id_is_explicit():
    with pytest.raises(ValueError, match="Unknown MMA organizational model"):
        make_organizational_model(
            "unknown",
            task=PettingZooTask.ORBITAL.get_from_yaml(),
            group_map={"sat": ["sat_0"]},
        )


def test_organizational_model_ascii_summary_lists_descriptions_and_assignments():
    model = OrganizationalModel(
        roles={
            "observer": ScriptedRole(
                lambda observation, agent_name: [0],
                "Keeps the observer on its acquisition rule.",
            )
        },
        goals={
            "deliver": ScriptedGoal(
                lambda context, observation, action, agent_name: 1.0,
                "Rewards useful data delivery.",
            )
        },
        role_assignments={"agent_1": "observer"},
        goal_assignments={"agent_1": ["deliver"], "agent_0": ["deliver"]},
    )

    summary = model.ascii_summary("demo")

    assert "MMA organizational model: demo" in summary
    assert "+----------+" in summary
    assert "| Role" in summary
    assert "| Goal" in summary
    assert "Keeps the observer on its acquisition rule." in summary
    assert "Rewards useful data delivery." in summary
    assert "agent_0, agent_1" in summary
