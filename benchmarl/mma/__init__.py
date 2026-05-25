#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from .core import (
    Goal,
    GoalContext,
    OrganizationalModel,
    Role,
    ScriptedGoal,
    ScriptedRole,
    make_organizational_model,
    organizational_model_registry,
    register_organizational_model,
)
from .orbital import (
    handcrafted,
    lb_action_only,
    lb_moise_marl,
    lb_reward_only,
    lb_unconstrained,
    rb_acquirer,
    rb_dcop_like,
    rb_deliverer,
)
from .transforms import MMAGoalRewardTransform, MMARoleMaskTransform

register_organizational_model("handcrafted", handcrafted)
register_organizational_model("lb_unconstrained", lb_unconstrained)
register_organizational_model("lb_moise_marl", lb_moise_marl)
register_organizational_model("lb_action_only", lb_action_only)
register_organizational_model("lb_reward_only", lb_reward_only)
register_organizational_model("rb_deliverer", rb_deliverer)
register_organizational_model("rb_dcop_like", rb_dcop_like)
register_organizational_model("rb_acquirer", rb_acquirer)

# Backward-compatible aliases for existing configs and runs.
register_organizational_model("orbital_none", lb_unconstrained)
register_organizational_model("orbital_lb_reward_only", lb_reward_only)
register_organizational_model("orbital_lb_action_only", lb_action_only)
register_organizational_model("orbital_mma_full", lb_moise_marl)
register_organizational_model("orbital_all", lb_moise_marl)
register_organizational_model("orbital_partial", lb_action_only)
register_organizational_model("moise-marl", handcrafted)
