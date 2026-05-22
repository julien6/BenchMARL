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
    orbital_all,
    orbital_lb_action_only,
    orbital_lb_reward_only,
    orbital_mma_full,
    orbital_none,
    orbital_partial,
    orbital_pb_dcop_lite,
    orbital_rb_relay_heavy,
    orbital_rb_rule,
)
from .transforms import MMAGoalRewardTransform, MMARoleMaskTransform

register_organizational_model("orbital_none", orbital_none)
register_organizational_model("orbital_partial", orbital_partial)
register_organizational_model("orbital_all", orbital_all)
register_organizational_model("orbital_lb_reward_only", orbital_lb_reward_only)
register_organizational_model("orbital_lb_action_only", orbital_lb_action_only)
register_organizational_model("orbital_mma_full", orbital_mma_full)
register_organizational_model("orbital_rb_rule", orbital_rb_rule)
register_organizational_model("orbital_rb_relay_heavy", orbital_rb_relay_heavy)
register_organizational_model("orbital_pb_dcop_lite", orbital_pb_dcop_lite)
