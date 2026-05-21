#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import MISSING, dataclass
from typing import Optional


@dataclass
class TaskConfig:
    num_satellites: int = MISSING
    num_tasks: int = MISSING
    task_spawn_rate: float = MISSING
    energy_budget: float = MISSING
    comm_radius: int = MISSING
    p_link_drop: float = MISSING
    adversarial_rate: float = MISSING
    enable_debris: bool = MISSING
    reward_mode: str = MISSING
    max_steps: int = MISSING
    render_mode: Optional[str] = MISSING
