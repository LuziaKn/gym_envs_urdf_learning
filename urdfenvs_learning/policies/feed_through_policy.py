from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from gym.spaces import Space
from typing import Any, Dict, Optional, Type, Union, List

from urdfenvs.policies.policy_base import BasePolicy

class FeedThroughPolicy(BasePolicy):
    """
    A policy that simply passes the input action through to the output.
    This is useful for doing RL to directly learn input actions to the dynamics model.
    The input space is the same as the output space, which is the action space of the dynamics model.

    """
    def __init__(
        self,
        robot,
        policy_config: Dict = None,
    ):
        super().__init__(
            robot,
            policy_config=policy_config,
            default_config={},
        )

    def step(
        self,
        input_action: Any,
        global_goal: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        return input_action

    def _init_input_action_space(self) -> Space:
        return self.output_action_space