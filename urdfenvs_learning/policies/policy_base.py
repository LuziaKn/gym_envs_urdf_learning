from __future__ import annotations
from typing import Any, Dict, Optional, Type, Union, List, TYPE_CHECKING
from abc import ABC, abstractmethod
from gym.spaces import Space, Box, Discrete, MultiDiscrete, MultiBinary
import numpy as np




class BasePolicy(ABC):
    """
    The base class for all policies.

    Policies determine the behavior of an agent by generating actions based on the agent's state and sensor observations.
    The actions are then fed into the agent's dynamics model to update the agent's state, or guide another policy in
    a hierarchical policy.

    Each policy has an input action space and an output action space. The input action space is used to determine the
    action space of the environment for reinforcement learning, and must be defined in any subclass.
    The output action space is by default set to the action space of the agent's dynamics model, but can be overridden
    by a hierarchical policy.

    The policy (default) config is given as a dictionary. The default config is a dictionary defined in the policy file
    and must be passed to the parent constructor. The policy config can be externally defined and passed to the
    policy (subclass) constructor. The policy config is merged with the default config, with the policy config taking
    precedence.

    Parameters
    ----------
    host_agent : Agent
        The agent that this policy is controlling.
    policy_config : Optional[Dict], optional
        The policy config, by default None
    default_config : Optional[Dict], optional
        The default policy config, by default None
    is_static : bool, optional
        Whether the policy is static, by default False. A static policy does not require external guidance and does not
        return an action.
    is_guided : bool, optional
        Whether the policy is guided, by default True. A guided policy requires external guidance such as through
        another policy in a hierarchical policy, or through a policy network in reinforcement learning
        (through the action parameter of the step function).
    """
    def __init__(
        self,
        robot,
        policy_config: Optional[Dict] = None,
        default_config: Optional[Dict] = None,
        is_static: bool = False,
        is_guided: bool = True,
    ):

        self.robot = robot
        # Init config
        self.config = self._init_config(default_config, policy_config)

        # Init properties
        self.is_static = is_static
        if not is_static:
            self.is_guided = is_guided  # This means it requires external guidance such as through a policy
        else:
            if is_guided:
                raise ValueError(
                    "A static policy cannot be guided. Set is_guided to False."
                )
            self.is_guided = False

        # Init input and output action spaces
        self._output_action_space = self._init_output_action_space()
        self._input_action_space = self._init_input_action_space()

        # Assert that the output action space is valid for the dynamics model
        if not is_static and self._output_action_space is None:
            raise ValueError(
                "The host agent's dynamics model has an empty action space. For a static agent, use a static policy."
            )
        elif is_static and self._output_action_space is not None:
            raise ValueError(
                "The host agent's dynamics model has a non-empty action space. Use a non-static policy."
            )

        # TODO setup this property in the agent
        self.required_sensors = []

    @abstractmethod
    def step(
            self,
            input_action: Any,
            global_goal: np.ndarray,
            **kwargs: Any,
    ) -> np.ndarray:
        """
        The step function of the policy generates an action based on the agent's state and sensor observations.
        The agent's state and sensor observations are available through the self.agent attribute.
        Any additional arguments can be passed through the kwargs parameter.

        Parameters
        ----------
        input_action : Any
            The input action to the policy. This is the action that the policy is guided by.
        global_goal : np.ndarray
            The global goal of the agent.
        **kwargs : Any
            Any additional arguments.

        Returns
        -------
        np.ndarray
            The output action of the policy.

        """
        pass

    @abstractmethod
    def _init_input_action_space(self) -> Space:
        """
        The input action space is used to determine the action space of the environment for reinforcement learning.

        Returns
        -------
        Space
            The input action space as a OpenAI gym space.
        """
        pass

    def _init_output_action_space(self) -> Space:
        """
        The output action space is by default set to the action space of the agent's dynamics model, but can be
        overridden in a hierarchical policy by the input action space of the subsequent policy.

        Returns
        -------
        Space
            The output action space as a OpenAI gym space.
        """
        (obs_space_robot_i, action_space_robot_i) = self.robot.get_spaces()


        return action_space_robot_i

    @staticmethod
    def _init_config(default_config: Dict, policy_config: Dict) -> Dict:
        # Init config
        if default_config is None:
            default_config = {}
        config = default_config.copy()
        if policy_config is not None:
            config.update(policy_config)
        return config

    @property
    def input_action_space(self) -> Space:
        return self._input_action_space

    @property
    def output_action_space(self) -> Space:
        return self._output_action_space

    @output_action_space.setter
    def output_action_space(self, new_output_action_space: Space) -> None:
        if isinstance(new_output_action_space, Space):
            self._output_action_space = new_output_action_space
        else:
            raise ValueError(
                "The output action space must be an instance of gym.spaces.Space"
            )

    @input_action_space.setter
    def input_action_space(self, new_input_action_space: Space) -> None:
        if isinstance(new_input_action_space, Space):
            self._input_action_space = new_input_action_space
        else:
            raise ValueError(
                "The input action space must be an instance of gym.spaces.Space"
            )