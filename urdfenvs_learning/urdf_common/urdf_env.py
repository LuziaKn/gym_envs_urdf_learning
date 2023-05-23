import gym
import time
import numpy as np
import pybullet as p
import warnings
import logging
from typing import List, Union, Optional

from mpscenes.obstacles.collision_obstacle import CollisionObstacle
from mpscenes.goals.goal_composition import GoalComposition
from mpscenes.goals.sub_goal import SubGoal

from urdfenvs.urdf_common.plane import Plane
from urdfenvs.sensors.sensor import Sensor
from urdfenvs.urdf_common.generic_robot import GenericRobot
from urdfenvs.urdf_common.reward import Reward
from urdfenvs_learning.policies.feed_through_policy import FeedThroughPolicy
from urdfenvs.urdf_common.urdf_env import UrdfEnv








class UrdfEnv_learning(UrdfEnv):
    """Generic urdf-environment for OpenAI-Gym"""

    def __init__(
        self,
        robots: List[GenericRobot],
        render: bool = False,
        dt: float = 0.01,
        observation_checking=True,
    ) -> None:

        super().__init__(robots = robots,
                         render = render,
                         dt = dt,
                         observation_checking = observation_checking)


        for robot in self._robots:
            robot._policy = FeedThroughPolicy(robot)




    def set_spaces(self) -> None:
        """Set observation and action space."""
        observation_space_as_dict = {}
        action_space_as_dict = {}

        for i, robot in enumerate(self._robots):
            (obs_space_robot_i, action_space_robot_i) = robot.get_spaces()
            obs_space_robot_i = dict(obs_space_robot_i)
            for sensor in robot._sensors:
                
                self.sensors.append(sensor) # Add the sensor to the list of sensors.

                obs_space_robot_i.update(
                    sensor.get_observation_space(self._obsts, self._goals)
                )
            observation_space_as_dict[f"robot_{i}"] = gym.spaces.Dict(
                obs_space_robot_i
            )
            action_space_as_dict[f"robot_{i}"] = action_space_robot_i

        self.observation_space = gym.spaces.Dict(observation_space_as_dict)
        action_space = gym.spaces.Dict(action_space_as_dict)
        self.action_space = gym.spaces.flatten_space(action_space)

    def step(self, action):
        self._t += self.dt()

        # compute action of policy
        for robot in self._robots:
            actions = robot._policy.step(action)

        # Feed action to the robot and get observation of robot's state

        if not self.action_space.contains(action):
            self._done = True
            self._info = {'action_limits': f"{action} not in {self.action_space}"}


        action_id = 0
        for robot in self._robots:
            action_robot = action[action_id : action_id + robot.n()]
            robot.apply_action(action_robot, self.dt())
            action_id += robot.n()

        self.update_obstacles()
        self.update_goals()
        self.update_collision_links()
        p.stepSimulation(self._cid)
        ob = self._get_ob()

        # Calculate the reward.
        # If there is no reward object, then the reward is 1.0.
        if self._reward_calculator is not None:
            reward = self._reward_calculator.calculateReward(ob) 
        else:
            reward = 1.0
        
        if self._render:
            self.render()
        return ob, reward, self._done, self._info


