from gym.envs.registration import register
from urdfenvs_learning.urdf_common.urdf_env import UrdfEnv_learning
register(
    id='urdf-env-learning-v0',
    entry_point='urdfenvs_learning.urdf_common:UrdfEnv_learning'
)
