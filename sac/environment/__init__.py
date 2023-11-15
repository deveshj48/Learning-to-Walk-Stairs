__all__ = ["custom_ant"]

# from gym.envs.mujoco.mujoco_env import MujocoEnv, MuJocoPyEnv  # isort:skip

# from environment.custom_ant import AntEnv

# import gymnasium as gym
from gym.envs.registration import make, register, registry, spec


register(
    id="CustomAnt-v0",
    entry_point="environment.custom_ant:AntEnv",
    max_episode_steps=10000,
    kwargs={},
)