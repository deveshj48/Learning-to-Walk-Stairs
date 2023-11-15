import gym
import os
import imageio

from environment import * # this contains the model

env = gym.make('CustomAnt-v0', render_mode='human') 


env.reset()
for _ in range(300):
  env.render()
  
  env.step(env.action_space.sample())
env.close() 
