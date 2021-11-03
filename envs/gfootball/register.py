"""Register gfootball environments with OpenAI gym."""

import gym
from envs.registration import register as gym_register

env_list = []

# TODO: change this
def register(env_id, entry_point):
  """Register a new environment with OpenAI gym based on id."""
  assert env_id.startswith("gfootball-")
  if env_id in env_list:
    del gym.envs.registry.env_specs[id]
  else:
    # Add the environment to the set
    env_list.append(id)

  # Register the environment with OpenAI gym
  # TODO: add parameters here
  gym_register(
      id=env_id, entry_point=entry_point)
