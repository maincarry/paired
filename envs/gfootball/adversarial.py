"""An environment which is built by a learning adversary.

Has additional functions, step_adversary, and reset_agent. How to use:
1. Call reset() to reset to an empty environment
2. Call step_adversary() to place the goal, agent, and obstacles. Repeat until
   a done is received.
3. Normal RL loop. Use learning agent to generate actions and use them to call
   step() until a done is received.
4. If required, call reset_agent() to reset the environment the way the
   adversary designed it. A new agent can now play it using the step() function.
"""
import random

import gym
import numpy as np
import torch

from . import scenicenv
from . import register

from dotmap import DotMap
from verifai.samplers.scenic_sampler import ScenicSampler
from verifai.scenic_server import ScenicServer
from verifai.falsifier import generic_falsifier
import os
from scenic.core.vectors import Vector
import math
from verifai.monitor import specification_monitor, mtl_specification


class AdversarialEnv(scenicenv.GFEnv):
  """Grid world where an adversary build the environment the agent plays.

  The adversary places the goal, agent, and up to n_clutter blocks in sequence.
  The action dimension is the number of squares in the grid, and each action
  chooses where the next item should be placed.
  """

  def __init__(self, initial_scenario, gf_env_settings, allow_render = False, rank = 0, num_adv_vars = 2):
    """Initializes environment in which adversary places goal, agent, obstacles.
    """

    super().__init__(
      initial_scenario, gf_env_settings, allow_render, rank
    )
    print(f"{rank=}")

    self.step_count = 0

    # TODO: Create spaces for adversary agent's specs. For now, we only have 2.
    self.adversary_action_dim = num_adv_vars
    self.adversary_action_space = gym.spaces.Discrete(self.adversary_action_dim)
    self.adversary_observation_space = gym.spaces.Box(low=0, high=255, shape=(72, 96, 16), dtype='uint8')

    # Eddie: instantiate VerifAI with the initial_scenario path
    # sampler = ScenicSampler.fromScenario(initial_scenario)

    # falsifier_params = DotMap(
    #     n_iters=5,
    #     save_error_table=True,
    #     save_safe_table=True,
    #     error_table_path='error_table.csv',
    #     safe_table_path='safe_table.csv'
    # )
    # server_options = DotMap(maxSteps=100, verbosity=0)
    # self.falsifier = generic_falsifier(sampler=sampler,
    #                           # monitor = MyMonitor(),
    #                           falsifier_params=falsifier_params,
    #                           server_class=ScenicServer,
    #                           server_options=server_options)
    # self.scene = None


  # this function returns an adv env obs
  def reset(self):
    """Fully resets the environment.
       Return an obs for adv env agent."""
    self.step_count = 0

    # Very important, we need obs for adv env! Here I just use agent obs.
    # obs, self.scene = super().ACL_reset(self.falsifier)
    obs = super().reset()
    return obs

  def reset_agent(self):
    """Resets the agent's start position, but leaves goal and walls."""

    # Step count since episode start
    self.step_count = 0

    # Return first observation
    # obs = super().ACL_resetAgent(self.scene)
    obs = super().reset()

    return obs

  # def reset_to_level(self, level):
  #   self.reset()
  #   actions = [int(a) for a in level.split()]
  #   for a in actions:
  #     obs, _, done, _ = self.step_adversary(a)
  #     if done:
  #       return self.reset_agent()

  def step_adversary(self, loc):
    """The adversary gets n_clutter + 2 moves to place the goal, agent, blocks.

    The action space is the number of possible squares in the grid. The squares
    are numbered from left to right, top to bottom.

    Args:
      loc: An integer specifying the location to place the next object which
        must be decoded into x, y coordinates.

    Returns:
      Standard RL observation, reward (always 0), done, and info
    """
    obs = self.first_obs

    return obs, 0, True, {}

  def reset_random(self):
    """
    Note, this is based on the original PAIRED implementation from 
    https://github.com/google-research/google-research/blob/master/social_rl/gym_multigrid/envs/adversarial.py,
    which sets the domain randomization baseline to use n_clutter/2 blocks.
    """
    return self.reset_agent()

  # TODO: required but what is this for???
  @property
  def processed_action_dim(self):
    return 1


class MiniAdversarialEnv(AdversarialEnv):
  def __init__(self, iprocess, **kwargs):
    gf_env_settings = {
        "stacked": True,
        "rewards": "scoring",
        "representation": 'extracted',
        "players": [f"agent:left_players=1"],
        "real_time": False,
        "action_set": "default",
        "dump_full_episodes": False,
        "dump_scores": False,
        "write_video": False,
        "tracesdir": "dummy",
        "write_full_episode_dumps": False,
        "write_goal_dumps": False,
        "render": False
    }
    scenario_file = "/home/curriculum_learning/rl/scenic4rl/training/gfrl/_scenarios/grf/pass_n_shoot.scenic"

    super().__init__(scenario_file, gf_env_settings, allow_render = False, rank=iprocess, num_adv_vars = 2)


if hasattr(__loader__, 'name'):
  module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
  module_path = __loader__.fullname

register.register(
    env_id='gfootball-MiniAdversarial-v0',
    entry_point=module_path + ':MiniAdversarialEnv',
)

