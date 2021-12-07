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
from scenic.simulators.gfootball.utilities.scenic_helper import buildScenario

class AdversarialEnv(scenicenv.GFEnv):
  """Grid world where an adversary build the environment the agent plays.

  The adversary places the goal, agent, and up to n_clutter blocks in sequence.
  The action dimension is the number of squares in the grid, and each action
  chooses where the next item should be placed.
  """

  def __init__(self, initial_scenario, gf_env_settings, allow_render = False, rank = 0, num_adv_vars = 2):
    """Initializes environment in which adversary places goal, agent, obstacles.
    """

    self.scenario = buildScenario(initial_scenario_file)

    super().__init__(
      self.scenario, gf_env_settings, allow_render, rank
    )
    print(f"{rank=}")

    # parse samplable parameters
    self.samplableVars = parseSamplableVars(self.scenario)
    # parse the ranges of each samplable parameter: self.varRanges = [(low1, high1), (low2, high2), ...]
    self.varRanges = parseVar(samplableVars)
    # self.low = [low1, low2, ...], self.high = [high1, high2, ...]
    self.low, self.high = low_high_ranges(self.varRanges)

    self.adversary_action_dim = len(self.samplableVars)
    self.adversary_action_space = gym.spaces.Box(low=self.low, high=self.high, dtype=np.float32)
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
       Return an obs for adv env agent. This part is done. no need to add 
       """
    obs = super().reset()
    return obs

  def reset_agent(self):
    """Resets the agent's start position, but leaves goal and walls."""
    obs = super().reset()
    return obs

  def step_adversary(self, action):
    ''' 
    action := sampled parameters from the Scenic program
    step_adversary() function should 
    (1) scale the agent model outputs (i.e. sampled parameters) to proper ranges
    (2) input the sampled parameters to the scenic program by "conditioning" samplable variables in scenario object
    (3) run reset() to validate whether sampled parameters are valid
        if valid, done = True, otherwise done = False. 
    '''
    assert isinstance(action, np.ndarray)
    assert len(action) == self.adversary_action_dim

    # (1) Assuming that the action's elements are all within [0,1]
    scaled_sampled_params = []
    for index in range(len(action)):
      scaled_param = param * (self.high[index] - self.low[index]) + self.low[index]
      scaled_sampled_params.append(scaled_param)

    # (2) input the sampled parameters to the scenic program by "conditioning" samplable variables in scenario object
    # inputDict:  key = samplable variable objects within 'scenario' object, value = sampled value of the object
    inputDict = createInputDictionary(self.samplableVars, scaled_sampled_params)
    # condition the sampled values to corresponding samplable variables in 'scenario' object
    inputVarToScenario(self.scenario, inputDict)

    # (3) run reset() to validate whether sampled parameters are valid
    obs = self.reset()
    
    if obs is None:
      done = False
    else: 
      done = True

    return obs, 0, done, {}

  def reset_random(self):
    """
    uncondition scenario object and then run self.reset()
    """
    unconditionScenario(self.scenario)
    return self.reset()

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


class AvoidPassShootAdversarialEnv(AdversarialEnv):
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
    scenario_file = "/home/curriculum_learning/rl/scenic4rl/training/gfrl/_scenarios/offense/avoid_pass_shoot.scenic"

    super().__init__(scenario_file, gf_env_settings, allow_render = False, rank=iprocess, num_adv_vars = 2)


if hasattr(__loader__, 'name'):
  module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
  module_path = __loader__.fullname

register.register(
    env_id='gfootball-MiniAdversarial-v0',
    entry_point=module_path + ':MiniAdversarialEnv',
)

register.register(
    env_id='gfootball-AvoidPassShootAdversarial-v0',
    entry_point=module_path + ':AvoidPassShootAdversarialEnv',
)
