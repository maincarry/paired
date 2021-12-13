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
# from verifai.samplers.scenic_sampler import ScenicSampler
# from verifai.scenic_server import ScenicServer
# from verifai.falsifier import generic_falsifier
import os
from scenic.core.vectors import Vector
import math
# from verifai.monitor import specification_monitor, mtl_specification
from scenic.simulators.gfootball.utilities.scenic_helper import buildScenario
from scenic.simulators.gfootball.samplableVarExtraction import *

import gfootball
from gfootball.env import football_action_set


class AdversarialEnv(scenicenv.GFEnv):
  """Grid world where an adversary build the environment the agent plays.

  The adversary places the goal, agent, and up to n_clutter blocks in sequence.
  The action dimension is the number of squares in the grid, and each action
  chooses where the next item should be placed.
  """

  def __init__(self, initial_scenario, gf_env_settings, allow_render = False, rank = 0, num_adv_vars = 2):
    """Initializes environment in which adversary places goal, agent, obstacles.
    """

    # breakpoint()
    self.scenario = buildScenario(initial_scenario)

    super().__init__(
      self.scenario, gf_env_settings, allow_render, rank
    )
    print(f"Rank: {rank}")

    # parse samplable parameters
    self.samplableVars = parseSamplableVars(self.scenario)
    # parse the ranges of each samplable parameter: self.varRanges = [(low1, high1), (low2, high2), ...]
    self.varRanges = parseVar(self.samplableVars)
    # self.low = [low1, low2, ...], self.high = [high1, high2, ...]
    self.low, self.high = low_high_ranges(self.varRanges)

    self.adversary_action_dim = len(self.varRanges)
    self.adversary_action_space = gym.spaces.Box(low=np.array(self.low), high=np.array(self.high), dtype=np.float32)
    self.adversary_observation_space = gym.spaces.Box(low=0, high=255, shape=(72, 96, 16), dtype='uint8')

    print("Adv Env sample vars: ", self.samplableVars)
    print("Adv Env var lows: ", self.low)
    print("Adv Env var highs: ", self.high)
    print("Adv Env action space: ", self.adversary_action_space)

    # A flag showing scenario has been constructed.
    self.finish_building_scene = False
    self.last_obs = None



  # this function returns an adv env obs
  def reset(self):
    """Fully resets the environment.
       Return an obs for adv env agent. This part is done. no need to add 
       """
    obs = super().reset()
    self.last_obs = obs
    unconditionScenario(self.scenario)
    return obs

  def reset_agent(self):
    """Resets the agent's start position, but leaves goal and walls."""
    obs = super().reset()
    self.last_obs = obs
    return obs

  def step_adversary(self, action):
    ''' 
    action := sampled parameters from the Scenic program
    step_adversary() function should 
    (1) scale the agent model outputs (i.e. sampled parameters) to proper ranges
    (2) input the sampled parameters to the scenic program by "conditioning" samplable variables in scenario object
    (3) run reset() to validate whether sampled parameters are valid
        if valid, done = True, otherwise done = False. 
    
    For now, we must manually make sure sampled parameters are valid.
    '''
    assert isinstance(action, np.ndarray)
    assert len(action) == self.adversary_action_dim, f"Action Shape: {action.shape}, Exp: {self.adversary_action_dim}"
    # assert self.last_obs, "last obs must always be valid"

    # (1) Assuming that the action's elements are all within [0,1]
    scaled_sampled_params = self.scale_sampled_params(action)
    # print(f"Adv Env: Sampled Scene: {action=} {scaled_sampled_params=}")

    # (2) input the sampled parameters to the scenic program by "conditioning" samplable variables in scenario object
    # inputDict:  key = samplable variable objects within 'scenario' object, value = sampled value of the object
    inputDict = createInputDictionary(self.samplableVars, scaled_sampled_params)
    # condition the sampled values to corresponding samplable variables in 'scenario' object
    inputVarToScenario(self.scenario, inputDict)

    # (3) run reset() to validate whether sampled parameters are valid
    info = {}
    obs = self.check_validity()
    
    if obs is None:
      # print(f"Encountered Invalid Scene: {action=} {scaled_sampled_params=}")

      # append sampled value to the info
      done = False
      
      while obs is None:
          # randomly re-sample parameters again
          # print("Resample start.")
          unconditionScenario(self.scenario)
          low_bound, high_bound = -1, 1
          resampled_action = [random.uniform(low_bound, high_bound) for i in range(len(self.varRanges))]
          scaled_sampled_params = self.scale_sampled_params(resampled_action)
          inputDict = createInputDictionary(self.samplableVars, scaled_sampled_params)
          # condition the sampled values to corresponding samplable variables in 'scenario' object
          inputVarToScenario(self.scenario, inputDict)

          # print(f"{self.rank=} Resample. Now checking validity.")
          obs = self.check_validity()
          # print(f"{self.rank=} Resample end.")

          if obs is not None:
            info['resampled_params'] = resampled_action
            # print(f"{self.rank=} Resampled Parameters.")
            # print(f"{self.rank=} Remedy: use random params. {action=} {resampled_action=}")
    else: 
      done = True

    return obs, 0, done, info

  def scale_sampled_params(self, action):
    ''' scale the action elements '''
    scaled_sampled_params = []

    for index, param in enumerate(action):
      scaled_param = ((param+1)/2) * (self.high[index] - self.low[index]) + self.low[index]
      scaled_sampled_params.append(scaled_param)

    return scaled_sampled_params

  def reset_random(self):
    """
    uncondition scenario object and then run self.reset()
    """
    unconditionScenario(self.scenario)
    return self.reset()

  def check_validity(self):
    try:
        self.scene, _ = self.scenario.generate(maxIterations=1)
        if self.scene is None:
            return None

        if hasattr(self, "simulation"): self.simulation.get_underlying_gym_env().close()

        from scenic.simulators.gfootball.simulator import GFootBallSimulation
        self.simulation = GFootBallSimulation(scene=self.scene, settings={}, for_gym_env=True,
                                              render=self.allow_render, verbosity=1,
                                              env_type="v2",
                                              gf_env_settings=self.gf_env_settings,
                                              tag=str(self.rank))

        self.gf_gym_env = self.simulation.get_underlying_gym_env()

        obs = self.simulation.reset()
        player_idx = self.simulation.get_controlled_player_idx()[0]

        self.simulation.pre_step()

        return obs[player_idx]

    except Exception as e:
        print("Resample Script. Cause Error: ", e)
        return None

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
    scenario_file = "/home/qcwu/gf/scenic4rl/training/gfrl/_scenarios/grf/pass_n_shoot.scenic"

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
    scenario_file = "/home/qcwu/gf/scenic4rl/training/gfrl/_scenarios/offense/avoid_pass_shoot.scenic"

    super().__init__(scenario_file, gf_env_settings, allow_render = False, rank=iprocess, num_adv_vars = 2)

class EasyCrossingAdversarialEnv(AdversarialEnv):
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
    # TODO: change path
    scenario_file = "/home/qcwu/gf/scenic4rl/training/gfrl/_scenarios/offense/easy_crossing.scenic"

    super().__init__(scenario_file, gf_env_settings, allow_render = False, rank=iprocess, num_adv_vars = 2)

class DefenseGoalKeeperOppoentEnv(AdversarialEnv):
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
    scenario_file = "/home/qcwu/gf/scenic4rl/training/gfrl/_scenarios/defense/goalkeeper_vs_opponent.scenic"

    super().__init__(scenario_file, gf_env_settings, allow_render = False, rank=iprocess, num_adv_vars = 2)


class DefenseTwoVTwoEnv(AdversarialEnv):
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
    scenario_file = "/home/qcwu/gf/scenic4rl/training/gfrl/_scenarios/defense/2vs2.scenic"

    super().__init__(scenario_file, gf_env_settings, allow_render = False, rank=iprocess, num_adv_vars = 2)


class Defense2V2HighPassForwardEnv(AdversarialEnv):
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
    scenario_file = "/home/qcwu/gf/scenic4rl/training/gfrl/_scenarios/defense/2vs2_with_scenic_high_pass_forward.scenic"

    super().__init__(scenario_file, gf_env_settings, allow_render = False, rank=iprocess, num_adv_vars = 2)

class Defense3V3CrossFromSideEnv(AdversarialEnv):
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
    scenario_file = "/home/qcwu/gf/scenic4rl/training/gfrl/_scenarios/defense/3vs3_cross_from_side.scenic"

    super().__init__(scenario_file, gf_env_settings, allow_render = False, rank=iprocess, num_adv_vars = 2)

class DefenseDefenderOpponentZigzagEnv(AdversarialEnv):
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
    scenario_file = "/home/qcwu/gf/scenic4rl/training/gfrl/_scenarios/defense/defender_vs_opponent_with_zigzag_dribble.scenic"

    super().__init__(scenario_file, gf_env_settings, allow_render = False, rank=iprocess, num_adv_vars = 2)

class Paired1v1Env(AdversarialEnv):
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
    scenario_file = "/home/qcwu/gf/paired/scenic_scenarios/1v1.scenic"

    super().__init__(scenario_file, gf_env_settings, allow_render = False, rank=iprocess, num_adv_vars = 2)

class Paired1v1Test0Env(AdversarialEnv):
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
    scenario_file = "/home/qcwu/gf/paired/scenic_scenarios/1v1_test0.scenic"

    super().__init__(scenario_file, gf_env_settings, allow_render = False, rank=iprocess, num_adv_vars = 2)

class Paired1v1Test1Env(AdversarialEnv):
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
    scenario_file = "/home/qcwu/gf/paired/scenic_scenarios/1v1_test1.scenic"

    super().__init__(scenario_file, gf_env_settings, allow_render = False, rank=iprocess, num_adv_vars = 2)

paired_gf_env_settings = {
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
class PairedChoosePassingEnv(AdversarialEnv):
  def __init__(self, iprocess, **kwargs):
    scenario_file = "/home/qcwu/gf/paired/scenic_scenarios/choose_passing.scenic"
    super().__init__(scenario_file, paired_gf_env_settings, allow_render = False, rank=iprocess, num_adv_vars = 2)

class PairedChoosePassingTest0Env(AdversarialEnv):
  def __init__(self, iprocess, **kwargs):
    scenario_file = "/home/qcwu/gf/paired/scenic_scenarios/choose_passing_test0.scenic"
    super().__init__(scenario_file, paired_gf_env_settings, allow_render = False, rank=iprocess, num_adv_vars = 2)

class PairedChoosePassingTest1Env(AdversarialEnv):
  def __init__(self, iprocess, **kwargs):
    scenario_file = "/home/qcwu/gf/paired/scenic_scenarios/choose_passing_test1.scenic"
    super().__init__(scenario_file, paired_gf_env_settings, allow_render = False, rank=iprocess, num_adv_vars = 2)

class Paired3v2Env(AdversarialEnv):
  def __init__(self, iprocess, **kwargs):
    scenario_file = "/home/qcwu/gf/paired/scenic_scenarios/3v2.scenic"
    super().__init__(scenario_file, paired_gf_env_settings, allow_render = False, rank=iprocess, num_adv_vars = 2)



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

register.register(
    env_id='gfootball-EasyCrossing-v0',
    entry_point=module_path + ':EasyCrossingAdversarialEnv',
)

register.register(
    env_id='gfootball-DefenseGoalKeeperOppoent-v0',
    entry_point=module_path + ':DefenseGoalKeeperOppoentEnv',
)

register.register(
    env_id='gfootball-DefenseTwoVTwo-v0',
    entry_point=module_path + ':DefenseTwoVTwoEnv',
)

register.register(
    env_id='gfootball-Defense2V2HighPassForward-v0',
    entry_point=module_path + ':Defense2V2HighPassForwardEnv',
)

register.register(
    env_id='gfootball-Defense3V3CrossFromSide-v0',
    entry_point=module_path + ':Defense3V3CrossFromSideEnv',
)

register.register(
    env_id='gfootball-DefenseDefenderOpponentZigzag-v0',
    entry_point=module_path + ':DefenseDefenderOpponentZigzagEnv',
)

register.register(
    env_id='gfootball-Paired1v1-v0',
    entry_point=module_path + ':Paired1v1Env',
)

register.register(
    env_id='gfootball-Paired1v1Test0-v0',
    entry_point=module_path + ':Paired1v1Test0Env',
)

register.register(
    env_id='gfootball-Paired1v1Test1-v0',
    entry_point=module_path + ':Paired1v1Test1Env',
)

register.register(
    env_id='gfootball-PairedChoosePassing-v0',
    entry_point=module_path + ':PairedChoosePassingEnv',
)

register.register(
    env_id='gfootball-PairedChoosePassingTest0-v0',
    entry_point=module_path + ':PairedChoosePassingTest0Env',
)

register.register(
    env_id='gfootball-PairedChoosePassingTest1-v0',
    entry_point=module_path + ':PairedChoosePassingTest1Env',
)

register.register(
    env_id='gfootball-Paired3v2-v0',
    entry_point=module_path + ':Paired3v2Env',
)



