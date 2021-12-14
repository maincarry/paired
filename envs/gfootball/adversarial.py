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
    self.scenario0 = buildScenario("/home/qcwu/gf/scenic4rl/training/gfrl/_scenarios/offense/test_away_away.scenic")
    self.scenario1 = buildScenario("/home/qcwu/gf/scenic4rl/training/gfrl/_scenarios/offense/test_ret_away.scenic")
    self.scenario2 = buildScenario("/home/qcwu/gf/scenic4rl/training/gfrl/_scenarios/offense/test_ret_ret.scenic")

    super().__init__(
      self.scenario, gf_env_settings, allow_render, rank
    )

    self.gf_env_settings = gf_env_settings
    self.allow_render = allow_render
    self.rank = rank
    self.num_adv_vars = 3

    # TODO: Create spaces for adversary agent's specs. For now, we only have 2.
    self.adversary_action_dim = self.num_adv_vars
    self.adversary_action_space = gym.spaces.Discrete(self.adversary_action_dim)
    self.adversary_observation_space = gym.spaces.Box(low=0, high=255, shape=(72, 96, 16), dtype='uint8')

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

  def step_adversary(self, action):
    scenario_name= ""
    if action == 0:
      # initial_scenario = "/home/qcwu/gf/scenic4rl/training/gfrl/_scenarios/offense/test_away_away.scenic"
      self.scenario = self.scenario0
      # scenario_name = "test_away_away"
    elif action == 1:
      # initial_scenario = "/home/qcwu/gf/scenic4rl/training/gfrl/_scenarios/offense/test_ret_away.scenic"
      self.scenario = self.scenario1
      # scenario_name = "test_ret_away"
    else:
      # initial_scenario = "/home/qcwu/gf/scenic4rl/training/gfrl/_scenarios/offense/test_ret_ret.scenic"
      self.scenario = self.scenario2
      # scenario_name = "test_ret_ret"

    # print("scenario_name: ", scenario_name)
    # self.scenario = buildScenario(initial_scenario)
    super().__init__(
      self.scenario, self.gf_env_settings, self.allow_render, self.rank
    )

    obs = self.reset()
    # obs = self.first_obs
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

class ScenarioComposition(AdversarialEnv):
  def __init__(self, iprocess, **kwargs):
    scenario_file = "/home/qcwu/gf/scenic4rl/training/gfrl/_scenarios/offense/test_away_away.scenic"
    super().__init__(scenario_file, paired_gf_env_settings, allow_render = False, rank=iprocess, num_adv_vars = 2)



if hasattr(__loader__, 'name'):
  module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
  module_path = __loader__.fullname

register.register(
    env_id='gfootball-scenarioComposition-v0',
    entry_point=module_path + ':ScenarioComposition',
)

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




