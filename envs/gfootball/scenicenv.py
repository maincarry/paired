from scenic.simulators.gfootball.rl.gfScenicEnv_v2 import GFScenicEnv_v2
from scenic.simulators.gfootball.utilities.scenic_helper import buildScenario

import torch
import numpy as np

class GFEnv(GFScenicEnv_v2):
  """Wrapper for GFScenicEnv_v2"""

  def __init__(
      self, scenario, gf_env_settings, allow_render = False, rank = 0
  ):
    """Constructor for GFScenicEnv_v2.
    """

    # print(f"{rank=}, {scenario=}")
    super().__init__(scenario, gf_env_settings, allow_render, rank)

    # # parse samplable parameters
    # self.samplableVars = parseSamplableVars(scenario)
    # self.varRanges = parseVar(samplableVars)
    # self.low, self.high = low_high_ranges(self.varRanges)

    # Initialize the state
    # self.first_obs = self.reset()

  def step(self, actions):
    # action = sampled env parameters
    if isinstance(actions, torch.Tensor):
        actions = actions.item()
    obs, rew, done, info = super().step(actions)

    # TODO need to reduce the reward by time

    
    return obs, rew, done, info
