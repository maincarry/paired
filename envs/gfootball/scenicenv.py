from scenic.simulators.gfootball.rl.gfScenicEnv_v2 import GFScenicEnv_v2
from scenic.simulators.gfootball.utilities.scenic_helper import buildScenario

import torch
import numpy as np

class GFEnv(GFScenicEnv_v2):
  """Wrapper for GFScenicEnv_v2"""

  def __init__(
      self, initial_scenario_file, gf_env_settings, allow_render = False, rank = 0
  ):
    """Constructor for GFScenicEnv_v2.
    """
    self.scenario = buildScenario(initial_scenario_file)
    # print(f"{rank=}, {scenario=}")
    super().__init__(scenario, gf_env_settings, allow_render, rank)

    # Initialize the state
    self.first_obs = self.reset()

  def step(self, actions):
      if isinstance(actions, torch.Tensor):
          actions = actions.item()
      obs, rew, done, info = super().step(actions)

      # TODO need to reduce the reward by time
      # print(rew)
      return obs, rew, done, info
