
from scenic.simulators.gfootball.model import *
from scenic.simulators.gfootball.simulator import GFootBallSimulator

param game_duration = 400
param deterministic = False
param offsides = False
param end_episode_on_score = True
param end_episode_on_out_of_play = True
param end_episode_on_possession_change = True

p1_spawn = get_reg_from_edges(40, 60, -20, 20)
o0_spawn = get_reg_from_edges(70, 80, -25, 25)

ego = LeftGK
p1 = LeftPlayer with role "CM", on p1_spawn
o0 = RightGK on o0_spawn

ball = Ball ahead of p1 by 2