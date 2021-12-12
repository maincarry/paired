from scenic.simulators.gfootball.model import *
from scenic.simulators.gfootball.simulator import GFootBallSimulator

param game_duration = 400
param deterministic = False
param offsides = False
param end_episode_on_score = True
param end_episode_on_out_of_play = True
param end_episode_on_possession_change = True

p1_spawn_point = 58 @ 20
o0_spawn_point = 80 @ -20

ego = LeftGK
p1 = LeftPlayer with role "CM", at p1_spawn_point
o0 = RightGK at o0_spawn_point

ball = Ball ahead of p1 by 2