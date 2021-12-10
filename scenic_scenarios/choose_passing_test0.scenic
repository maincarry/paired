from scenic.simulators.gfootball.model import *
from scenic.simulators.gfootball.simulator import GFootBallSimulator

param game_duration = 400
param deterministic = False
param offsides = False
param end_episode_on_score = True
param end_episode_on_out_of_play = True
param end_episode_on_possession_change = True


p1_spawn_point = 0 @ 0
p2_spawn_point = 55 @ -30
require (abs(p2_spawn_point.y) > 20)

p3_spawn_point = p2_spawn_point.x @ -p2_spawn_point.y

ego = LeftGK

p1 = LeftPlayer with role "CM", at p1_spawn_point
p2 = LeftPlayer with role "CF", at p2_spawn_point
p3 = LeftPlayer with role "CF", at p3_spawn_point

# front blocker
o2_spawn_point = 50 @ 0
# p2 blocker
o1_spawn_point = Point right of p2 by 2

o1 = RightCB at o1_spawn_point
o2 = RightGK at o2_spawn_point


ball = Ball ahead of p1 by 2