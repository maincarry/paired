import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym.spaces import MultiDiscrete

from .distributions import Categorical  
from .common import *


class GFootballNetwork(DeviceAwareModule):
    """
    Actor-Critic module 
    """
    def __init__(self, 
        observation_space, 
        action_space, 
        actor_fc_layers=(32,32),
        value_fc_layers=(32,32),
        random=False):        
        super().__init__()

        self.random = random
        self.recurrent_hidden_state_size = 1  # required but not used

        self.num_actions = action_space.n
        self.multi_dim = False
        self.action_dim = 1
        self.num_action_logits = self.num_actions

        self.action_space = action_space
        self.observation_space = observation_space
        obs_shape = np.prod(observation_space.shape)

        # Policy head
        self.actor = nn.Sequential(
            nn.Flatten(),
            make_fc_layers_with_hidden_sizes(actor_fc_layers, input_size=obs_shape),
            Categorical(actor_fc_layers[-1], self.num_actions)
        )

        # Value head
        self.critic = nn.Sequential(
            nn.Flatten(),
            make_fc_layers_with_hidden_sizes(value_fc_layers, input_size=obs_shape),
            nn.Linear(value_fc_layers[-1], 1)
        )

        # apply_init_(self.modules())
        # self.train()


    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        if self.random:
            print("Random not implemented.")

        dist = self.actor(inputs)
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()
        action_log_dist = dist.logits

        value = self.critic(inputs)

        # print(type(action))

        return value, action, action_log_dist, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        return self.critic(inputs)

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        dist = self.actor(inputs)
        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        value = self.critic(inputs)
        return value, action_log_probs, dist_entropy, rnn_hxs

    @property
    def is_recurrent(self):
        return False



## this is a custom network for the adversary, it is like the multigrid network as it uses the map
class GFootballAdversaryNetwork(DeviceAwareModule):
    def __init__(self,
        observation_space,
        action_space,
        actor_fc_layers=(32,32),
        value_fc_layers=(32,32),
        random=False):
        super().__init__()

        self.random = random
        self.recurrent_hidden_state_size = 1  # required but not used

        self.num_actions = action_space.n
        self.multi_dim = False
        self.action_dim = 1
        self.num_action_logits = self.num_actions

        self.action_space = action_space
        self.observation_space = observation_space
        obs_shape = np.prod(observation_space.shape)

        # Policy head
        self.actor = nn.Sequential(
            nn.Flatten(),
            make_fc_layers_with_hidden_sizes(actor_fc_layers, input_size=obs_shape),
            Categorical(actor_fc_layers[-1], self.num_actions)
        )

        # Value head
        self.critic = nn.Sequential(
            nn.Flatten(),
            make_fc_layers_with_hidden_sizes(value_fc_layers, input_size=obs_shape),
            nn.Linear(value_fc_layers[-1], 1)
        )

        # apply_init_(self.modules())
        # self.train()


    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        if self.random:
            print("Random not implemented.")

        dist = self.actor(inputs)
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()
        action_log_dist = dist.logits

        value = self.critic(inputs)

        return value, action, action_log_dist, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        return self.critic(inputs)

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        dist = self.actor(inputs)
        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        value = self.critic(inputs)
        return value, action_log_probs, dist_entropy, rnn_hxs

    @property
    def is_recurrent(self):
        return False
