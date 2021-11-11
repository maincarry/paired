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

        self.device_name = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("device: ", self.device_name)

        self.random = random
        self.recurrent_hidden_state_size = 1  # required but not used

        self.num_actions = action_space.n
        self.multi_dim = False
        self.action_dim = 1
        self.num_action_logits = self.num_actions

        self.action_space = action_space
        self.observation_space = observation_space
        obs_shape = np.prod(observation_space.shape)

        # --- cnn ---
        self.features_dim = 256
        n_input_channels = observation_space.shape[0]

        self.conv_layers_config = [(16, 2), (32, 2), (32, 2), (32, 2)]
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv_blocks = [
            nn.Conv2d(in_channels=n_input_channels, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        ]

        #https://www.tensorflow.org/api_docs/python/tf/nn/pool  -> If padding = "SAME": output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides[i])
        self.pools = [nn.MaxPool2d(kernel_size=3, stride=2, padding=1) for _ in range(4)]

        self.resblocks_1 = [
            self.create_basic_res_block(16, 16),
            self.create_basic_res_block(32, 32),
            self.create_basic_res_block(32, 32),
            self.create_basic_res_block(32, 32)
        ]
        self.resblocks_2 = [
            self.create_basic_res_block(16, 16),
            self.create_basic_res_block(32, 32),
            self.create_basic_res_block(32, 32),
            self.create_basic_res_block(32, 32)
        ]

        if "cuda" in self.device_name.type:
            self.conv_blocks = [c.cuda() for c in self.conv_blocks]
            self.resblocks_1 = [c.cuda() for c in self.resblocks_1]
            self.resblocks_2 = [c.cuda() for c in self.resblocks_2]


        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        #print("flatten", self.conv_blocks[0].is_cuda)

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.feat_extract(
                torch.as_tensor(observation_space.sample()[None]).float().cuda()
            )
            n_flatten = n_flatten.shape[1]
        # TODO: make sure n_flatten = 192
        print(f"{n_flatten=}")
        self.linear = nn.Sequential(nn.Linear(n_flatten, self.features_dim), nn.ReLU()) # n_flatten=192
        # --- cnn ends ---


        # Policy head
        self.actor = nn.Sequential(
            make_fc_layers_with_hidden_sizes(actor_fc_layers, input_size=self.features_dim),
            Categorical(actor_fc_layers[-1], self.num_actions)
        )

        # Value head
        self.critic = nn.Sequential(
            make_fc_layers_with_hidden_sizes(value_fc_layers, input_size=self.features_dim),
            nn.Linear(value_fc_layers[-1], 1)
        )

        # apply_init_(self.modules())
        # self.train()

    def feat_extract(self, observations: torch.Tensor) -> torch.Tensor:
        observations = observations.float()
        observations /= 255

        conv_out = observations
        for i in range(4):
            #print("", i)
            #print(" 1. conv_out.is_cuda() ", conv_out.is_cuda)
            #print("     conv block weight", self.conv_blocks[i].weight.is_cuda)
            conv_out = self.conv_blocks[i](conv_out)
            #print(" 2. conv_out.is_cuda() ", conv_out.is_cuda)
            conv_out = self.pools[i](conv_out)

            block_input = conv_out
            conv_out = self.resblocks_1[i](conv_out)
            conv_out += block_input

            block_input = conv_out
            conv_out = self.resblocks_2[i](conv_out)
            conv_out += block_input
            #print(" 3. conv_out.is_cuda() ", conv_out.is_cuda)

        #print(" before relu . conv_out.is_cuda() ", conv_out.is_cuda)
        conv_out = self.relu(conv_out)
        #print(" after relu . conv_out.is_cuda() ", conv_out.is_cuda)
        conv_out = self.flatten(conv_out)
        #print(" after flatten . conv_out.is_cuda() ", conv_out.is_cuda)
        return conv_out


    def create_basic_res_block(self, in_channel, out_channel):
        return nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        if self.random:
            print("Random not implemented.")

        conv_out = self.feat_extract(inputs)
        conv_out = self.linear(conv_out)

        dist = self.actor(conv_out)
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()
        action_log_dist = dist.logits

        value = self.critic(conv_out)

        # print(type(action))

        return value, action, action_log_dist, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        conv_out = self.feat_extract(inputs)
        conv_out = self.linear(conv_out)
        return self.critic(conv_out)

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):

        conv_out = self.feat_extract(inputs)
        conv_out = self.linear(conv_out)

        dist = self.actor(conv_out)
        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        value = self.critic(conv_out)
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
