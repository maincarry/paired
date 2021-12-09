import numpy as np
import gym
import torch
from torch import nn
from torch.nn import functional as F

from .distributions import FixedCategorical, SquashedDiagGaussianDistribution
from .common import *

"""
this network is modified for the google football
"""


# the convolution layer of deepmind
class deepmind(nn.Module):
    def __init__(self):
        super(deepmind, self).__init__()
        self.conv1 = nn.Conv2d(16, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 32, 3, stride=1)
        self.fc1 = nn.Linear(32 * 5 * 8, 512)
        # start to do the init...
        nn.init.orthogonal_(self.conv1.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.conv2.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.conv3.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.fc1.weight.data, gain=nn.init.calculate_gain('relu'))
        # init the bias...
        nn.init.constant_(self.conv1.bias.data, 0)
        nn.init.constant_(self.conv2.bias.data, 0)
        nn.init.constant_(self.conv3.bias.data, 0)
        nn.init.constant_(self.fc1.bias.data, 0)

    def forward(self, x):
        assert x.shape[1] == 16 and x.shape[2] == 72 and x.shape[3] == 96, f"Wrong input obs shape {x.shape}"
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # print(x.shape)  # torch.Size([16, 32, 5, 8])
        # if x.shape[0] != 16:
        #     print(x.shape)

        x = x.reshape((-1, 32 * 5 * 8))
        x = F.relu(self.fc1(x))

        return x


# in the initial, just the nature CNN
class GFootballNetwork(DeviceAwareModule):
    def __init__(self, observation_space, action_space):
        super().__init__()
        print("Using SIMPLE version of GF Model.")
        self.device_name = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("device: ", self.device_name)
        self.recurrent_hidden_state_size = 1  # required but not used

        # network

        self.cnn_layer = deepmind()
        self.critic = nn.Linear(512, 1)
        self.actor = nn.Linear(512, action_space.n)



        # init the linear layer..
        nn.init.orthogonal_(self.critic.weight.data)
        nn.init.constant_(self.critic.bias.data, 0)
        # init the policy layer...
        nn.init.orthogonal_(self.actor.weight.data, gain=0.01)
        nn.init.constant_(self.actor.bias.data, 0)

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        x = self.cnn_layer(inputs / 255.0)

        dist = self.actor(x)
        dist = FixedCategorical(logits=dist)
        if deterministic:
            action = dist.mode()
            print("Warning: deterministic in model act")
        else:
            action = dist.sample()

        action_log_dist = dist.logits

        value = self.critic(x)

        return value, action, action_log_dist, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        x = self.cnn_layer(inputs / 255.0)
        return self.critic(x)

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):

        x = self.cnn_layer(inputs / 255.0)

        dist = self.actor(x)
        dist = FixedCategorical(logits=dist)
        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        value = self.critic(x)
        return value, action_log_probs, dist_entropy, rnn_hxs

    @property
    def is_recurrent(self):
        return False

    # def forward(self, inputs):
    #     x = self.cnn_layer(inputs / 255.0)
    #     value = self.critic(x)
    #     pi = F.softmax(self.actor(x), dim=1)
    #     return value, pi



class GFootballAdversaryNetwork(DeviceAwareModule):
    def __init__(self, observation_space, action_space):
        super().__init__()
        print("For ADV: Using SIMPLE version of GF Model.")
        self.device_name = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Adv Network device: ", self.device_name)
        self.recurrent_hidden_state_size = 1  # required but not used

        # scenic vars
        self.action_dim = action_space.shape[0] # number of sampleable variables

        # network
        self.cnn_layer = deepmind()
        self.critic = nn.Linear(512, 1)

        # TODO: output a guassian dist with -1,1 bound
        self.dist = SquashedDiagGaussianDistribution(self.action_dim)
        self.mean_net, self.logstd = self.dist.proba_distribution_net(512)
        # self.actor = nn.Linear(512, self.action_dim)


        # init the linear layer..
        nn.init.orthogonal_(self.critic.weight.data)
        nn.init.constant_(self.critic.bias.data, 0)
        # # init the policy layer...
        # nn.init.orthogonal_(self.actor.weight.data, gain=0.01)
        # nn.init.constant_(self.actor.bias.data, 0)

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        x = self.cnn_layer(inputs / 255.0)

        mean = self.mean_net(x)
        logstd = self.logstd
        normal_dist = self.dist.proba_distribution(mean, logstd)
        if deterministic:
            action = normal_dist.mode()
        else:
            action = normal_dist.sample()
        
        action_log_dist = mean # logits, in continuous case we should not need it

        value = self.critic(x)

        return value, action, action_log_dist, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        x = self.cnn_layer(inputs / 255.0)
        return self.critic(x)

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):

        x = self.cnn_layer(inputs / 255.0)

        mean = self.mean_net(x)
        logstd = self.logstd
        normal_dist = self.dist.proba_distribution(mean, logstd)
        action_log_probs = normal_dist.log_prob(action)
        dist_entropy = -action_log_probs.mean()


        value = self.critic(x)
        return value, action_log_probs, dist_entropy, rnn_hxs

    @property
    def is_recurrent(self):
        return False

    # def forward(self, inputs):
    #     x = self.cnn_layer(inputs / 255.0)
    #     value = self.critic(x)
    #     pi = F.softmax(self.actor(x), dim=1)
    #     return value, pi
