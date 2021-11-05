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
        actor_fc_layers=(32, 32),
        value_fc_layers=(32, 32),

        random=False):        
        super().__init__()

        self.random = random

        self.num_actions = action_space.n
        self.multi_dim = False
        self.action_dim = 1
        self.num_action_logits = self.num_actions

        self.action_space = action_space
        self.observation_space = observation_space
        obs_shape = observation_space.shape

        # Policy head
        self.actor = nn.Sequential(
            make_fc_layers_with_hidden_sizes(actor_fc_layers, input_size=obs_shape),
            Categorical(actor_fc_layers[-1], self.num_actions)
        )

        # Value head
        self.critic = nn.Sequential(
            make_fc_layers_with_hidden_sizes(value_fc_layers, input_size=obs_shape),
            init_(nn.Linear(value_fc_layers[-1], 1))
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



## this is a custom network for the adversary, it is like the multigrid network as it uses the map
class GFootballAdversaryNetwork(DeviceAwareModule):
    def __init__(
        self,
        observation_space,
        action_space,
        actor_fc_layers=(32, 32),
        value_fc_layers=(32, 32),
    ):
        super().__init__()

        self.input_shape = observation_space.shape
        self.action_space = action_space

        self.num_actions = action_space.n
        self.multi_dim = False
        self.action_dim = 1
        self.num_action_logits = self.num_actions

        self.actor = nn.Sequential(
                make_fc_layers_with_hidden_sizes(actor_fc_layers, input_size=self.input_shape),
                Categorical(actor_fc_layers[-1], self.num_actions)
            )
        self.critic = nn.Sequential(
            make_fc_layers_with_hidden_sizes(value_fc_layers, input_size=self.input_shape),
            init_(nn.Linear(value_fc_layers[-1], 1))
        )

    @property
    def is_recurrent(self):
        return self.rnn is not None

    def _select(self, embed, x):
        # Work around slow backward pass of nn.Embedding, see
        # https://github.com/pytorch/pytorch/issues/24912
        out = embed.weight.index_select(0, x.reshape(-1))
        return out.reshape(x.shape + (-1,))

    def _forward_base(self, inputs, rnn_hxs, masks):

        image = inputs.get(self.obs_key)

        if self.use_scalar:
            scalar = inputs.get('time_step')
            in_scalar = one_hot(self.scalar_dim, scalar).to(self.device)
            in_scalar = self.scalar_embed(in_scalar)
        else:
            in_scalar = torch.tensor([], device=self.device)

        in_z = inputs.get('random_z', torch.tensor([], device=self.device))

        if len(image.shape) == 3:
            image = image.unsqueeze(1)
        image_emb = self.image_embedding(image)
        in_features = torch.cat((image_emb, in_scalar, in_z), dim=-1)

        if self.recurrent_arch:
            core_features, rnn_hxs = self.rnn(in_features, rnn_hxs, masks)
        else:
            core_features = in_features

        return core_features, rnn_hxs

    def act(self, inputs, rnn_hxs, masks, deterministic=False):

        if self.random:
            B = inputs['image'].shape[0]
            if self.multi_dim:
                action = torch.zeros((B, 2), dtype=torch.int64, device=self.device)
                values = torch.zeros((B, 1), device=self.device)
                action_log_dist = torch.ones(B, self.action_space.nvec[0] + self.action_space.nvec[1], device=self.device)
                for b in range(B):
                    action[b] = torch.tensor(self.action_space.sample()).to(self.device)
            else:
                action = torch.zeros((B,1), dtype=torch.int64, device=self.device)
                values = torch.zeros((B,1), device=self.device)
                action_log_dist = torch.ones(B, self.action_space.n, device=self.device)
                for b in range(B):
                   action[b] = self.action_space.sample()

            return values, action, action_log_dist, rnn_hxs

        core_features, rnn_hxs = self._forward_base(inputs, rnn_hxs, masks)


        dist = self.actor(core_features)
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()
        action_log_dist = dist.logits
        dist_entropy = dist.entropy().mean()

        value = self.critic(core_features)

        return value, action, action_log_dist, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        core_features, rnn_hxs = self._forward_base(inputs, rnn_hxs, masks)
        return self.critic(core_features)

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        core_features, rnn_hxs = self._forward_base(inputs, rnn_hxs, masks)

        if self.multi_dim:
            dist_obj = self.actor_obj(core_features)
            dist_loc = self.actor_loc(core_features)

            action_obj_log_probs = dist_obj.log_probs(action[:, 0])
            action_loc_log_probs = dist_loc.log_probs(action[:, 1])

            action_log_probs = torch.cat((action_obj_log_probs, action_loc_log_probs),dim=1)

            obj_entropy = dist_obj.entropy().mean()
            loc_entropy = dist_loc.entropy().mean()
            dist_entropy = [obj_entropy,loc_entropy]
        else:
            dist = self.actor(core_features)
            action_log_probs = dist.log_probs(action)
            dist_entropy = dist.entropy().mean()

        value = self.critic(core_features)
        return value, action_log_probs, dist_entropy, rnn_hxs