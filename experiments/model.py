"""
https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/vpg/core.py
"""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

import dm_construction


class RNNEncoder(nn.Module):
    """
    Preprocess state observations, a sequence of object state vectors,
    into a single output vector for downstream tasks.

    Input: object state vectors / graph nodes
    Output: hidden state to be used as input for policy
    """
    def __init__(self, obs_dim, hidden_size=256):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size=obs_dim,
                          hidden_size=hidden_size,
                          num_layers=1,
                          bidirectional=False)

    def forward(self, obs):
        # TODO: handle batch_size more generally (currently set to 1)
        # TODO: tensor to device
        # Grab nodes from graph observation, ignore other attributes
        obs_nodes = torch.tensor(obs["nodes"])
        obs_nodes = obs_nodes.reshape(obs_nodes.shape[0], 1, -1)
        output, hn = self.gru(obs_nodes)
        # output shape: (seq_len, batch, num_directions * hidden_size)
        # hn shape: (num_layers * num_directions, batch, hidden_size)
        return output, hn

    
class MLPGaussianActorCritic(nn.Module):
    """
    Actor-Critic with dual policy and value heads.

    Object observation encoding:
    RNN encodes graph observations by processing the sequence of
    object state vectors in the order they are listed into a single
    vector (the final hidden state).

    Policy:
    An MLP with 4 hidden layers of size 256 units receives the hidden
    state from the RNN as input and outputs a Gaussian policy.

    Value function:
    An MLP with 2 hidden layers of size 256 units.
    """
    def __init__(self, obs_dim, act_dim, rnn_hidden_size=256, mlp_hidden_size=256):
        super().__init__()
        self.rnn_hidden_size = rnn_hidden_size
        self.mlp_hidden_size = mlp_hidden_size

        # RNN encoder
        self.rnn_encoder = RNNEncoder(obs_dim, rnn_hidden_size)

        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

        # MLP core
        self.mu_net = nn.Sequential(
            nn.Linear(rnn_hidden_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, act_dim)
        )

        # Critic
        self.v = nn.Sequential(
            nn.Linear(rnn_hidden_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, 1)
        )

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        _, hn = self.rnn_encoder(obs)
        pi = self._distribution(hn)
        v = torch.squeeze(self.v(hn), -1)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, v, logp_a

    def _distribution(self, obs_embed):
        mu = self.mu_net(obs_embed)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)

    @staticmethod
    def _action_to_dict(a):
        a = a.squeeze().numpy()
        return {'Horizontal': a[0],
                'Vertical': a[1],
                'Sticky': a[2],
                'Selector': a[3]}

    # def initHidden(self):
    #     pass


def wrapper_type_from_ac_spec(ac_spec):

    if ac_spec.keys() == set(['Index', 'x_action', 'sticky']):
        return 'discrete_relative'
    elif ac_spec.keys() == set(['Horizontal', 'Vertical', 'Sticky', 'Selector']):
        return 'continuous_absolute'
    else:
        raise ValueError("Unrecognized action spec")


class ActorCritic(nn.Module):

    def __init__(self, ob_spec, ac_spec):
        super().__init__()

        assert 'nodes' in ob_spec, \
            "Only graph obs supported"

        obs_dim = ob_spec["nodes"].shape[1]  # 18 for all tasks except marble run
        ac_spec_type = wrapper_type_from_ac_spec(ac_spec)

        # TODO: assertion for ac_spec dimension

        # policy builder depends on action space
        if ac_spec_type == 'continuous_absolute':
            self.ac = MLPGaussianActorCritic(obs_dim=obs_dim,
                                             act_dim=4,
                                             rnn_hidden_size=256,
                                             mlp_hidden_size=256)
        else:
            raise NotImplementedError

    def step(self, obs):
        with torch.no_grad():
            _, obs_embed = self.ac.rnn_encoder(obs)
            pi = self.ac._distribution(obs_embed)
            a = pi.sample()
            logp_a = self.ac._log_prob_from_distribution(pi, a)
            v = torch.squeeze(self.ac.v(obs_embed), -1)
        return self.ac._action_to_dict(a), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]
