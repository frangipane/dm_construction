"""
https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/vpg/core.py
"""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

import dm_construction


class CNNEncoder(nn.Module):
    """Process RGB image observations.
    """
    def __init__(self, obs_dim,):
        super().__init__()
        self.height, self.width, self.channels = obs_dim

        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(self.channels, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )

    @property
    def image_embedding_size(self):
        return ((self.height-1)//2-2)*((self.width-1)//2-2)*64

    def forward(self, obs):
        x = torch.as_tensor(obs, dtype=torch.float32)
        if x.dim() == 3:
            x = x.unsqueeze(0)  # add batch size as 0th dim
        x = x.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)
        return x


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
        return hn


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
    def __init__(self,
                 obs_dim,
                 act_dim,
                 ob_type,
                 mlp_hidden_size=256,
                 encoder_embedding_size=None,):
        super().__init__()

        if ob_type == "graph":
            self.encoder = RNNEncoder(obs_dim, encoder_embedding_size)
            self.encoder_embedding_size = encoder_embedding_size
        elif ob_type == "image":
            if encoder_embedding_size is not None:
                print("Ignore user input embedding size for CNN")
            self.encoder = CNNEncoder(obs_dim)
            self.encoder_embedding_size = self.encoder.image_embedding_size
        else:
            raise ValueError

        self.mlp_hidden_size = mlp_hidden_size

        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

        # MLP core
        self.mu_net = nn.Sequential(
            nn.Linear(self.encoder_embedding_size, mlp_hidden_size),
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
            nn.Linear(self.encoder_embedding_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, 1)
        )

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        obs_embed = self.encoder(obs)
        pi = self._distribution(obs_embed)
        v = torch.squeeze(self.v(obs_embed), -1)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, v, logp_a

    def _distribution(self, obs_embed):
        mu = self.mu_net(obs_embed)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)

    # def initHidden(self):
    #     pass


def wrapper_type_from_ac_spec(ac_spec):

    if ac_spec.keys() == set(['Index', 'x_action', 'sticky']):
        return 'discrete_relative'
    elif ac_spec.keys() == set(['Horizontal', 'Vertical', 'Sticky', 'Selector']):
        return 'continuous_absolute'
    else:
        raise ValueError("Unrecognized action spec")


def wrapper_type_from_ob_spec(ob_spec):
    """Returns:
      wrapper_type: str,
        'image' or 'graph'
      ob_dim: int
    """
    ob_name = getattr(ob_spec, 'name', None)

    if ob_name == 'AgentCamera':
        wrapper_type = 'image'
        ob_dim = ob_spec.shape  # (64, 64, 3)
    elif ob_name is None and 'nodes' in ob_spec:
        wrapper_type = 'graph'
        ob_dim = ob_spec['nodes'].shape[1] # 18 for all tasks except marble run
    else:
        raise ValueError("Unrecognized observation spec")

    return wrapper_type, ob_dim


class ActorCritic(nn.Module):

    def __init__(self, ob_spec, ac_spec, embed_size=None, mlp_hidden_size=256):
        super().__init__()

        ob_type, ob_dim = wrapper_type_from_ob_spec(ob_spec)
        ac_type = wrapper_type_from_ac_spec(ac_spec)

        # TODO: assertion for ac_spec dimension

        # policy builder depends on action space
        if ac_type == 'continuous_absolute':
            embed_size = embed_size if ob_type == 'graph' else None
            self.ac = MLPGaussianActorCritic(obs_dim=ob_dim,
                                             act_dim=4,
                                             ob_type=ob_type,
                                             mlp_hidden_size=mlp_hidden_size,
                                             encoder_embedding_size=embed_size)
        else:
            raise NotImplementedError(f"Model does not support {ac_type} actions")

    def step(self, obs):
        with torch.no_grad():
            obs_embed = self.ac.encoder(obs)
            pi = self.ac._distribution(obs_embed)
            a = pi.sample()
            logp_a = self.ac._log_prob_from_distribution(pi, a)
            v = torch.squeeze(self.ac.v(obs_embed), -1)
        return a.squeeze().numpy(), v.numpy(), logp_a.squeeze().numpy()

    def act(self, obs):
        return self.step(obs)[0]

    @staticmethod
    def action_to_dict(a):
        # a is a numpy vector
        return {'Horizontal': a[0],
                'Vertical': a[1],
                'Sticky': a[2],
                'Selector': a[3]}

