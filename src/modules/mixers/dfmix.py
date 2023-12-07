import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DFMixer(nn.Module):
    def __init__(self, args):
        super(DFMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))

        self.embed_dim = args.mixing_embed_dim
        self.hypernet_embed = args.hypernet_embed

        self.n_quantiles = args.n_quantiles
        self.n_target_quantiles = args.n_target_quantiles
        self.n_agents = args.n_agents

        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_factorize = nn.Linear(
                self.state_dim, self.embed_dim * self.n_agents)
            self.hyper_w_final_factorize = nn.Linear(
                self.state_dim, self.embed_dim)
            self.hyper_w_shape = nn.Linear(
                self.state_dim, self.embed_dim * self.n_agents)
            self.hyper_w_final_shape = nn.Linear(
                self.state_dim, self.embed_dim)
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = self.hypernet_embed
            self.hyper_w_factorize = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                                   nn.ReLU(),
                                                   nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
            self.hyper_w_final_factorize = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                                         nn.ReLU(),
                                                         nn.Linear(hypernet_embed, self.embed_dim))
            self.hyper_w_shape = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                               nn.ReLU(),
                                               nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
            self.hyper_w_final_shape = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                                     nn.ReLU(),
                                                     nn.Linear(hypernet_embed, self.embed_dim))
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, agent_vars, states):
        # Factorization network
        q_mean_expected = self.forward_factorize(agent_qs, states)

        # Shape network
        q_var_expected = self.forward_shape(agent_vars, states)

        return q_mean_expected + q_var_expected

    def forward_factorize(self, inputs, states):
        bs = inputs.size(0)
        inputs = inputs.view(-1, 1, self.n_agents)
        states = states.reshape(-1, self.state_dim)

        # First layer
        w1 = th.abs(self.hyper_w_factorize(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(inputs, w1) + b1)
        # Second layer
        w_final = th.abs(self.hyper_w_final_factorize(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = th.bmm(hidden, w_final) + v
        # Reshape and return
        q_mean = y.view(bs, -1, 1)
        q_mean = q_mean.unsqueeze(3)

        return q_mean

    def forward_shape(self, inputs, states):
        bs = inputs.size(0)
        inputs = inputs.view(-1, 1, self.n_agents)
        states = states.reshape(-1, self.state_dim)

        # First layer
        w1 = th.abs(self.hyper_w_shape(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(inputs, w1) + b1)
        # Second layer
        w_final = th.abs(self.hyper_w_final_shape(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = th.bmm(hidden, w_final) + v
        # Reshape and return
        q_shape = y.view(bs, -1, 1)
        q_shape = q_shape.unsqueeze(3)

        return q_shape
