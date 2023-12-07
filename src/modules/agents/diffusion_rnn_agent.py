import torch.nn as nn
import torch.nn.functional as F

from modules.agents.utils.diffusion import Diffusion
from modules.agents.utils.model import MLP


class DiffusionRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(DiffusionRNNAgent, self).__init__()
        self.args = args
        self.beta_schedule = 'cosine'
        self.n_timesteps = 10
        self.max_action = 1.0

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)

        self.model = MLP(state_dim=args.rnn_hidden_dim, action_dim=args.n_actions, device=args.device)

        self.actor = Diffusion(state_dim=args.rnn_hidden_dim, action_dim=args.n_actions, model=self.model, max_action=self.max_action,
                               beta_schedule=self.beta_schedule, n_timesteps=self.n_timesteps,).to(args.device)

        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)

        # start diffusion process
        _, q, q_log, nonezero_mask, noise = self.actor(h)

        # q = self.fc2(h)
        return q, h, q_log, nonezero_mask, noise
