import torch as th
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
        self.max_action = 10.0

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)

        self.model = MLP(state_dim=args.rnn_hidden_dim, action_dim=args.n_actions, device=args.device)

        self.actor = Diffusion(state_dim=args.rnn_hidden_dim, action_dim=args.n_actions, model=self.model, max_action=self.max_action,
                               beta_schedule=self.beta_schedule, n_timesteps=self.n_timesteps, predict_epsilon=False).to(args.device)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)

        # start diffusion process
        _, q, q_log, nonezero_mask, noise = self.actor(h)

        return q, h, q_log, nonezero_mask, noise
    
    def sample_actions(self, inputs, actions, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)

        # start diffusion process
        actions = actions.type(th.float32)
        noise = th.randn_like(actions)
        t = th.randint(0, self.actor.n_timesteps, (actions.shape[0],), device=actions.device).long()

        x_noisy = self.actor.q_sample(x_start = actions, t = t, noise = noise)
        t = t.unsqueeze(-1).float()
        x_recon = self.actor.model(x_noisy, t, h)

        return x_recon.clone()


    def get_bc_loss(self, inputs, actions, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)

        # computing the loss
        actions = actions.type(th.float32)
        loss = self.actor.loss(actions, h)

        return loss
        