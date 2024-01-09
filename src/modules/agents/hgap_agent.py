import torch as th
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter

from modules.agents.utils.model import SetTransformerEncoderLayer


def kaiming_uniform_(tensor_w, tensor_b, mode='fan_in', gain=12 ** (-0.5)):
    fan = nn.init._calculate_correct_fan(tensor_w.data, mode)
    std = gain / math.sqrt(fan)
    bound_w = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    bound_b = 1 / math.sqrt(fan)
    with th.no_grad():
        tensor_w.data.uniform_(-bound_w, bound_w)
        if tensor_b is not None:
            tensor_b.data.uniform_(-bound_b, bound_b)


class Merger(nn.Module):
    def __init__(self, head, fea_dim):
        super(Merger, self).__init__()
        self.head = head
        if head > 1:
            self.weight = Parameter(th.Tensor(1, head, fea_dim).fill_(1.))
            self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        :param x: [bs, n_head, fea_dim]
        :return: [bs, fea_dim]
        """
        if self.head > 1:
            return th.sum(self.softmax(self.weight) * x, dim=1, keepdim=False)
        else:
            return th.squeeze(x, dim=1)

class HGAP_Agent(nn.Module):
    def __init__(self, input_shape, args):
        super(HGAP_Agent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_allies = args.n_allies
        self.n_enemies = args.n_enemies
        self.n_entities = 1 + self.n_allies + self.n_enemies
        self.n_actions = args.n_actions
        self.n_heads = args.hpn_head_num
        self.rnn_hidden_dim = args.rnn_hidden_dim
        self.hgap_hyper_dim = args.hgap_hyper_dim

        # [4 + 1, (6, 5), (4, 5)]
        self.own_feats_dim, self.enemy_feats_dim, self.ally_feats_dim = input_shape
        self.enemy_feats_dim = self.enemy_feats_dim[-1]  # [n_enemies, feat_dim]
        self.ally_feats_dim = self.ally_feats_dim[-1]  # [n_allies, feat_dim]

        if self.args.obs_agent_id:
            # embedding table for agent_id
            self.agent_id_embedding = th.nn.Embedding(self.n_agents, self.rnn_hidden_dim)

        if self.args.obs_last_action:
            # embedding table for action id
            self.action_id_embedding = th.nn.Embedding(self.n_actions, self.rnn_hidden_dim)

        # Hyper network for GAT embedding
        self.hyper_embedding = nn.Sequential(
            nn.Linear(self.own_feats_dim, args.hgap_hyper_dim),
            nn.ReLU(inplace=True),
            nn.Linear(args.hgap_hyper_dim, self.own_feats_dim * self.hgap_hyper_dim * self.n_heads)
        )

        self.hyper_allies_embedding = nn.Sequential(
            nn.Linear(self.ally_feats_dim, args.hgap_hyper_dim),
            nn.ReLU(inplace=True),
            nn.Linear(args.hgap_hyper_dim, self.ally_feats_dim * self.hgap_hyper_dim * self.n_heads)
        )

        self.hyper_enemies_embedding = nn.Sequential(
            nn.Linear(self.enemy_feats_dim, args.hgap_hyper_dim),
            nn.ReLU(inplace=True),
            nn.Linear(args.hgap_hyper_dim, self.enemy_feats_dim * self.hgap_hyper_dim * self.n_heads)
        )
            
        # Hyper network for GAT attention weights
        self.hyper_attention_weight = nn.Sequential(
            nn.Linear(self.own_feats_dim, args.hgap_hyper_dim),
            nn.ReLU(inplace=True),
            nn.Linear(args.hgap_hyper_dim, 2 * args.hgap_hyper_dim * self.n_heads)
        )

        self.activation = nn.LeakyReLU(negative_slope = 0.2)
        self.softmax = nn.Softmax(dim = 1)
        self.dropout = nn.Dropout(0.5)

        self.attetion_weight_mapper = nn.Linear(2 * self.hgap_hyper_dim, 1)
        self.unify_output_heads_rescue = Merger(self.n_heads, 1)

        self.fc1_own = nn.Linear(self.own_feats_dim, self.rnn_hidden_dim, bias=True)  # only one bias is OK
        self.rnn = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)
        self.fc2_normal_actions = nn.Linear(self.rnn_hidden_dim, args.output_normal_actions)  # (no_op, stop, up, down, right, left)
        self.unify_output_heads = Merger(self.n_heads, 1)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1_own.weight.new(1, self.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):

        # [bs, n_agents, mv_fea_dim], [bs * n_agents * n_enemies, enemy_fea_dim], [bs * n_agents * n_allies, ally_fea_dim], [bs, n_agents, own_fea_dim]
        bs, own_feats_t, enemy_feats_t, ally_feats_t, embedding_indices = inputs

        own_feats_t = own_feats_t.reshape(bs * self.n_agents, 1, self.own_feats_dim)  # [bs * n_agents, own_fea_dim]
        enemy_feats_t = enemy_feats_t.reshape(bs * self.n_agents, self.n_enemies, self.enemy_feats_dim)  # [bs * n_agents * n_enemies, enemy_fea_dim]
        ally_feats_t = ally_feats_t.reshape(bs * self.n_agents, self.n_allies, self.ally_feats_dim)  # [bs * n_agents * n_allies, ally_fea_dim]

        hyper_own_embedding_weight = self.hyper_embedding(own_feats_t).view(bs * self.n_agents, self.own_feats_dim, self.hgap_hyper_dim * self.n_heads)  # [bs * n_agents, feat_dim, rnn_hidden_dim * n_heads]
        hyper_ally_embedding_weight = self.hyper_allies_embedding(ally_feats_t).view(bs * self.n_agents * (self.n_allies), self.ally_feats_dim, self.hgap_hyper_dim * self.n_heads)  # [bs * n_agents * (1 + n_enemies + n_allies), feat_dim, rnn_hidden_dim * n_heads]
        hyper_enemy_embedding_weight = self.hyper_enemies_embedding(enemy_feats_t).view(bs * self.n_agents * self.n_enemies, self.enemy_feats_dim, self.hgap_hyper_dim * self.n_heads)  # [bs * n_agents * n_enemies, feat_dim, rnn_hidden_dim * n_heads]

        # hyper_embedding_weight = self.hyper_embedding(feats_t).view(bs * self.n_agents * self.n_entities, self.own_feats_dim, self.hgap_hyper_dim * self.n_heads)  # [bs * n_agents * (1 + n_enemies + n_allies), feat_dim, rnn_hidden_dim * n_heads]
        # entities_embedding = th.matmul(feats_t.view(bs * self.n_agents * self.n_entities, 1, self.own_feats_dim), hyper_embedding_weight).view(bs, self.n_agents, self.n_entities, self.n_heads, self.hgap_hyper_dim)
        own_embedding = th.matmul(own_feats_t.view(bs * self.n_agents, 1, self.own_feats_dim), hyper_own_embedding_weight).view(bs, self.n_agents, 1, self.n_heads, self.hgap_hyper_dim)
        ally_embedding = th.matmul(ally_feats_t.view(bs * self.n_agents * (self.n_allies), 1, self.ally_feats_dim), hyper_ally_embedding_weight).view(bs, self.n_agents, self.n_allies, self.n_heads, self.hgap_hyper_dim)
        enemy_embedding = th.matmul(enemy_feats_t.view(bs * self.n_agents * self.n_enemies, 1, self.enemy_feats_dim), hyper_enemy_embedding_weight).view(bs, self.n_agents, self.n_enemies, self.n_heads, self.hgap_hyper_dim)
        entities_embedding = th.cat([own_embedding, ally_embedding, enemy_embedding], dim=2)  # [bs, n_agents, n_entities, n_heads, rnn_hidden_dim]

        entities_embedding_repeat = entities_embedding.repeat(1, 1, self.n_entities, 1, 1)  # [bs, n_agents, n_entities * n_entities, n_heads, rnn_hidden_dim]
        entities_embedding_interleave_repeat = entities_embedding.repeat_interleave(self.n_entities, dim=2)  # [bs, n_agents, n_entities * n_entities, n_heads, rnn_hidden_dim]

        entities_embedding_concat = self.activation(th.cat([entities_embedding_repeat, entities_embedding_interleave_repeat], dim=-1)).view(bs * self.n_agents * self.n_entities * self.n_heads, self.n_entities, 2 * self.hgap_hyper_dim)  # [bs * n_agents * n_entities * n_entities, n_heads, rnn_hidden_dim * 2]

        hyper_attention_weight = self.attetion_weight_mapper(entities_embedding_concat).view(bs * self.n_agents, self.n_entities, self.n_entities, self.n_heads)  # [bs * n_agents * n_entities * n_entities, n_heads, 1]

        entities_embedding = entities_embedding.view(bs * self.n_agents, self.n_entities, self.n_heads, self.hgap_hyper_dim)  # [bs * n_agents, n_entities, n_heads, rnn_hidden_dim]
        hyper_attention_output = th.einsum('bijh, bjhf->bihf', hyper_attention_weight, entities_embedding).view(bs * self.n_agents, self.n_entities, self.n_heads, self.hgap_hyper_dim)
        hyper_attention_output = self.unify_output_heads_rescue(hyper_attention_output).view(bs * self.n_agents, self.n_entities, self.hgap_hyper_dim)  # [bs * n_agents, n_entities, rnn_hidden_dim]

        # Split the output of hyper net into 2 parts along last dimension
        recurrent_weight = hyper_attention_output[:, :, :self.rnn_hidden_dim]  # [bs * n_agents, n_entities, rnn_hidden_dim]
        embedding = th.mean(recurrent_weight, dim = 1)  # [bs * n_agents, 1, rnn_hidden_dim]
        action_embedding = hyper_attention_output[:, -self.n_enemies:, self.rnn_hidden_dim:].transpose(1, 2)  # [bs * n_agents, n_enemies, rnn_hidden_dim]

        x = F.relu(embedding, inplace=True)
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)
        hh = self.rnn(x, h_in)  # [bs * n_agents, rnn_hidden_dim]

        # Q-values of normal actions
        q_normal = self.fc2_normal_actions(hh).view(bs, self.n_agents, -1)  # [bs, n_agents, 6]

        # [bs*n_agents, 1, rnn_hidden_dim] * [bs*n_agents, rnn_hidden_dim, n_enemies*head] -> [bs*n_agents, 1, n_enemies*head]
        q_attacks = (th.matmul(hh.unsqueeze(1), action_embedding).squeeze(1)).view(
            bs, self.n_agents, self.n_enemies
        )

        # Concat 2 types of Q-values
        q = th.cat((q_normal, q_attacks), dim=-1)  # [bs, n_agents, 6 + n_enemies]
        return q.view(bs, self.n_agents, -1), hh.view(bs, self.n_agents, -1)  # [bs, n_agents, 6 + n_enemies]
