import torch.nn as nn
import torch.nn.functional as F
import torch
import argparse


class UPDeT(nn.Module):
    def __init__(self, input_shape, args):
        super(UPDeT, self).__init__()
        self.args = args

        self.n_agents = args.n_agents
        self.n_allies = args.n_allies
        self.n_enemies = args.n_enemies
        self.n_entities = 1 + self.n_allies + self.n_enemies

        # [4 + 1, (6, 5), (4, 5)]
        self.own_feats_dim, self.enemy_feats_dim, self.ally_feats_dim = input_shape
        self.enemy_feats_dim = self.enemy_feats_dim[-1]  # [n_enemies, feat_dim]
        self.ally_feats_dim = self.ally_feats_dim[-1]  # [n_allies, feat_dim]

        # Hyper network for UPDeT embedding
        self.hyper_embedding = nn.Sequential(
            nn.Linear(self.own_feats_dim, args.hgap_hyper_dim),
            nn.ReLU(inplace=True),
            nn.Linear(args.hgap_hyper_dim, args.emb)
        )

        self.hyper_allies_embedding = nn.Sequential(
            nn.Linear(self.ally_feats_dim, args.hgap_hyper_dim),
            nn.ReLU(inplace=True),
            nn.Linear(args.hgap_hyper_dim, args.emb)
        )

        self.hyper_enemies_embedding = nn.Sequential(
            nn.Linear(self.enemy_feats_dim, args.hgap_hyper_dim),
            nn.ReLU(inplace=True),
            nn.Linear(args.hgap_hyper_dim, args.emb)
        )

        self.transformer = Transformer(args.token_dim, args.emb, args.heads, args.depth, args.emb)
        self.q_basic = nn.Linear(args.emb, 6)

    def init_hidden(self):
        # make hidden states on same device as model
        return torch.zeros(1, self.args.emb).cuda()

    def forward(self, inputs, hidden_state):

        # [bs, n_agents, mv_fea_dim], [bs * n_agents * n_enemies, enemy_fea_dim], [bs * n_agents * n_allies, ally_fea_dim], [bs, n_agents, own_fea_dim]
        bs, own_feats_t, enemy_feats_t, ally_feats_t, embedding_indices = inputs

        own_feats_t = own_feats_t.reshape(bs * self.n_agents, 1, self.own_feats_dim)  # [bs * n_agents, own_fea_dim]
        enemy_feats_t = enemy_feats_t.reshape(bs * self.n_agents, self.n_enemies, self.enemy_feats_dim)  # [bs * n_agents * n_enemies, enemy_fea_dim]
        ally_feats_t = ally_feats_t.reshape(bs * self.n_agents, self.n_allies, self.ally_feats_dim)  # [bs * n_agents * n_allies, ally_fea_dim]

        own_feats = self.hyper_embedding(own_feats_t)  # [bs * n_agents, emb]
        enemy_feats = self.hyper_enemies_embedding(enemy_feats_t)  # [bs * n_agents * n_enemies, emb]
        ally_feats = self.hyper_allies_embedding(ally_feats_t)  # [bs * n_agents * n_allies, emb]

        # [bs * n_agents, 1, emb], [bs * n_agents, n_enemies, emb], [bs * n_agents, n_allies, emb]
        feats = torch.cat((own_feats, ally_feats, enemy_feats), dim=1).reshape(bs * self.n_agents, 1 + self.n_enemies + self.n_allies, self.args.emb)

        outputs, _, dot = self.transformer.forward(feats, hidden_state, None)
        # first output for 6 action (no_op stop up down left right)
        q_basic_actions = self.q_basic(outputs[:, 0, :])

        # last dim for hidden state
        h = outputs[:, -1:, :]

        q_enemies_list = []

        # each enemy has an output Q
        for i in range(self.args.n_enemies):
            q_enemy = self.q_basic(outputs[:, 1 + i, :])
            q_enemy_mean = torch.mean(q_enemy, 1, True)
            q_enemies_list.append(q_enemy_mean)

        # concat enemy Q over all enemies
        q_enemies = torch.stack(q_enemies_list, dim=1).squeeze()

        # concat basic action Q with enemy attack Q
        q = torch.cat((q_basic_actions, q_enemies), 1)

        if self.args.evaluate:
            return q, h, dot[:self.n_agents, :-1, :-1].view(self.n_agents, self.n_entities, self.n_entities)

        return q, h

class SelfAttention(nn.Module):
    def __init__(self, emb, heads=8, mask=False):

        super().__init__()

        self.emb = emb
        self.heads = heads
        self.mask = mask

        self.tokeys = nn.Linear(emb, emb * heads, bias=False)
        self.toqueries = nn.Linear(emb, emb * heads, bias=False)
        self.tovalues = nn.Linear(emb, emb * heads, bias=False)

        self.unifyheads = nn.Linear(heads * emb, emb)

    def forward(self, x, mask):

        b, t, e = x.size()
        h = self.heads
        keys = self.tokeys(x).view(b, t, h, e)
        queries = self.toqueries(x).view(b, t, h, e)
        values = self.tovalues(x).view(b, t, h, e)

        # compute scaled dot-product self-attention

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)

        queries = queries / (e ** (1 / 4))
        keys = keys / (e ** (1 / 4))
        # - Instead of dividing the dot products by sqrt(e), we scale the keys and values.
        #   This should be more memory efficient

        # - get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))

        assert dot.size() == (b * h, t, t)

        if self.mask:  # mask out the upper half of the dot matrix, excluding the diagonal
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)

        if mask is not None:
            dot = dot.masked_fill(mask == 0, -1e9)

        dot = F.softmax(dot, dim=2)
        # - dot now has row-wise self-attention probabilities

        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b, h, t, e)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, h * e)

        return self.unifyheads(out), dot

class TransformerBlock(nn.Module):

    def __init__(self, emb, heads, mask, ff_hidden_mult=4, dropout=0.0):
        super().__init__()

        self.attention = SelfAttention(emb, heads=heads, mask=mask)
        self.mask = mask

        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)

        self.ff = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb)
        )

        self.do = nn.Dropout(dropout)

    def forward(self, x_mask):
        x, mask = x_mask

        attended, dot = self.attention(x, mask)

        x = self.norm1(attended + x)

        x = self.do(x)

        fedforward = self.ff(x)

        x = self.norm2(fedforward + x)

        x = self.do(x)

        return x, mask, dot


class Transformer(nn.Module):

    def __init__(self, input_dim, emb, heads, depth, output_dim):
        super().__init__()

        self.num_tokens = output_dim

        # self.token_embedding = nn.Linear(input_dim, emb)

        tblocks = []
        for i in range(depth):
            tblocks.append(
                TransformerBlock(emb=emb, heads=heads, mask=False))

        self.tblocks = nn.Sequential(*tblocks)

        self.toprobs = nn.Linear(emb, output_dim)

    def forward(self, x, h, mask):

        bn = x.size(0)
        h = h.view(bn, 1, -1)

        tokens = torch.cat((x, h), 1)

        b, t, e = tokens.size()

        x, mask, dot = self.tblocks((tokens, mask))

        x = self.toprobs(x.view(b * t, e)).view(b, t, self.num_tokens)
        
        return x, tokens, dot

def mask_(matrices, maskval=0.0, mask_diagonal=True):

    b, h, w = matrices.size()
    indices = torch.triu_indices(h, w, offset=0 if mask_diagonal else 1)
    matrices[:, indices[0], indices[1]] = maskval


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unit Testing')
    parser.add_argument('--token_dim', default='5', type=int)
    parser.add_argument('--emb', default='32', type=int)
    parser.add_argument('--heads', default='3', type=int)
    parser.add_argument('--depth', default='2', type=int)
    parser.add_argument('--ally_num', default='5', type=int)
    parser.add_argument('--enemy_num', default='5', type=int)
    parser.add_argument('--episode', default='20', type=int)
    args = parser.parse_args()


    # testing the agent
    agent = UPDeT(None, args).cuda()
    hidden_state = agent.init_hidden().cuda().expand(args.ally_num, 1, -1)
    tensor = torch.rand(args.ally_num, args.ally_num+args.enemy_num, args.token_dim).cuda()
    q_list = []
    for _ in range(args.episode):
        q, hidden_state = agent.forward(tensor, hidden_state, args.ally_num, args.enemy_num)
        q_list.append(q)