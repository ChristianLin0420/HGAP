from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th


# This multi-agent controller shares parameters between agents
class BasicMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        if self.args.agent == "iqn_rnn":
            agent_outputs, rnd_quantiles = self.forward(ep_batch, t_ep, forward_type="approx")
        elif self.args.agent == "diffusion_rnn":
            agent_outputs, _, _, _ = self.forward(ep_batch, t_ep)
            # if self.args.use_bc_loss:
            #     agent_outputs, q_log, nonezero_mask, noise, bc_losses = self.forward(ep_batch, t_ep)
            # else:
            #     agent_outputs, q_log, nonezero_mask, noise = self.forward(ep_batch, t_ep)
        else:
            agent_outputs = self.forward(ep_batch, t_ep, forward_type=test_mode)
        if self.args.agent in ["iqn_rnn", "diffusion_rnn"]:
            agent_outputs = agent_outputs.view(ep_batch.batch_size, self.n_agents, self.args.n_actions, -1).mean(dim=3)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)

        if self.args.agent == "diffusion_rnn" and self.args.use_bc_loss:
            return chosen_actions, agent_outputs
        return chosen_actions
    
    def sample_actions(self, ep_batch, t, forward_type=None):
        agent_inputs, action_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]

        if self.args.agent == "diffusion_rnn":
            agent_outs = self.agent.sample_actions(agent_inputs, action_inputs, self.hidden_states)
        else:
            raise NotImplementedError
        
        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            if not forward_type:
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    # With probability epsilon, we will pick an available action uniformly
                    epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()

                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                               + th.ones_like(agent_outs) * self.action_selector.epsilon/epsilon_action_num)

                if getattr(self.args, "mask_before_softmax", True):
                    # Zero out the unavailable actions
                    agent_outs[reshaped_avail_actions == 0] = 0.0

        return agent_outs.clone()

    def forward(self, ep_batch, t, forward_type=None):
        if self.args.agent == "hpns_rnn":
            agent_inputs = self._build_inputs(ep_batch, t)
        else:
            agent_inputs, action_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]

        if self.args.agent == "iqn_rnn":
            agent_outs, self.hidden_states, rnd_quantiles = self.agent(agent_inputs, self.hidden_states, forward_type=forward_type)
        elif self.args.agent == "diffusion_rnn":
            # bc_losses = self.agent.get_bc_loss(agent_inputs, action_inputs, self.hidden_states)
            agent_outs, self.hidden_states, q_log, nonezero_mask, noise = self.agent(agent_inputs, self.hidden_states)
        else:
            agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            if not forward_type:
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    # With probability epsilon, we will pick an available action uniformly
                    epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()

                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                               + th.ones_like(agent_outs) * self.action_selector.epsilon/epsilon_action_num)

                if getattr(self.args, "mask_before_softmax", True):
                    # Zero out the unavailable actions
                    agent_outs[reshaped_avail_actions == 0] = 0.0
        if self.args.agent == "iqn_rnn":
            return agent_outs, rnd_quantiles
        elif self.args.agent == "diffusion_rnn":
            # if self.args.use_bc_loss:
            #     return agent_outs, q_log, nonezero_mask, noise, bc_losses
            return agent_outs, q_log, nonezero_mask, noise
        else:
            return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def parameters(self):
        return self.agent.parameters()

    def set_train_mode(self):
        self.agent.train()

    def set_evaluation_mode(self):
        self.agent.eval()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def cpu(self):
        self.agent.cpu()

    def get_device(self):
        return next(self.parameters()).device

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        action_inputs = []        
        inputs.append(batch["obs"][:, t])  # b1av
        action_inputs.append(batch["q_actions"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        action_inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in action_inputs], dim=1)
        return inputs, action_inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
