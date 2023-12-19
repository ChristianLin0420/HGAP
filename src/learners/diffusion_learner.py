import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.ddn import DDNMixer
from modules.mixers.dmix import DMixer
from modules.mixers.dfmix import DFMixer
import torch as th
import torch.nn.functional as F
from torch.optim import RMSprop
from torch.optim import Adam
import numpy as np


class DiffusionLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())
        self.agent_params = list(mac.agent.actor.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "dfmix":
                self.mixer = DFMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        # Q-learning optimiser
        if args.optimizer == "RMSProp":
            self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        elif args.optimizer == "Adam":
            self.optimiser = Adam(params=self.params, lr=args.lr, eps=args.optim_eps)
        else:
            raise ValueError("Unknown Optimizer")
        
        # BC optimiser
        self.agent_optimiser = Adam(params=self.agent_params, lr=args.lr, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.n_quantiles = args.n_quantiles
        self.n_target_quantiles = args.n_target_quantiles

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        states = batch["state"][:, :-1]
        actions = batch["actions"][:, :-1]
        q_actions = batch["q_actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        mac_out = []
        log_variances = []
        bc_losses = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            if self.args.agent == "diffusion_rnn":
                agent_outs, agent_log_variance, nonezero_mask, noise = self.mac.forward(batch, t=t, forward_type="policy")
                orig_actions = batch["q_actions"][:, t]
                recon_actions = self.mac.sample_actions(batch, t=t, forward_type="policy").view(batch.batch_size, self.args.n_agents, self.args.n_actions)
                bc_loss = F.mse_loss(orig_actions, recon_actions, reduction='none').view(batch.batch_size, self.args.n_agents, self.args.n_actions)
                bc_losses.append(bc_loss)
            agent_log_variance = nonezero_mask * (0.5 * agent_log_variance).exp() * noise
            agent_outs = agent_outs.view(batch.batch_size, self.args.n_agents, self.args.n_actions)
            agent_log_variance = agent_log_variance.view(batch.batch_size, self.args.n_agents, self.args.n_actions)
            mac_out.append(agent_outs)
            log_variances.append(agent_log_variance)

        mac_out = th.stack(mac_out, dim=1) # Concat over time
        log_variances = th.stack(log_variances, dim=1) # Concat over time

        if self.args.use_bc_loss:
            bc_losses = th.stack(bc_losses, dim=1)
            bc_losses = bc_losses.mean(-1, keepdim=True)

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:,:-1], dim=3, index=actions).squeeze(3)  # Remove the action dim
        chosen_action_q_logs = th.gather(log_variances[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the action dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        target_log_variances = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            if self.args.agent == "diffusion_rnn":
                target_agent_outs, target_agent_log_variance, target_nonezero_mask, target_noise = self.mac.forward(batch, t=t, forward_type="policy")
            target_agent_log_variance = target_nonezero_mask * (0.5 * target_agent_log_variance).exp() * target_noise
            target_agent_outs = target_agent_outs.view(batch.batch_size, self.args.n_agents, self.args.n_actions)
            target_agent_log_variance = target_agent_log_variance.view(batch.batch_size, self.args.n_agents, self.args.n_actions)
            target_mac_out.append(target_agent_outs)
            target_log_variances.append(target_agent_log_variance)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time
        target_log_variances = th.stack(target_log_variances[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:,1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:,1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            target_max_q_logs = th.gather(target_log_variances, 3, cur_max_actions).squeeze(3)
        else:
            # [0] is for max value; [1] is for argmax
            cur_max_actions = target_mac_out.max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            target_max_q_logs = th.gather(target_log_variances, 3, cur_max_actions).squeeze(3)

        # Mix
        if self.mixer is not None:
            target_max_qvals = self.target_mixer(target_max_qvals.clone(), target_max_q_logs, batch["state"][:, 1:]).squeeze(3)
            chosen_action_qvals = self.mixer(chosen_action_qvals.clone(), chosen_action_q_logs, batch["state"][:, :-1]).squeeze(3)

        # Calculate 1-step Q-Learning targets
        targets = rewards + (self.args.gamma * (1 - terminated)) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # bc loss
        bc_delta = 0.001
        accumulated_bc_loss = bc_delta * bc_losses.sum() / (self.args.n_agents * self.args.n_actions)

        # Optimise Agents by BC loss
        # if self.args.use_bc_loss:
        #     self.agent_optimiser.zero_grad()
        #     accumulated_bc_loss.backward(retain_graph=True)
        #     bc_grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.bc_grad_norm_clip)
        #     self.agent_optimiser.step()

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum() #+ accumulated_bc_loss

        # Optimise Mixer + Agents
        self.agent_optimiser.zero_grad()
        self.optimiser.zero_grad()
        accumulated_bc_loss.backward(retain_graph=True)
        loss.backward()
        bc_grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.bc_grad_norm_clip)
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.agent_optimiser.step()
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
            if self.args.use_bc_loss:
                self.logger.log_stat("bc_loss", accumulated_bc_loss.item(), t_env)
                self.logger.log_stat("bc_grad_norm", bc_grad_norm.item(), t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
