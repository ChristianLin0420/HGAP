from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
import os

class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False):
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        map_width, map_height = self.env.get_map_sizes()

        episode_record = {"map_size": [map_width, map_height], "positions": [], "actions": [], "attention_weights": []}

        while not terminated:

            current_positions = self.env.get_positions()

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()],
            }

            if self.args.evaluate:
                episode_record["positions"].append([current_positions])

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            if self.args.agent == "diffusion_rnn" and self.args.use_bc_loss:
                actions, q_actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            elif self.args.agent in ["hgap", "updet"] and self.args.evaluate:
                actions, attention_weights = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            else:
                actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

            detached_actions = actions.detach()
            detached_q_actions = None

            if self.args.agent == "diffusion_rnn" and self.args.use_bc_loss:
                detached_q_actions = q_actions.detach()

            reward, terminated, env_info = self.env.step(detached_actions[0])
            episode_return += reward

            if self.args.agent in ["hgap", "updet"] and self.args.evaluate:
                detached_attention_weights = attention_weights.detach()
                post_transition_data = {
                    "actions": detached_actions,
                    "attention_weights": detached_attention_weights,
                    "reward": [(reward,)],
                    "terminated": [(terminated != env_info.get("episode_limit", False),)],
                }
                episode_record["actions"].append([detached_actions[0].tolist()])
                episode_record["attention_weights"].append([detached_attention_weights.tolist()])
            else:
                post_transition_data = {
                    "actions": detached_actions,
                    # "q_actions": detached_q_actions,
                    "reward": [(reward,)],
                    "terminated": [(terminated != env_info.get("episode_limit", False),)],
                }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()],
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        if self.args.agent == "diffusion_rnn" and self.args.use_bc_loss:
            actions, q_actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        elif self.args.agent in ["hgap", "updet"] and self.args.evaluate:
            actions, attention_weights = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        else:
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

        detached_actions = actions.detach()
        detached_q_actions = None

        if self.args.agent == "diffusion_rnn" and self.args.use_bc_loss:
            detached_q_actions = q_actions.detach()

        # self.batch.update({"actions": detached_actions, "q_actions": detached_q_actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode:
            print("Replay win rate:", cur_stats["battle_won"], flush=True)
            episode_record["positions"].append([self.env.get_positions()])

            for k, v in episode_record.items():
                episode_record[k] = np.array(v)
                if k != "map_size":
                    episode_record[k] = episode_record[k].squeeze(1)
                episode_record[k] = episode_record[k].tolist()

            import json
            json_object = json.dumps(episode_record, indent=4)

            os.makedirs(f"/home/chrislin/MADP/results/hgap_testing/{self.args.map_name}/records", exist_ok=True)
            with open(f"/home/chrislin/MADP/results/hgap_testing/{self.args.map_name}/records/{self.args.load_step}.json", "w") as outfile:
                outfile.write(json_object)


        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            if self.args.save_replay:
                print("Replay mean total reward:", np.mean(cur_returns), flush=True)
                print("Replay win rate:", cur_stats["battle_won"], flush=True)
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()
