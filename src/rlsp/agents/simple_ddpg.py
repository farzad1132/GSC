"""
    Simple DDPG implementation inspired by CleanRL project
"""

import random
import time
from copy import deepcopy

import gym
import numpy as np
import torch as th
import torch.nn.functional as F
import torch.optim as optim
import wandb
from stable_baselines3.common.buffers import DictReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.rlsp.agents.agent_helper import AgentHelper
from src.rlsp.agents.models import Actor, QNetwork, GSCActor, GSCCritic
from src.rlsp.utils.util_functions import (graph_to_dict, simple_make_env,
                                           torch_stack_to_graph_batch)


class SimpleDDPG:
    def __init__(self, agent_helper: AgentHelper) -> None:
        self.agent_helper = agent_helper
        self._set_seeds(self.agent_helper.seed)

        # TODO: add GPU support
        #self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        self.device = "cpu"

        """ 
        TODO: Consider adding tracking functionality
        if args.track:
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=vars(args),
                name=run_name,
                monitor_gym=True,
                save_code=True,
            ) """
        
        self._writer_setup()
        self.env = simple_make_env(agent_helper)
        assert isinstance(self.env.action_space, gym.spaces.Box), "only continuous action space is supported"

        self.num_nodes = self.agent_helper.env.env_limits.MAX_NODE_COUNT
        self.num_sfs = self.agent_helper.env.env_limits.MAX_SERVICE_FUNCTION_COUNT
        self.num_sfcs = self.agent_helper.env.env_limits.MAX_SF_CHAIN_COUNT
        self.schedule_threshold = 0.1
        self.scheduling_accuracy = np.sqrt(np.finfo(np.float64).eps)

        self._init_networks()

        self.batch_size = self.agent_helper.config["batch_size"]
        self.policy_frequency = 1
        self.n_action = self.env.action_space.shape[-1]
        
        self.rb = DictReplayBuffer(
            buffer_size=self.agent_helper.config["mem_limit"],
            action_space=self.env.action_space,
            observation_space=self.env.observation_space,
            device=self.device,
            handle_timeout_termination=False,
        )

        self.ep_counter = 0
        self.avg_rew_ep = 0
        
    def _init_networks(self):
        if self.agent_helper.config["graph_mode"]:
            self.actor = GSCActor(self.agent_helper).to(self.device)
            self.target_actor = GSCActor(self.agent_helper).to(self.device)
            self.qf1 = GSCCritic(self.agent_helper).to(self.device)
            self.qf1_target = GSCCritic(self.agent_helper).to(self.device)
        else:
            self.actor = Actor(self.agent_helper).to(self.device)
            self.target_actor = Actor(self.agent_helper).to(self.device)
            self.qf1 = QNetwork(self.agent_helper).to(self.device)
            self.qf1_target = QNetwork(self.agent_helper).to(self.device)
        
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.q_optimizer = optim.Adam(list(self.qf1.parameters()), lr=self.agent_helper.config['learning_rate'])
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=self.agent_helper.config['learning_rate'])

    def _writer_setup(self):
        self.writer = SummaryWriter(f"runs/{self.agent_helper.test}")

        """ 
        TODO: Add hyperparameter logging
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        ) """
    
    def _set_seeds(self, seed: int, cuda_deterministic: bool = True):
        random.seed(seed)
        np.random.seed(seed)
        th.manual_seed(seed)
        th.backends.cudnn.deterministic = cuda_deterministic
    
    def _choose_action(self, obs, global_step):
        if global_step < self.agent_helper.config['nb_steps_warmup_critic']:
            actions = self.env.action_space.sample()
        else:
            with th.no_grad():
                if self.agent_helper.config["graph_mode"]:
                    for value in obs.node_feature.values():
                        value.to(self.device) 
                    actions = self.actor(obs)
                else:
                    actions = self.actor(th.Tensor(obs).view(1, -1).to(self.device))
                actions = actions.cpu().numpy()
                scaled_actions = self.actor.scale_action(actions)
                scaled_actions += np.random.normal(
                    np.ones(self.n_action)*self.agent_helper.config['rand_mu'],
                    np.ones(self.n_action)*self.agent_helper.config['rand_sigma'])
                scaled_actions.clip(-1, 1)
                actions = self.actor.unscale_action(scaled_actions)
                actions = actions.clip(self.env.action_space.low, self.env.action_space.high)
        
        return np.squeeze(actions)

    def _set_models_in_train_mode(self):
        self.actor.train()
        self.target_actor.train()
        self.qf1.train()
        self.qf1_target.train()
    

    def _update_critic(self, next_obs, cur_obs, dones, rewards, actions):
        with th.no_grad():
            next_state_actions = self.target_actor(next_obs).clamp(-1, 1)
            qf1_next_target = self.qf1_target(next_obs, next_state_actions)
            next_q_value = rewards.flatten() + (1 - dones.flatten()) \
                * self.agent_helper.config['gamma'] * (qf1_next_target).view(-1)
        
        qf1_a_values = self.qf1(cur_obs, actions).view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)

        # optimize the model
        self.q_optimizer.zero_grad()
        qf1_loss.backward()
        self.q_optimizer.step()

        return qf1_a_values, qf1_loss
    
    def _update_actor(self, cur_obs):
        actor_loss = -self.qf1(cur_obs, self.actor(cur_obs)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss
    
    def _update_target_networks(self):
        tau = self.agent_helper.config['target_model_update']
        for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def _convert_obs(self, observations, next_observations):
        """
            This function is useful to adapt obs when we want to operate in graph mode
        """
        if self.agent_helper.config["graph_mode"]:
            next_obs = torch_stack_to_graph_batch(next_observations)
            cur_obs = torch_stack_to_graph_batch(observations)
        else:
            next_obs = next_observations
            cur_obs = observations

        return cur_obs, next_obs

    def _update_episode_metrics(self, infos, new_best_reward, global_step):
        if "final_info" in infos:
            ep_reward = float(f"{infos['episode']['r']:0.3f}")
            self.ep_counter += 1
            self.avg_rew_ep = self.avg_rew_ep + (1/self.ep_counter)*(ep_reward-self.avg_rew_ep)
            if ep_reward > new_best_reward:
                new_best_reward = ep_reward
                tqdm.write(f"global_step={global_step}, episodic_return={ep_reward}, average_return={self.avg_rew_ep:0.2f} NEW best reward")
            else:
                tqdm.write(f"global_step={global_step}, episodic_return={ep_reward}, average_return={self.avg_rew_ep:0.2f}")
            self.writer.add_scalar("charts/episodic_return", infos["episode"]["r"], global_step)
            self.writer.add_scalar("charts/episodic_length", infos["episode"]["l"], global_step)
        
        return new_best_reward
    
    def _update_loss_metrics(self, qf1_loss, actor_loss, qf1_a_values, start_time, global_step):
        self.writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
        self.writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
        self.writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
        #print("SPS:", int(global_step / (time.time() - start_time)))
        self.writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
    
    def _store_transition_in_buffer(self, obs, next_obs, actions, rewards, dones, infos):
        real_next_obs = deepcopy(next_obs)
        if dones:
            real_next_obs = infos["final_observation"]
        self.rb.add(graph_to_dict(obs), graph_to_dict(real_next_obs), actions, rewards, dones, infos)
    

    def train(self, episodes: int):
        start_time = time.time()
        new_best_reward = float('-inf')

        # TRY NOT TO MODIFY: start the game
        obs, _ = self.env.reset()
        for global_step in tqdm(range(self.agent_helper.episode_steps*episodes)):
            # ALGO LOGIC: put action logic here
            actions = self._choose_action(obs, global_step)
            
            # Post-processing: Threshold + Normalization
            # actions = self.post_process_actions(np.squeeze(actions))

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, dones, _, infos = self.env.step(actions)

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            new_best_reward = self._update_episode_metrics(infos, new_best_reward, global_step)

            # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
            self._store_transition_in_buffer(obs, next_obs, actions, rewards, dones, infos)

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs

            # ALGO LOGIC: training.
            # Train frequency: (1, "episode")
            if global_step % self.agent_helper.episode_steps == self.agent_helper.episode_steps-1:
                if global_step >= self.agent_helper.config['nb_steps_warmup_critic']-1:

                    # Multiple gradient steps
                    for _ in range(self.agent_helper.episode_steps):
                        data = self.rb.sample(self.batch_size)

                        cur_obs, next_obs = self._convert_obs(data.observations, data.next_observations)

                        qf1_a_values, qf1_loss = self._update_critic(next_obs, cur_obs, data.dones, data.rewards, data.actions)

                        if global_step % self.policy_frequency == 0:
                            actor_loss = self._update_actor(cur_obs)

                            self._update_target_networks()

                        if global_step % 100 == 0:
                            self._update_loss_metrics(qf1_loss, actor_loss, qf1_a_values, start_time, global_step)

        self.env.close()
        self.writer.close()
    
    @th.no_grad()
    def predict(self, obs):
        if self.agent_helper.config["graph_mode"]:
            return self.actor(obs)
        else:
            return self.actor(th.tensor(obs, dtype=th.float32).view(1, -1))

    def normalize_scheduling_probabilities(self, input_list: list) -> list:

        output_list = []
        # to handle the empty list case, we just return the empty list back
        if len(input_list) == 0:
            return output_list

        offset = 1 - sum(input_list)

        # a list with all elements 0, will be equally distributed to sum-up to 1.
        # sum can also be 0 if some elements of the list are negative.
        # In our case the list contains probabilities and they are not supposed to be negative, hence the case won't arise
        if sum(input_list) == 0:
            output_list = [round(1 / len(input_list), 10)] * len(input_list)

        # Because of floating point precision (.59 + .33 + .08) can be equal to .99999999
        # So we correct the sum only if the absolute difference is more than a tolerance(0.000000014901161193847656)
        else:
            if abs(offset) > self.scheduling_accuracy:
                sum_list = sum(input_list)
                # we divide each number in the list by the sum of the list, so that Prob. Distribution is approx. 1
                output_list = [round(prob / sum_list, 10) for prob in input_list]
            else:
                output_list = input_list.copy()

        # 1 - sum(output_list) = the diff. by which the elements of the list are away from 1.0, could be +'ive /-i've
        new_offset = 1 - sum(output_list)
        if new_offset != 0:
            i = 0
            while output_list[i] + new_offset < 0:
                i += 1
            # the difference is added/subtracted from the 1st element of the list, which is also rounded to 2 decimal points
            output_list[i] = output_list[i] + new_offset
        assert abs(1 - sum(output_list)) < self.scheduling_accuracy, "Sum of list not equal to 1.0"
        return output_list

    def post_process_actions(self, action):
        assert action.shape[0] == self.num_nodes * self.num_sfcs * self.num_sfs * self.num_nodes, "wrong dimensions"

        # iterate through action array, select slice with probabilities belonging to one SF
        # processes probabilities (round low probs to 0, normalize), append and move on
        processed_action = []
        start_idx = 0
        for _ in range(self.num_nodes * self.num_sfcs * self.num_sfs):
            end_idx = start_idx + self.num_nodes
            probs = action[start_idx:end_idx]
            for _ in range(2):
                rounded_probs = [p if p >= self.schedule_threshold else 0 for p in probs]
                normalized_probs = np.array(self.normalize_scheduling_probabilities(rounded_probs))
                probs = normalized_probs

            # check that normalized probabilities sum up to 1 (accurate to specified float accuracy)
            assert (1 - sum(normalized_probs)) < np.sqrt(np.finfo(np.float64).eps)
            processed_action.extend(normalized_probs)
            start_idx += self.num_nodes

        assert len(processed_action) == action.shape[0]
        return np.array(processed_action)

