"""
    Simple DDPG implementation inspired by CleanRL project
"""

import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.rlsp.agents.agent_helper import AgentHelper
from src.rlsp.agents.main import create_environment


class AutoResetWithSeed(gym.wrappers.AutoResetWrapper):
    def __init__(self, env: gym.Env, seed: int):
        super().__init__(env)
        self.seed = seed
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:

            new_obs, new_info = self.env.reset(self.seed)
            assert (
                "final_observation" not in new_info
            ), 'info dict cannot contain key "final_observation" '
            assert (
                "final_info" not in new_info
            ), 'info dict cannot contain key "final_info" '

            new_info["final_observation"] = obs
            new_info["final_info"] = info

            obs = new_obs
            info = new_info

        return obs, reward, terminated, truncated, info


# TODO: This network doesn't use any feature extractor. See SB3 implementation for more insight
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() \
                             + np.prod(env.single_action_space.shape), 64)
        self.fc2 = nn.Linear(64, 1)


    def forward(self, x, a):
        x = th.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Actor(nn.Module):
    def __init__(self, env, num_nodes: int, num_sfs: int):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 128)
        self.fc_mu = nn.Linear(128, np.prod(env.single_action_space.shape))
        self.before_softmax = nn.Sequential(self.fc1, nn.ReLU(), self.fc_mu)
        self.low = env.unwrapped.action_space.low
        self.high = env.unwrapped.action_space.high

        self.num_nodes = num_nodes
        self.num_softmax = self.num_nodes*num_sfs
        self.softmax_layers = [nn.Softmax(1) for _ in range(self.num_softmax)]
    
    def scale_action(self, action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [low, high] to [-1, 1]
        (no need for symmetric action space)

        :param action: Action to scale
        :return: Scaled action
        """
        return 2.0 * ((action - self.low) / (self.high - self.low)) - 1.0

    def unscale_action(self, scaled_action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [-1, 1] to [low, high]
        (no need for symmetric action space)

        :param scaled_action: Action to un-scale
        """
        return self.low + (0.5 * (scaled_action + 1.0) * (self.high - self.low))

    def forward(self, x):
        x = self.before_softmax(x)
        y = [self.softmax_layers[i](x[:, i*self.num_nodes:(i+1)*self.num_nodes]) for i in range(self.num_softmax)]
        x = th.concat(y, 1)
        return x


def make_env(agent_helper: AgentHelper, seed, idx, capture_video, run_name):
    def thunk():
        env = create_environment(agent_helper)
        env = AutoResetWithSeed(env, seed)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        #env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

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
        self.envs = gym.vector.SyncVectorEnv([make_env(self.agent_helper, self.agent_helper.sim_seed,
                                                0, False, self.agent_helper.config_dir)])
        assert isinstance(self.envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

        self.num_nodes = self.envs.envs[0].env.env_limits.MAX_NODE_COUNT
        self.num_sfs = self.envs.envs[0].env.env_limits.MAX_SERVICE_FUNCTION_COUNT
        self.num_sfcs = self.envs.envs[0].env.env_limits.MAX_SF_CHAIN_COUNT
        self.schedule_threshold = 0.1
        self.scheduling_accuracy = np.sqrt(np.finfo(np.float64).eps)

        self.actor = Actor(self.envs, self.num_nodes, self.num_sfs).to(self.device)
        self.qf1 = QNetwork(self.envs).to(self.device)
        self.qf1_target = QNetwork(self.envs).to(self.device)
        self.target_actor = Actor(self.envs, self.num_nodes, self.num_sfs).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.q_optimizer = optim.Adam(list(self.qf1.parameters()), lr=agent_helper.config['learning_rate'])
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=agent_helper.config['learning_rate'])

        self.batch_size = 100
        self.policy_frequency = 1
        self.n_action = self.envs.single_action_space.shape[-1]
        
        self.envs.single_observation_space.dtype = np.float32
        self.rb = ReplayBuffer(
            self.batch_size,
            self.envs.single_observation_space,
            self.envs.single_action_space,
            self.device,
            handle_timeout_termination=False,
        )

        


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
    

    def train(self, episodes: int):
        start_time = time.time()
        new_best_reward = float('-inf')

        # TRY NOT TO MODIFY: start the game
        obs, _ = self.envs.reset()
        for global_step in tqdm(range(self.agent_helper.episode_steps*episodes)):
            # ALGO LOGIC: put action logic here
            if global_step < self.agent_helper.config['nb_steps_warmup_critic']:
                actions = np.array([self.envs.single_action_space.sample() for _ in range(self.envs.num_envs)])
            else:
                with th.no_grad():
                    actions = self.actor(th.Tensor(obs).to(self.device))
                    actions = actions.cpu().numpy()
                    scaled_actions = self.actor.scale_action(actions)
                    scaled_actions += np.random.normal(
                        np.ones(self.n_action)*self.agent_helper.config['rand_mu'],
                        np.ones(self.n_action)*self.agent_helper.config['rand_sigma'])
                    scaled_actions.clip(-1, 1)
                    actions = self.actor.unscale_action(scaled_actions)
                    actions = actions.clip(self.envs.single_action_space.low,
                                            self.envs.single_action_space.high)
            
            # Post-processing: Threshold + Normalization
            actions = self.post_process_actions(actions)

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, dones, _, infos = self.envs.step(actions)

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            if "final_info" in infos:
                stats = infos["final_info"][0]
                ep_reward = float(f"{stats['episode']['r']:0.3f}")
                if ep_reward > new_best_reward:
                    new_best_reward = ep_reward
                    tqdm.write(f"global_step={global_step}, episodic_return={ep_reward}, NEW best reward")
                else:
                    tqdm.write(f"global_step={global_step}, episodic_return={ep_reward}")
                self.writer.add_scalar("charts/episodic_return", stats["episode"]["r"], global_step)
                self.writer.add_scalar("charts/episodic_length", stats["episode"]["l"], global_step)

            # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
            real_next_obs = next_obs.copy()
            for idx, d in enumerate(dones):
                if d:
                    real_next_obs[idx] = infos["final_observation"][idx]
            self.rb.add(obs, real_next_obs, actions, rewards, dones, infos)

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs

            # ALGO LOGIC: training.
            # Train frequency: (1, "episode")
            if global_step % self.agent_helper.episode_steps == 0:
                # TODO: train/test on/off for models
                if global_step > self.agent_helper.config['nb_steps_warmup_critic']:

                    # Multiple gradient steps
                    for _ in range(self.agent_helper.episode_steps):
                        data = self.rb.sample(self.batch_size)
                        with th.no_grad():
                            next_state_actions = self.target_actor(data.next_observations).clamp(-1, 1)
                            qf1_next_target = self.qf1_target(data.next_observations, next_state_actions)
                            next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) \
                                * self.agent_helper.config['gamma'] * (qf1_next_target).view(-1)

                        qf1_a_values = self.qf1(data.observations, data.actions).view(-1)
                        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)

                        # optimize the model
                        self.q_optimizer.zero_grad()
                        qf1_loss.backward()
                        self.q_optimizer.step()

                        if global_step % self.policy_frequency == 0:
                            actor_loss = -self.qf1(data.observations, self.actor(data.observations)).mean()
                            self.actor_optimizer.zero_grad()
                            actor_loss.backward()
                            self.actor_optimizer.step()

                            # update the target network
                            tau = self.agent_helper.config['target_model_update']
                            for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                            for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                        if global_step % 100 == 0:
                            self.writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                            self.writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                            self.writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                            #print("SPS:", int(global_step / (time.time() - start_time)))
                            self.writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        self.envs.close()
        self.writer.close()
    
    @th.no_grad()
    def predict(self, obs):
        return self.actor(th.Tensor(obs).to(self.device))

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
        assert action.shape[1] == self.num_nodes * self.num_sfcs * self.num_sfs * self.num_nodes, "wrong dimensions"

        # iterate through action array, select slice with probabilities belonging to one SF
        # processes probabilities (round low probs to 0, normalize), append and move on
        processed_action = []
        start_idx = 0
        for _ in range(self.num_nodes * self.num_sfcs * self.num_sfs):
            end_idx = start_idx + self.num_nodes
            probs = action[0, start_idx:end_idx]
            for _ in range(2):
                rounded_probs = [p if p >= self.schedule_threshold else 0 for p in probs]
                normalized_probs = np.array(self.normalize_scheduling_probabilities(rounded_probs))
                probs = normalized_probs

            # check that normalized probabilities sum up to 1 (accurate to specified float accuracy)
            assert (1 - sum(normalized_probs)) < np.sqrt(np.finfo(np.float64).eps)
            processed_action.extend(normalized_probs)
            start_idx += self.num_nodes

        assert len(processed_action) == action.shape[1]
        return np.reshape(np.array(processed_action), (1, -1))

