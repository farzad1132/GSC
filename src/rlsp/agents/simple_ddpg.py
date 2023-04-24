"""
    Simple DDPG implementation inspired by CleanRL project
"""

import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import pybullet_envs  # noqa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

from src.rlsp.agents.agent_helper import AgentHelper
from src.rlsp.agents.main import create_environment


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


def make_env(agent_helper: AgentHelper, seed, idx, capture_video, run_name):
    def thunk():
        env = create_environment(agent_helper)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

class SimpleDDPG:
    def __init__(self, agent_helper: AgentHelper) -> None:
        self.agent_helper = agent_helper
        self._set_seeds(self.agent_helper.sim_seed)

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

        self.actor = Actor(self.envs).to(self.device)
        self.qf1 = QNetwork(self.envs).to(self.device)
        self.qf1_target = QNetwork(self.envs).to(self.device)
        self.target_actor = Actor(self.envs).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.q_optimizer = optim.Adam(list(self.qf1.parameters()), lr=agent_helper.config['learning_rate'])
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=agent_helper.config['learning_rate'])

        self.batch_size = 100
        self.n_action = self.envs.single_action_space.shape[-1]
        
        self.envs.single_observation_space.dtype = np.float32
        self.rb = ReplayBuffer(
            self.batch_size,
            self.envs.single_observation_space,
            self.envs.single_action_space,
            self.device,
            handle_timeout_termination=True,
        )


    def _writer_setup(self):
        print(self.agent_helper.config_dir, "test")
        self.writer = SummaryWriter(f"runs/{self.agent_helper.config_dir}")

        """ 
        TODO: Add hyperparameter logging
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        ) """
    
    def _set_seeds(seed: int, cuda_deterministic: bool = True):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = cuda_deterministic
    

    def train(self, episodes: int):
        start_time = time.time()

        # TRY NOT TO MODIFY: start the game
        obs = self.envs.reset()
        for global_step in range(self.agent_helper.episode_steps*episodes):
            # ALGO LOGIC: put action logic here
            if global_step < self.agent_helper.config['nb_steps_warmup_critic']:
                actions = np.array([self.envs.single_action_space.sample() for _ in range(self.envs.num_envs)])
            else:
                with torch.no_grad():
                    actions = self.actor(torch.Tensor(obs).to(self.device))
                    actions += torch.normal(
                        torch.ones(self.n_action)*self.agent_helper.config['rand_mu'],
                        torch.ones(self.n_action)*self.agent_helper.config['rand_sigma'])
                    actions = actions.cpu().numpy().clip(self.envs.single_action_space.low,
                                                          self.envs.single_action_space.high)

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, dones, infos = self.envs.step(actions)

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            for info in infos:
                if "episode" in info.keys():
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    self.writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    self.writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    break

            # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
            real_next_obs = next_obs.copy()
            for idx, d in enumerate(dones):
                if d:
                    real_next_obs[idx] = infos[idx]["terminal_observation"]
            self.rb.add(obs, real_next_obs, actions, rewards, dones, infos)

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs

            # ALGO LOGIC: training.
            if global_step > self.agent_helper.config['nb_steps_warmup_critic']:
                data = self.rb.sample(self.batch_size)
                with torch.no_grad():
                    next_state_actions = self.target_actor(data.next_observations)
                    qf1_next_target = self.qf1_target(data.next_observations, next_state_actions)
                    next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) \
                        * self.agent_helper.config['gamma'] * (qf1_next_target).view(-1)

                qf1_a_values = self.qf1(data.observations, data.actions).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)

                # optimize the model
                self.q_optimizer.zero_grad()
                qf1_loss.backward()
                self.q_optimizer.step()

                if global_step % self.agent_helper.episode_steps == 0:
                    # TODO: Add multiple gradient steps feature
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
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    self.writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        self.envs.close()
        self.writer.close()

