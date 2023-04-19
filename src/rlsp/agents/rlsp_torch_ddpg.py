from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch as th
from gym import spaces
from stable_baselines3 import DDPG
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import ActionNoise, NormalActionNoise
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (BaseFeaturesExtractor,
                                                   FlattenExtractor,
                                                   create_mlp)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.ddpg import MlpPolicy
from stable_baselines3.td3.policies import Actor, TD3Policy
from torch import nn

from src.rlsp.agents.agent_helper import AgentHelper
from src.rlsp.agents.main import create_environment
from src.rlsp.envs.gym_env import GymEnv


class CustomActor(BasePolicy):
    """
    Actor network (policy) for TD3.
    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        **kwargs
    ):
        assert "squash_output" in kwargs, "squash_output is not given"
        squash_output = kwargs["squash_output"]

        
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=squash_output,
        )

        self._squash_output = squash_output

        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn

        action_dim = get_action_dim(self.action_space)
        actor_net = create_mlp(features_dim, action_dim, net_arch, activation_fn, squash_output=squash_output)
        

        assert "num_nodes" in kwargs, "num_nodes is not given"
        assert "num_sfs" in kwargs, "num_sfs is not given"

        self.before_softmax = nn.Sequential(*actor_net)

        self.num_softmax = kwargs["num_nodes"]*kwargs["num_sfs"]
        self.num_nodes = kwargs["num_nodes"]
        self.softmax_layers = [nn.Softmax(1) for _ in range(self.num_softmax)]

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
            )
        )
        return data

    def forward(self, obs: th.Tensor) -> th.Tensor:
        features = self.extract_features(obs, self.features_extractor)
        x = self.before_softmax(features)
        y = [self.softmax_layers[i](x[:, i*self.num_nodes:(i+1)*self.num_nodes]) for i in range(self.num_softmax)]
        """ for i in range(self.num_softmax):
            x[:, i*self.num_nodes:(i+1)*self.num_nodes] = \
                self.softmax_layers[i](x[:, i*self.num_nodes:(i+1)*self.num_nodes]) """
        x = th.concat(y, 1)
        return x

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # Note: the deterministic deterministic parameter is ignored in the case of TD3.
        #   Predictions are always deterministic.
        return self(observation)


class CustomMlpPolicy(MlpPolicy):
    def __init__(self, observation_space: spaces.Space, action_space: spaces.Space,
            lr_schedule: Schedule,
            net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
            activation_fn: Type[nn.Module] = nn.ReLU,
            features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            normalize_images: bool = True, optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            n_critics: int = 2, share_features_extractor: bool = False,
            **kwargs):
        
        self.custom_kwagrs = kwargs
        # TODO: Refactor
        squash_output = False
        self.custom_kwagrs["squash_output"] = squash_output

        super().__init__(observation_space, action_space, lr_schedule, net_arch, activation_fn,
        features_extractor_class, features_extractor_kwargs, normalize_images, optimizer_class,
        optimizer_kwargs, n_critics, share_features_extractor)

        self._squash_output = squash_output     
        

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return CustomActor(**actor_kwargs, **self.custom_kwagrs).to(self.device)

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # TODO: go back to the original state
        test =  super()._predict(observation, deterministic)
        return test


class CustomDDPG(DDPG):
    def __init__(self, policy: Union[str, Type[TD3Policy]], env: Union[GymEnv, str],
    learning_rate: Union[float, Schedule] = 0.001, buffer_size: int = 1000000,
    learning_starts: int = 100, batch_size: int = 100, tau: float = 0.005, gamma: float = 0.99,
    train_freq: Union[int, Tuple[int, str]] = (1, "episode"), gradient_steps: int = -1,
    action_noise: Optional[ActionNoise] = None, replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
    replay_buffer_kwargs: Optional[Dict[str, Any]] = None, optimize_memory_usage: bool = False,
    tensorboard_log: Optional[str] = None, policy_kwargs: Optional[Dict[str, Any]] = None,
    verbose: int = 0, seed: Optional[int] = None, device: Union[th.device, str] = "auto",
    _init_setup_model: bool = True):

        super().__init__(policy, env, learning_rate, buffer_size, learning_starts, batch_size,
        tau, gamma, train_freq, gradient_steps, action_noise, replay_buffer_class, replay_buffer_kwargs,
        optimize_memory_usage, tensorboard_log, policy_kwargs, verbose, seed, device, _init_setup_model)

        assert "num_nodes" in self.policy_kwargs, "num_nodes is not provided"
        assert "num_sfs" in self.policy_kwargs, "num_sfs is not provided"
        assert "num_sfcs" in self.policy_kwargs, "num_sfcs is not provided"
        self.num_nodes = self.policy_kwargs["num_nodes"]
        self.num_sfs = self.policy_kwargs["num_sfs"]
        self.num_sfcs = self.policy_kwargs["num_sfcs"]
        self.schedule_threshold = 0.1

        self.scheduling_accuracy = np.sqrt(np.finfo(np.float64).eps)

    def _sample_action(self, learning_starts: int, action_noise: Optional[ActionNoise] = None, n_envs: int = 1)\
         -> Tuple[np.ndarray, np.ndarray]:
        action, scaled_action = super()._sample_action(learning_starts, action_noise, n_envs)
        #action = buffer_action = self.post_process_actions(action)
        action = self.post_process_actions(action)
        buffer_action = self.policy.scale_action(action)
        return action, buffer_action
    
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


class TorchDDPG:
    def __init__(self, agent_helper: AgentHelper):
        self.agent_helper = agent_helper
        self.create(create_environment(agent_helper))

    
    def create(self, env: GymEnv):
        env = Monitor(env, filename=self.agent_helper.config_dir)

        self.model = CustomDDPG(
            policy=CustomMlpPolicy,
            env=env,
            tensorboard_log="./graph",
            policy_kwargs={
                "num_sfs": env.env_limits.MAX_SERVICE_FUNCTION_COUNT,
                "num_nodes": env.env_limits.MAX_NODE_COUNT,
                "num_sfcs": env.env_limits.MAX_SF_CHAIN_COUNT,
                "net_arch": {
                    "pi": self.agent_helper.config['actor_hidden_layer_nodes'],
                    "qf": self.agent_helper.config['critic_hidden_layer_nodes']
                }
            },
            learning_starts=self.agent_helper.config['nb_steps_warmup_critic'],
            gamma=self.agent_helper.config['gamma'],
            tau=self.agent_helper.config['target_model_update'],
            action_noise=NormalActionNoise(
                mean=self.agent_helper.config['rand_mu'],
                sigma=self.agent_helper.config['rand_sigma']
            ),
            batch_size=100,
            buffer_size=self.agent_helper.config['mem_limit'],
            train_freq=(1, "episode"),
            gradient_steps=-1,
            #learning_rate=agent_helper.config['learning_rate']
            #learning_rate=linear_schedule(1e-3)
            learning_rate=1e-3,
            #learning_rate=exp_decay(init=1e-3, end=1e-4)
            verbose=1
        )
    
    def train(self, episodes: int):

        # TODO: Add callback to print training episode rewards
        """
            callback=EvalCallback(
                eval_env=eval_env,
                eval_freq=400,
                n_eval_episodes=1
            )
        """

        self.model.learn(
            total_timesteps=self.agent_helper.episode_steps*episodes,
            progress_bar=True
        )

