# -*- coding: utf-8 -*-
"""
Gym envs representing the coordination-simulation from
REAL NFV https://github.com/RealVNF/coordination-simulation


For help on "Implementing New Environments" see:
https://github.com/openai/gym/blob/master/gym/core.py
https://github.com/rll/rllab/blob/master/docs/user/implement_env.rst

"""
import inspect
import logging
from typing import Tuple

import gymnasium as gym
import numpy as np
from gymnasium.utils import seeding

from coordsim.reader.builders import network_builder
from coordsim.reader.reader import get_sf, get_sfc, network_diameter
from spinterface import SimulatorInterface, SimulatorState
from src.rlsp.envs.environment_limits import EnvironmentLimits
from src.rlsp.envs.simulator_wrapper import SimulatorWrapper
from src.rlsp.utils.constants import SUPPORTED_OBJECTIVES

logger = logging.getLogger(__name__)


class GymEnv(gym.Env):
    """
    Gym Environment class, which abstracts the coordination simulator.
    """
    current_simulator_state: SimulatorState
    simulator: SimulatorInterface = ...
    simulator_wrapper: SimulatorWrapper = ...

    metadata = {'render_modes': ['human']}

    def __init__(self, agent_config, simulator, network_file, service_file, seed=None, sim_seed=None,
                 render_mode: str = None):

        self.network_file = network_file
        self.agent_config = agent_config
        self.simulator = simulator
        self.sim_seed = sim_seed
        self.simulator_wrapper = None
        self.current_simulator_state = None
        self.render_mode = None

        self.last_succ_flow = 0
        self.last_drop_flow = 0
        self.last_gen_flow = 0
        self.run_count = 0

        """ self.np_random = np.random.RandomState()
        self.seed(seed) """

        self.network, _, _ = network_builder(self.network_file, self.simulator.config)
        self.network_diameter = network_diameter(self.network)
        self.sfc_list = get_sfc(service_file)
        self.sf_list = get_sf(service_file)
        self.env_limits = EnvironmentLimits(
            num_nodes=len(self.network.nodes),
            sfc_list=self.sfc_list,
            node_obs_space_len=len(agent_config['observation_space']),
            link_obs_space_len=len(agent_config["link_observation_space"]),
            graph_mode=self.agent_config["graph_mode"]
        )
        self.min_delay, self.max_delay = self.min_max_delay()
        self.action_space = self.env_limits.action_space
        self.observation_space = self.env_limits.observation_space
        logger.info('Observation space: ' + str(self.agent_config['observation_space']))

        # order of permutation for shuffling state
        self.permutation = None

        # related to objective/reward
        self.objective = self.agent_config['objective']
        self.target_success = self.agent_config['target_success']
        self.soft_deadline = self.agent_config['soft_deadline']
        # start at the best case with the moving average to encourage high standards!
        self.ewma_flows = 1
        # self.ewma_delay = 0

    def update_ewma(self, metric, value, weight=0.5):
        """
        Update the exponentially weighted moving average (EMWA) for the given metric with the given value and weight
        """
        assert metric in {'flows', 'delay'}, f"Unsupported metric {metric} for EWMA."
        if metric == 'flows':
            self.ewma_flows = weight * value + (1 - weight) * self.ewma_flows
        elif metric == 'delay':
            self.ewma_delay = weight * value + (1 - weight) * self.ewma_delay

    def min_max_delay(self):
        """Return the min and max e2e-delay for the current network topology and SFC. Independent of capacities."""
        vnf_delays = sum([sf['processing_delay_mean'] for sf in self.sf_list.values()])
        # min delay = sum of VNF delays (corresponds to all VNFs at ingress)
        min_delay = vnf_delays
        # max delay = VNF delays + num_vnfs * network diameter (corresponds to max distance between all VNFs)
        max_delay = vnf_delays + len(self.sf_list) * self.network_diameter
        logger.info(f"min_delay: {min_delay}, max_delay: {max_delay}, diameter: {self.network_diameter}")
        return min_delay, max_delay

    def reset(self, seed: int = None, **kwargs):
        """
        Resets the state of the envs, returning an initial observation.
        Outputs
        -------
        observation : the initial observation of the space.
        (Initial reward is assumed to be 0.)

        """
        if seed is not None:
            super().reset(seed=seed)
        else:
            seed = self.np_random.integers(0, np.iinfo(np.int32).max, dtype=np.int32)

        """ if self.sim_seed is None:
            simulator_seed = self.np_random.integers(0, np.iinfo(np.int32).max, dtype=np.int32)
        else:
            simulator_seed = self.sim_seed """
        logger.debug(f"Simulator seed is {seed}")
        self.simulator_wrapper = SimulatorWrapper(self.simulator, self.env_limits, self.agent_config["graph_mode"],
                                                  self.agent_config['observation_space'])

        self.last_succ_flow = 0
        self.last_drop_flow = 0
        self.last_gen_flow = 0
        self.run_count = 0

        # self.ewma_flows = 0
        # self.ewma_delay = self.network_diameter

        # to get initial state and instantiate
        obs, self.current_simulator_state = self.simulator_wrapper.init(seed)

        # permute state and save permutation for reversing action later
        if self.agent_config['shuffle_nodes']:
            obs, permutation = self.simulator_wrapper.permute_node_order(obs)
            self.permutation = permutation

        return obs, {}

    def step(self, action: np.ndarray) -> Tuple[object, float, bool, dict]:

        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the environment

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether episode has ended, in which case further step calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        done = False
        self.run_count += 1
        logger.debug(f"Action array (NN output + noise, normalized): {action}")

        # reverse action order using permutation from previous state shuffle
        if self.agent_config['shuffle_nodes']:
            action = self.simulator_wrapper.reverse_node_permutation(action, self.permutation)
            self.permutation = None

        # apply reversed action, calculate reward
        obs, self.current_simulator_state = self.simulator_wrapper.apply(action)
        reward = self.calculate_reward(self.current_simulator_state)

        # then shuffle new state again and save new permutation
        if self.agent_config['shuffle_nodes']:
            obs, permutation = self.simulator_wrapper.permute_node_order(obs)
            self.permutation = permutation
        if self.run_count == self.agent_config['episode_steps']:
            done = True
            self.run_count = 0

        logger.debug(f"NN input (observation): {obs}")
        return obs, reward, done, False, {}

    def render(self, mode='cli'):
        """Renders the envs.
        Implementation required by Gym.
        """
        assert mode in ['human']

    def reward_func_repr(self):
        """returns a string describing the reward function"""
        return inspect.getsource(self.calculate_reward)

    def get_flow_reward(self, simulator_state):
        """Calculate and return both success ratio and flow reward"""
        # calculate ratio of successful flows in the last run
        cur_succ_flow = simulator_state.network_stats['run_successful_flows']
        cur_drop_flow = simulator_state.network_stats['run_dropped_flows']
        succ_ratio = 0
        flow_reward = 0
        if cur_succ_flow + cur_drop_flow > 0:
            succ_ratio = cur_succ_flow / (cur_succ_flow + cur_drop_flow)
            # use this for flow reward instead of succ ratio to use full [-1, 1] range rather than just [0,1]
            flow_reward = (cur_succ_flow - cur_drop_flow) / (cur_succ_flow + cur_drop_flow)
        return succ_ratio, flow_reward

    def get_delay_reward(self, simulator_state, succ_ratio):
        """Return avg e2e delay and delay reward"""
        # get avg. e2e delay in last run and calculate delay reward
        delay = simulator_state.network_stats['run_avg_end2end_delay']
        # ensure the delay is at least min_delay/VNF delay. may be lower if no flow was successful
        delay = max(delay, self.min_delay)
        # require some flows to be successful for delay to have any meaning; init to -1
        if succ_ratio == 0:
            delay_reward = -1
        else:
            # subtract from min delay = vnf delay;
            # to disregard VNF delay, which cannot be affected and may already be larger than the diameter
            delay_reward = ((self.min_delay - delay) / self.network_diameter) + 1
            delay_reward = np.clip(delay_reward, -1, 1)
        return delay, delay_reward

    def get_node_reward(self, simulator_state):
        """Return reward based on the number of used nodes: Fewer = better"""
        # calculate reward for number of used nodes (less = better; lower energy)
        # only consider nodes with any capacity; also scale to [-1,1]
        # num_nodes_available = len([v for v in simulator_state.network['nodes'] if v['resource'] > 0])
        # num_nodes_used = len([v for v in simulator_state.network['nodes'] if v['used_resources'] > 0])
        # better: consider all nodes; whether with cap or without
        num_nodes = len(simulator_state.network['nodes'])
        num_nodes_used = len(simulator_state.placement.keys())
        # similar to delay, require some flows to be successful
        # if succ_ratio == 0:
        #     nodes_reward = -1
        # else:
        nodes_reward = 2 * (-num_nodes_used / num_nodes) + 1
        return nodes_reward

    def get_node_reward_shaped(self, simulator_state):
        """
        Calculate shaped node reward, where the agent is rewarded for placing fewer instances on a node, even before
        the node is completely free and unused.
        Without shaped reward, the agent learns nothing! Because it is extremely unlikely that it schedules no traffic
        to any VNF to a node, thus it always gets -1 reward and learns nothing. With shaped reward, it does.
        Nodes are counted as partially used [0.5,1] depending on their number of instances rather than just {0,1}.
        """
        num_nodes = len(simulator_state.network['nodes'])
        num_sfs = len(self.sf_list.keys())
        num_nodes_used = 0
        for node, placed_sfs in simulator_state.placement.items():
            if len(placed_sfs) > 0:
                # scale to 0.5 if just 1 SF is placed up to 1 if all SFs are placed on the node; add to total
                # https://stats.stackexchange.com/a/281164/68084
                num_nodes_used += ((len(placed_sfs) - 1) / (num_sfs - 1)) * 0.5 + 0.5

        return 2 * (-num_nodes_used / num_nodes) + 1

    def get_instance_reward(self, simulator_state):
        """Return instance reward based on the number of placed instances (fewer = better)"""
        # similar reward based on number of instances (less = better; less licensing costs)
        num_nodes = len(simulator_state.network['nodes'])
        num_sfs = len(self.sf_list.keys())
        max_num_instances = num_nodes * num_sfs
        num_instances = len([inst for inst_list in simulator_state.placement.values() for inst in inst_list])
        # if succ_ratio == 0:
        #     nodes_reward = -1
        # else:
        instance_reward = 2 * (-num_instances / max_num_instances) + 1
        return instance_reward

    def calculate_reward(self, simulator_state: SimulatorState) -> float:
        """
        Calculate reward per step based on the chosen objective.

        :param simulator_state: Current simulator state
        :return: The agent's reward
        """
        succ_ratio, flow_reward = self.get_flow_reward(simulator_state)
        delay, delay_reward = self.get_delay_reward(simulator_state, succ_ratio)
        nodes_reward = self.get_node_reward_shaped(simulator_state)
        instance_reward = self.get_instance_reward(simulator_state)

        # combine rewards based on chosen objective (and weights)
        if self.objective == 'prio-flow':
            nodes_reward = 0
            instance_reward = 0
            # prioritize flow reward and only optimize delay when the flow success target is met
            # if the target is set to 'auto', use the EWMA instead
            target = self.target_success
            if self.target_success == 'auto':
                # hard-code "safety" value/thershold of 90%, could sth else
                target = 0.9 * self.ewma_flows
                # update exponentially weighted moving average (EWMA)
                self.update_ewma('flows', succ_ratio)
            # as long as the target is not met, ignore delay and set it to -1
            if succ_ratio < target:
                delay_reward = -1

        elif self.objective == 'soft-deadline':
            nodes_reward = 0
            instance_reward = 0
            # ensure flows reach their soft deadline as primary objective and ignore flow success until then
            if delay > self.soft_deadline:
                flow_reward = -1
            # after reaching the soft deadline, optimize flow success rather than further optimizing delay
            else:
                # keep delay reward constant
                delay_reward = np.clip(-self.soft_deadline / self.network_diameter, -1, 1)

        elif self.objective == 'soft-deadline-exp':
            # example of more complex utility function, where the utility drops of exponentially if the avg. e2e delay
            # exceeds the soft deadline
            # utility function U(succ_ratio, delay) = succ_ratio * U_d(delay)
            # U_d = constant 1 until deadline, then exp dropoff; then 0
            # set both as delay reward; and flow and node reward to 0
            flow_reward = 0
            nodes_reward = 0
            instance_reward = 0
            # calc U_d (delay utility)
            delay_utility = 1
            if delay > self.soft_deadline:
                # drops of from 1 starting at the soft deadline down to 0 for configured dropoff duration
                delay_utility = -np.log10((1 / self.agent_config['dropoff']) * (delay - self.soft_deadline))
                # clip to >=0 in case delay even exceeds the acceptable extra delay (would otherwise be negative)
                delay_utility = np.clip(delay_utility, 0, 1)

            # multiply with success ratio (not reward!; needs to be in [0,1]) to get total utility; set as delay reward
            delay_reward = succ_ratio * delay_utility

        elif self.objective == 'weighted':
            # weight all objectives as configured before summing them
            flow_reward *= self.agent_config['flow_weight']
            delay_reward *= self.agent_config['delay_weight']
            nodes_reward *= self.agent_config['node_weight']
            instance_reward *= self.agent_config['instance_weight']

        else:
            raise ValueError(f"Unexpected objective {self.objective}. Must be in {SUPPORTED_OBJECTIVES}.")

        # calculate and return the sum, ie, total reward
        total_reward = flow_reward + delay_reward + nodes_reward + instance_reward
        assert -4 <= total_reward <= 4, f"Unexpected total reward: {total_reward}."

        logger.debug(f"Flow reward: {flow_reward}, success ratio: {succ_ratio}, target: {self.target_success}")
        logger.debug(f"Delay reward: {delay_reward}, delay: {delay}, target: {self.soft_deadline}")
        logger.debug(f"Nodes reward: {nodes_reward}")
        logger.debug(f"Instance reward: {instance_reward}")
        logger.debug(f"Total reward: {total_reward}, flow reward: {flow_reward}, delay reward: {delay_reward},"
                     f"objective: {self.objective}")

        return total_reward
