# -*- coding: utf-8 -*-
"""
This module contains the SimulatorWrapper class.
"""
import logging
from copy import deepcopy
from typing import Tuple

import networkx as nx
import numpy as np
import torch as th
from torch_geometric.utils import from_networkx

from siminterface.simulator import Simulator
from spinterface import SimulatorAction, SimulatorState
from src.rlsp.envs.action_norm_processor import ActionScheduleProcessor
from src.rlsp.envs.environment_limits import EnvironmentLimits

logger = logging.getLogger(__name__)


class SimulatorWrapper:
    """
    Wraps a simulator which implements the SimulatorInterface to the gym interface.

    Instead of interacting with Interface Object,
     this class awaits numpy arrays which are used in gym environments.
    """

    def __init__(self, simulator: Simulator, env_limits: EnvironmentLimits,
                graph_mode: bool, observations_space=('ingress_traffic', 'node_load')):
        self.simulator = simulator
        self.env_limits = env_limits
        self.sfc_dict = {}
        self.node_map = {}
        self.sfc_map = {}
        self.sf_map = {}
        self.observations_space = observations_space
        self.graph_mode = graph_mode

    def init(self, seed) -> Tuple[object, SimulatorState]:
        """Creates a new simulation environment.

        Reuses network_file, service_functions_file from object scope.
        Creates mapping from string identifier to integer IDs for nddes, SFC, and sf
        Calculates shortest paths array for network graph.

        Parameters
        ----------
        seed : int
            The seed value enables reproducible gym environments respectively
            reproducible simulator environments. This value should initialize
            the random number generator used by the simulator when executing
            randomized functions.

        Returns
        -------
        vectorized_state: np.ndarray

        state: SimulatorState
        """
        logger.debug("INIT Simulator")
        # get initial state
        init_state = self.simulator.init(seed)

        # create a mapping such that every node, SF and SFC has a fixed array position
        # this is important for the RL
        # create also an inverted mapping
        node_index = 0
        sfc_index = 0
        sf_index = 0

        for node in init_state.network['nodes']:
            self.node_map[node['id']] = node_index
            node_index = node_index + 1

        self.sfc_dict = init_state.sfcs

        for sfc in init_state.sfcs:
            self.sfc_map[sfc] = sfc_index
            sfc_index = sfc_index + 1

        for service_function in init_state.service_functions:
            self.sf_map[service_function] = sf_index
            sf_index = sf_index + 1

        obs = self._parse_state_as_graph(init_state) if self.graph_mode else self._parse_state(init_state)
        return obs, init_state

    def add_placement_recursive(self, source_node, sf_id, sfc, schedule, placement):
        """
        recursively adds service functions to placement based on schedule and virtual possible traffic.
        Only needed service functions are contained in the placement.
        Initially call this function with sf_id 0 and the ingress node as node_source

        Parameters
        ----------
        source_node: str
            The current node to process traffic from.
        sf_id: int
            The current index of sf in the given sfc.
            Determines the end in recursion if the last sf is reached.
        sfc
            The service function chain defines the chain of service functions.
        schedule
            The schedule dictionary in form of schedule from SimulatorAction.
        placement
            A dict with default sets for each node where the service functions are added.
        """
        sf: str = self.sfc_dict[sfc][sf_id]

        for target_node, weight in schedule[source_node][sfc][sf].items():
            # test schedule weight threshold
            if weight > 0:
                # add current (node, sf) tuple to placement
                placement[target_node].add(sf)
                # are there following service functions?
                if sf_id + 1 < len(self.sfc_dict[sfc]):
                    # call this function for target node and next service function
                    self.add_placement_recursive(target_node, sf_id + 1, sfc, schedule, placement)

    def apply(self, action_array: np.ndarray) -> Tuple[object, SimulatorState]:
        """
        Encapsulates the simulators apply method to use the gym interface

        Creates a SimulatorAction object from the agent's return array.
        Applies it to the simulator, translates the returning SimulatorState to an array and returns it.

        Parameters
        ----------
        action_array: np.ndarray

        Returns
        -------
        vectorized_state: dict
        state: SimulatorState
        """
        logger.debug(f"Action array (NN output + noise, normalized): {action_array}")
        num_nodes = self.simulator.network.number_of_nodes()
        mask = np.zeros(self.env_limits.scheduling_shape)
        mask[:num_nodes, :, :, :num_nodes] = 1
        mask = mask.flatten()
        action_array = action_array[mask == 1]
        
        scheduling = np.reshape(action_array, (num_nodes, self.env_limits.MAX_SF_CHAIN_COUNT,
                                self.env_limits.MAX_SERVICE_FUNCTION_COUNT, num_nodes))

        # initialize with empty schedule and placement for each node, SFC, SF
        scheduling_dict = {v: {sfc: {sf: {} for sf in self.sf_map.keys()} for sfc in self.sfc_map.keys()}
                           for v in self.node_map.keys()}
        placement_dict = {v: set() for v in self.node_map.keys()}

        # parse schedule and prepare dict
        for src_node, src_node_idx in self.node_map.items():
            for sfc, sfc_idx in self.sfc_map.items():
                for sf, sf_idx in self.sf_map.items():
                    for dst_node, dst_node_idx in self.node_map.items():
                        index = (src_node_idx, sfc_idx, sf_idx, dst_node_idx)
                        scheduling_dict[src_node][sfc][sf][dst_node] = scheduling[index]

        # compute dynamic placement depending on schedule and traffic for over all active ingress nodes and all sfcs
        for sfc, sfc_idx in self.sfc_map.items():
            active_ing_nodes = self.simulator.get_active_ingress_nodes()
            logger.debug(f"Active ingress nodes: {active_ing_nodes}")
            for ing in active_ing_nodes:
                # follow possible traffic to calculate sufficient placement
                self.add_placement_recursive(ing, 0, sfc, scheduling_dict, placement_dict)

        # invoke simulator
        logger.debug("call apply on Simulator")
        simulator_action = SimulatorAction(placement_dict, scheduling_dict)
        state = self.simulator.apply(simulator_action)

        obs = self._parse_state_as_graph(state) if self.graph_mode else self._parse_state(state)

        return obs, state

    def _parse_state(self, state: SimulatorState) -> np.ndarray:
        """Formats the SimulationState as an observation space object

        The returned dict contains numpy arrays to form the observation space of the gym env.

        Parameters
        ----------
        state: SimulatorState

        Returns
        -------
        state: dict
            The translated state according to the observation space specification.

        """
                
        # Calculating average relative node utilization
        nodes_utilization = np.array([0.0 for v in state.network['nodes']])
        for node in state.network["nodes"]:
            cap = node["resource"]
            usage = sum(state.network_stats["run_total_processed_traffic"][node['id']].values())
            if cap == 0:
                nodes_utilization[self.node_map[node['id']]] = 1
            else:
                nodes_utilization[self.node_map[node['id']]] = usage/cap
        
        nodes_utilization = np.clip(nodes_utilization / (np.max(nodes_utilization)+1e-3), 0, 1)

        # normalized ingress traffic
        ingress_traffic = np.array([0.0 for v in state.network['nodes']])
        for node, sfc_dict in state.traffic.items():
            for sfc, sf_dict in sfc_dict.items():
                ingress_sf = state.sfcs[sfc][0]
                ingress_traffic[self.node_map[node]] = sf_dict[ingress_sf]
        
        ingress_traffic = np.clip(ingress_traffic / (np.max(ingress_traffic)+1e-3), 0, 1)

        if 'node_cap' in self.observations_space:
            node_cap = np.array([0.0 for v in state.network['nodes']], dtype=np.float32)
            for node in state.network["nodes"]:
                node_cap[self.node_map[node['id']]] = node["resource"]
            
            node_cap = np.clip(node_cap / (np.max(node_cap)+1e-3), 0, 1)

        nn_input_state = np.array([])
        if 'ingress_traffic' in self.observations_space:
            nn_input_state = np.concatenate((nn_input_state, ingress_traffic,), axis=None)
        if 'node_load' in self.observations_space:
            nn_input_state = np.concatenate((nn_input_state, nodes_utilization,), axis=None)
        if "node_cap" in self.observations_space:
            nn_input_state = np.concatenate((nn_input_state, node_cap), axis=None)

        # log RL state to file during testing. need instance check because it requires the simulator writer
        if isinstance(self.simulator, Simulator) and self.simulator.test_mode:
            self.simulator.writer.write_rl_state([self.simulator.episode, self.simulator.env.now]
                                                 + nn_input_state.tolist())

        return nn_input_state

    def _parse_state_as_graph(self, state):
        """
            This method parses SimulatorState to GraphState
        """

        # Making a copy from simulator's network object, so to not interfere with simulator job
        net: nx.Graph = deepcopy(self.simulator.network)
        
        # Deleting dict attributes (We don't need them in our GraphState)
        for (_,d) in net.nodes(data=True):
            d.clear()
        for _,_,d in net.edges(data=True):
            d.clear()
        del net.graph["shortest_paths"]

        group_node_attrs = []

        # Adding `ingress_traffic` attribute to nodes
        if 'ingress_traffic' in self.observations_space:
            group_node_attrs.append("ingress_traffic")
            ingress_traffic = np.array([0.0 for v in state.network['nodes']], dtype=np.float32)
            for node, sfc_dict in state.traffic.items():
                for sfc, sf_dict in sfc_dict.items():
                    ingress_sf = state.sfcs[sfc][0]
                    ingress_traffic[self.node_map[node]] = sf_dict[ingress_sf]
            
            ingress_traffic = np.clip(ingress_traffic / (np.max(ingress_traffic)+1e-3), 0, 1)

            for (node, d) in net.nodes(data=True):
                d["ingress_traffic"] = ingress_traffic[self.node_map[node]]
        
        # Adding `node_load` attribute to nodes
        if 'node_load' in self.observations_space:
            group_node_attrs.append("node_load")
            nodes_utilization = np.array([0.0 for v in state.network['nodes']], dtype=np.float32)
            for node in state.network["nodes"]:
                cap = node["resource"]
                usage = sum(state.network_stats["run_total_processed_traffic"][node['id']].values())
                if cap == 0:
                    nodes_utilization[self.node_map[node['id']]] = 1
                else:
                    nodes_utilization[self.node_map[node['id']]] = usage/cap
            
            nodes_utilization = np.clip(nodes_utilization / (np.max(nodes_utilization)+1e-3), 0, 1)

            for (node, d) in net.nodes(data=True):
                d["node_load"] = nodes_utilization[self.node_map[node]]
        
        if 'node_cap' in self.observations_space:
            group_node_attrs.append("node_cap")
            node_cap = np.array([0.0 for v in state.network['nodes']], dtype=np.float32)
            for node in state.network["nodes"]:
                node_cap[self.node_map[node['id']]] = node["resource"]
            
            node_cap = np.clip(node_cap / (np.max(node_cap)+1e-3), 0, 1)

            for (node, d) in net.nodes(data=True):
                d["node_cap"] = node_cap[self.node_map[node]]

        # Using pytorch_geometric utility to convert networkx's Graph to pytorch_geometric's Data
        data = from_networkx(net,
                            group_node_attrs=group_node_attrs,
                            group_edge_attrs=None)
        
        # generating mask
        num_nodes = len(state.network["nodes"])
        mask = th.zeros(self.env_limits.scheduling_shape, dtype=th.float32)
        mask[:num_nodes, :, :, :num_nodes] = 1

        data["mask"] = mask.view(1, -1)
        
        return data
    
    def permute_node_order(self, state, perm=None):
        """
        Apply random permutation to given vectorized state to shuffle order of nodes within state, eg, node load.
        Important: Depends on definition of state!
        :param state: Vectorized state to be shuffled
        :param perm: Optionally, specify fixed permutation order.
        :return: Shuffled state vector and permutation for reversing
        """
        if perm is None:
            perm = np.random.permutation(len(self.node_map))
        # Current assumption: State = ingress traffic and load per node. Update if state representation changes!
        assert len(perm) == len(self.node_map) == len(state) / 2

        # same node order permutation for ingress traffic and node load
        perm_state = np.array([state[i] for i in perm] + [state[i + len(self.node_map)] for i in perm])
        return perm_state, perm

    def reverse_permutation(self, permutation):
        """Return new permutation to reverse the given permutation"""
        inverse = [0 for _ in range(len(permutation))]
        for idx, pos in enumerate(permutation):
            inverse[pos] = idx
        return inverse

    def reverse_node_permutation(self, action, node_perm):
        """
        Restore the correct order of nodes within an action that was produced for a permuted state.
        Important: Depends on definition of action! Curr assumption: Traffic split per source node, SFC, SF, dest node
        Reorders both source and dest nodes
        :param action: Action vector to be rearranged
        :param node_perm: Permutation order that was originally applied to shuffle the node order in the state
        :return: Reversed action vector
        """
        rev_perm = self.reverse_permutation(node_perm)
        # action: for each src_node, SFC, SF, dest_node: traffic split --> reverse order of nodes
        assert len(action) == len(self.node_map) * len(self.sfc_map) * len(self.sf_map) * len(self.node_map)

        # get dest node slices and reorder them
        rev_dest_action = []
        for i in range(len(self.sfc_map) * len(self.sf_map) * len(self.node_map)):
            start_idx = i * len(self.node_map)
            end_idx = start_idx + len(self.node_map)
            dest_node_slice = action[start_idx:end_idx]
            # reverse order of dest nodes, but keep order of source nodes for now
            rev_dest_action.extend([dest_node_slice[i] for i in rev_perm])

        # get source node slices (containing for each src node: split for SFC, SF, dest node)
        src_node_slices = []
        slice_len = len(self.sfc_map) * len(self.sf_map) * len(self.node_map)
        for src_node_idx in range(len(self.node_map)):
            start_idx = src_node_idx * slice_len
            end_idx = start_idx + slice_len
            src_node_slices.append(rev_dest_action[start_idx:end_idx])
        # reorder source node slices according to reversed permutation
        rev_action = []
        for i in rev_perm:
            rev_action.extend(src_node_slices[i])
        assert len(action) == len(rev_dest_action) == len(rev_action)

        return np.array(rev_action)
