# -*- coding: utf-8 -*-

import numpy as np
from gym import spaces

from coordsim.reader.reader import read_network, get_sfc


class EnvironmentLimits:
    """
    Data class which contains all the space definitions for the envs.

    Environment's Observation Space needs to be fixed size.
    Hence, this class wraps all limits for all dimensions
    and provides properties to get the resulting spaces.
    """

    def __init__(self, num_nodes: int, sfc_list, node_obs_space_len: int,
                 link_obs_space_len: int, num_edges, graph_mode: bool = False):
        """
        Adapt the env to max len of SFs
        """
        self.MAX_NODE_COUNT = num_nodes
        self.MAX_SF_CHAIN_COUNT = len(sfc_list)
        self.observation_space_len = node_obs_space_len
        self.link_obs_space_len = link_obs_space_len
        self.graph_mode = graph_mode
        self.num_edges = num_edges

        max_sf_length = 0
        for _, sf_list in sfc_list.items():
            if max_sf_length < len(sf_list):
                max_sf_length = len(sf_list)
        self.MAX_SERVICE_FUNCTION_COUNT = max_sf_length

    @property
    def node_load_shape(self):
        """
        Shape of network load dict
        """
        return (self.MAX_NODE_COUNT,)

    @property
    def scheduling_shape(self):
        """
        Shape of simulator scheduling dict
        """
        return (self.MAX_NODE_COUNT,
                self.MAX_SF_CHAIN_COUNT,
                self.MAX_SERVICE_FUNCTION_COUNT,
                self.MAX_NODE_COUNT)

    @property
    def action_space(self):
        """The Space object (gym.space) corresponding to valid actions

        Returns
        -------

        """
        # shape is flattened array of scheduling spaces:
        shape_flattened = (np.prod(self.scheduling_shape),)

        return spaces.Box(low=0, high=1, shape=shape_flattened)

    @property
    def observation_space(self):
        """
        The Space object corresponding to valid observations
        Observation state is ingress traffic of network nodes + load of each node

        Returns
        -------
        gym.space
        """

        node_load_size = self.MAX_NODE_COUNT
        node_shape = (self.observation_space_len * node_load_size,)
        node_obs = spaces.Box(low=0, high=1, shape=node_shape, dtype=np.float32)

        if not self.graph_mode:
            node_load_size = self.MAX_NODE_COUNT
            node_shape = (self.observation_space_len * node_load_size,)
            return spaces.Box(low=0, high=1, shape=node_shape, dtype=np.float32)
        else:
            return spaces.Dict(spaces={
                "nodes": spaces.Box(low=0, high=1, shape=(self.MAX_NODE_COUNT, self.observation_space_len), dtype=np.float32),
                #"edges":  spaces.Box(low=0, high=1, shape=(2*self.num_edges, self.link_obs_space_len), dtype=np.float32),
                "adj": spaces.Box(0, 1000, shape=(2, 2*self.num_edges), dtype=np.int64) 
            })

    def create_filled_node_load_array(self, default=0.0) -> np.ndarray:
        """creates an array with shape and type of the node_load array.

        The array is filled with zeroes or any other default

        Parameters
        ----------
        default
            The default value

        Returns
        -------
            a filled numpy array
        """
        return np.full(shape=self.node_load_shape, fill_value=default, dtype=float)
