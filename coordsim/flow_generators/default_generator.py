import logging
import numpy as np
import random
from typing import Tuple
from coordsim.network.flow import Flow
from coordsim.flow_generators import BaseFlowGenerator
log = logging.getLogger(__name__)


class DefaultFlowGenerator(BaseFlowGenerator):
    """
    This is the default generator class. It generates and inits flows from simulatorparams
    """
    def __init__(self, env, params):
        self.env = env
        self.params = params

    def generate_flow(self, flow_id, node_id) -> Tuple[float, Flow]:
        """ Generate a flow for a given node_id """
        if self.params.deterministic_arrival:
            inter_arr_time = self.params.inter_arr_mean[node_id]
        else:
            # Poisson arrival -> exponential distributed inter-arrival time
            inter_arr_time = random.expovariate(lambd=1.0 / self.params.inter_arr_mean[node_id])
        flow_dr, flow_size = self.get_flow_dr_size()
        # if flow_dr <= 0.00 or flow_size <= 0.00:
        #     continue

        # Assign a random SFC to the flow
        flow_sfc = np.random.choice([sfc for sfc in self.params.sfc_list.keys()])
        # Get the flow's creation time (current environment time)
        creation_time = self.env.now
        # Set the egress node for the flow if some are specified in the network file
        flow_egress_node = None
        if self.params.eg_nodes:
            flow_egress_node = random.choice(self.params.eg_nodes)
        # Generate flow based on given params
        ttl = random.choice(self.params.ttl_choices)
        # Generate flow based on given params
        flow = Flow(str(flow_id), flow_sfc, flow_dr, flow_size, creation_time,
                    current_node_id=node_id, egress_node_id=flow_egress_node, ttl=ttl)
        # Update metrics for the generated flow
        self.params.metrics.generated_flow(flow, node_id)

        return inter_arr_time, flow

    def get_flow_dr_size(self):
        """ Generate data rate and size values for a flow """
        # set normally distributed flow data rate
        flow_dr = np.random.normal(self.params.flow_dr_mean, self.params.flow_dr_stdev)
        if self.params.deterministic_size:
            flow_size = self.params.flow_size_shape
        else:
            # heavy-tail flow size
            flow_size = np.random.pareto(self.params.flow_size_shape) + 1
        # Recursive call if flow size or dr < 0 until a value is obtained
        while flow_dr < 0.00 or flow_size < 0.00:
            flow_dr, flow_size = self.get_flow_dr_size()

        return flow_dr, flow_size
