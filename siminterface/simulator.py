import logging
import random
import time
import os
from shutil import copyfile
from coordsim.metrics.metrics import Metrics
import coordsim.reader.reader as reader
from coordsim.simulation.flowsimulator import FlowSimulator
from coordsim.simulation.simulatorparams import SimulatorParams
import numpy
import simpy
from spinterface import SimulatorAction, SimulatorInterface, SimulatorState
from coordsim.writer.writer import ResultWriter
from coordsim.trace_processor.trace_processor import TraceProcessor
#from coordsim.traffic_predictor.traffic_predictor import TrafficPredictor
#from coordsim.traffic_predictor.lstm_predictor import LSTM_Predictor
from coordsim.controller import *

logger = logging.getLogger(__name__)


class Simulator(SimulatorInterface):
    def __init__(self, network_file, service_functions_file, config_file, resource_functions_path="", test_mode=False,
                 test_dir=None):
        super().__init__(test_mode)
        # Number of time the simulator has run. Necessary to correctly calculate env run time of apply function
        self.network_file = network_file
        self.test_dir = test_dir
        # init network, sfc, sf, and config files
        self.sfc_list = reader.get_sfc(service_functions_file)
        self.sf_list = reader.get_sf(service_functions_file, resource_functions_path)
        self.config = reader.get_config(config_file)
        self.config["force_link_cap"] = self.config.get("force_link_cap", None)
        self.network, self.ing_nodes, self.eg_nodes = reader.read_network(self.network_file,
                                                                         force_link_cap=self.config["force_link_cap"])
        self.metrics = Metrics(self.network, self.sf_list)
        # Assume result path is the path where network file is in.
        self.result_base_path = os.path.dirname(self.network_file)
        if 'trace_path' in self.config:
            # Quick solution to copy trace file to same path for network file as provided by calling algo.
            trace_path = os.path.join(os.getcwd(), self.config['trace_path'])
            copyfile(trace_path, os.path.join(self.result_base_path, os.path.basename(trace_path)))

        self.prediction = False
        """ # Check if future ingress traffic setting is enabled
        if 'future_traffic' in self.config and self.config['future_traffic']:
            self.prediction = True """
        self.params = SimulatorParams(logger, self.network, self.ing_nodes, self.eg_nodes, self.sfc_list, self.sf_list,
                                      self.config, self.metrics, prediction=self.prediction)
        write_schedule = False
        if 'write_schedule' in self.config and self.config['write_schedule']:
            write_schedule = True
        write_flow_actions = False
        if 'write_flow_actions' in self.config and self.config['write_flow_actions']:
            write_flow_actions = True
        # Create CSV writer
        self.writer = ResultWriter(self.test_mode, self.test_dir, write_schedule, write_flow_actions)
        self.params.writer = self.writer
        self.episode = 0
        self.params.episode = 0
        self.last_apply_time = None
        # Load trace file
        if 'trace_path' in self.config:
            trace_path = os.path.join(os.getcwd(), self.config['trace_path'])
            self.trace = reader.get_trace(trace_path)

        self.lstm_predictor = None
        """ if 'lstm_prediction' in self.config and self.config['lstm_prediction']:
            self.lstm_predictor = LSTM_Predictor(self.trace, params=self.params,
                                                 weights_dir=self.config['lstm_weights']) """

    def __del__(self):
        # write dropped flow locs to yaml
        self.writer.write_dropped_flow_locs(self.metrics.metrics['dropped_flows_locs'])

    def init(self, seed):
        # Reset predictor class at beginning of every init
        """ if self.prediction:
            self.predictor = TrafficPredictor(self.params, self.lstm_predictor) """
        # increment episode count
        self.episode += 1
        self.params.episode += 1
        # reset network caps and available SFs:
        reader.reset_cap(self.network)
        # Initialize metrics, record start time
        self.params.run_times = int(1)
        self.start_time = time.time()

        # Generate SimPy simulation environment
        self.env = simpy.Environment()
        self.env.process(self.writer.begin_writing(self.env, self.params))

        self.params.metrics.reset_metrics()

        # Instantiate the parameter object for the simulator.
        if self.params.use_states and 'trace_path' in self.config:
            logger.warning('Two state model and traces are both activated, thi will cause unexpected behaviour!')

        if self.params.use_states:
            if self.params.in_init_state:
                self.params.in_init_state = False
            # else:
            self.env.process(self.params.start_mmpp(self.env))

        self.duration = self.params.run_duration
        # Get and plant random seed
        self.seed = seed
        random.seed(self.seed)
        numpy.random.seed(self.seed)

        self.params.reset_flow_lists()
        # generate flow lists 1x here since we are in `init()`
        self.params.generate_flow_lists()

        # Instantiate a simulator object, pass the environment and params
        self.simulator = FlowSimulator(self.env, self.params)

        # Trace handling
        if 'trace_path' in self.config:
            TraceProcessor(self.params, self.env, self.trace, self.simulator)

        # Start the simulator
        self.simulator.start()

        # TODO: Create runner here
        controller_cls = eval(self.params.controller_class)
        self.controller = controller_cls(self.env, self.params, self.simulator)
        # # Run the environment for one step to get initial stats.
        # self.env.step()

        # # Parse the NetworkX object into a dict format specified in SimulatorState. This is done to account
        # # for changing node remaining capacities.
        # # Also, parse the network stats and prepare it in SimulatorState format.
        # self.parse_network()
        # self.network_metrics()

        # Record end time and running time metrics
        self.end_time = time.time()
        self.params.metrics.running_time(self.start_time, self.end_time)
        simulator_state = self.controller.get_init_state()
        # Check to see if traffic prediction is enabled to provide future traffic not current traffic
        # if self.prediction:
        #     requested_traffic = self.get_current_ingress_traffic()
        #     self.predictor.predict_traffic(self.env.now, current_traffic=requested_traffic)
        #     stats = self.params.metrics.get_metrics()
        #     self.traffic = stats['run_total_requested_traffic']
        # simulator_state = SimulatorState(self.network_dict, self.simulator.params.sf_placement, self.sfc_list,
        #                                  self.sf_list, self.traffic, self.network_stats)
        # logger.debug(f"t={self.env.now}: {simulator_state}")
        # set time stamp to calculate runtime of next apply call
        self.last_apply_time = time.time()
        # Check to see if init called in warmup, if so, set warmup to false
        # This is to allow for better prediction and better overall control
        # in the future
        return simulator_state

    def apply(self, actions):

        logger.debug(f"t={self.env.now}: {actions}")

        # calc runtime since last apply (or init): that's the algorithm's runtime without simulation
        alg_runtime = time.time() - self.last_apply_time
        self.writer.write_runtime(alg_runtime)
        simulator_state = self.controller.get_next_state(actions)

        # # Get the new placement from the action passed by the RL agent
        # # Modify and set the placement parameter of the instantiated simulator object.
        # self.simulator.params.sf_placement = actions.placement
        # Update which sf is available at which node
        # for node_id, placed_sf_list in actions.placement.items():
        #     available = {}
        #     # Keep only SFs which still process
        #     for sf, sf_data in self.simulator.params.network.nodes[node_id]['available_sf'].items():
        #         if sf_data['load'] != 0:
        #             available[sf] = sf_data
        #     # Add all SFs which are in the placement
        #     for sf in placed_sf_list:
        #         if sf not in available.keys():
        #             available[sf] = available.get(sf, {
        #                 'load': 0.0,
        #                 'last_active': self.env.now,
        #                 'startup_time': self.env.now
        #             })
        #     self.simulator.params.network.nodes[node_id]['available_sf'] = available

        # Get the new schedule from the SimulatorAction
        # Set it in the params of the instantiated simulator object.
        # self.simulator.params.schedule = actions.scheduling

        # reset metrics for steps; now done in result writer
        # self.params.metrics.reset_run_metrics()

        # Run the simulation again with the new params for the set duration.
        # Due to SimPy restraints, we multiply the duration by the run times because SimPy does not reset when run()
        # stops and we must increase the value of "until=" to accomodate for this. e.g.: 1st run call runs for 100 time
        # uniits (1 run time), 2nd run call will also run for 100 more time units but value of "until=" is now 200.
        # runtime_steps = self.duration * self.params.run_times
        # logger.debug("Running simulator until time step %s", runtime_steps)
        # self.env.run(until=runtime_steps)

        # Parse the NetworkX object into a dict format specified in SimulatorState. This is done to account
        # for changing node remaining capacities.
        # Also, parse the network stats and prepare it in SimulatorState format.
        # self.parse_network()
        # self.network_metrics()

        # Increment the run times variable
        self.params.run_times += 1

        # Record end time of the apply round, doesn't change start time to show the running time of the entire
        # simulation at the end of the simulation.
        self.end_time = time.time()
        self.params.metrics.running_time(self.start_time, self.end_time)

        # if self.params.use_states:
        #     self.params.update_state()
        # generate flow data for next run (used for prediction)
        self.params.generate_flow_lists(now=self.env.now)

        # Check to see if traffic prediction is enabled to provide future traffic not current traffic
        # if self.prediction:
        #     requested_traffic = self.get_current_ingress_traffic()
        #     self.predictor.predict_traffic(self.env.now, current_traffic=requested_traffic)
        #     stats = self.params.metrics.get_metrics()
        #     self.traffic = stats['run_total_requested_traffic']
        # # Create a new SimulatorState object to pass to the RL Agent
        # simulator_state = SimulatorState(self.network_dict, self.simulator.params.sf_placement, self.sfc_list,
        #                                  self.sf_list, self.traffic, self.network_stats)
        # self.writer.write_state_results(self.episode, self.env.now, simulator_state,
        # self.params.metrics.get_metrics())
        logger.debug(f"t={self.env.now}: {simulator_state}")
        # set time stamp to calculate runtime of next apply call
        self.last_apply_time = time.time()
        return simulator_state

    def get_current_ingress_traffic(self) -> float:
        """
        Get current ingress traffic for the LSTM module
        Current limitation: works for 1 SFC and 1 ingress node
        """
        # Get name of ingress SF from first SFC
        first_sfc = list(self.sfc_list.keys())[0]
        ingress_sf = self.params.sfc_list[first_sfc][0]
        ingress_node = self.params.ing_nodes[0][0]
        ingress_traffic = self.metrics.metrics['run_total_requested_traffic'][ingress_node][first_sfc][ingress_sf]
        return ingress_traffic

    def get_active_ingress_nodes(self):
        """Return names of all ingress nodes that are currently active, ie, produce flows."""
        return [ing[0] for ing in self.ing_nodes if self.params.inter_arr_mean[ing[0]] is not None]


# for debugging
if __name__ == "__main__":
    # run from project root for file paths to work
    # I changed triangle to have 2 ingress nodes for debugging
    network_file = 'params/networks/triangle.graphml'
    service_file = 'params/services/abc.yaml'
    config_file = 'params/config/sim_config.yaml'

    sim = Simulator(network_file, service_file, config_file)
    state = sim.init(seed=1234)
    dummy_action = SimulatorAction(placement={}, scheduling={})
    # FIXME: this currently breaks - negative flow counter?
    #  should be possible to have an empty action and just drop all flows!
    state = sim.apply(dummy_action)
    