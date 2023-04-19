import argparse
import simpy
import random
import numpy
from coordsim.simulation.flowsimulator import FlowSimulator
from coordsim.reader import reader
from coordsim.metrics.metrics import Metrics
from coordsim.simulation.simulatorparams import SimulatorParams
import coordsim.network.dummy_data as dummy_data
from coordsim.trace_processor.trace_processor import TraceProcessor
import logging
import time
import os


log = logging.getLogger(__name__)


def main():
    args = parse_args()
    start_time = time.time()
    logging.basicConfig(level=logging.INFO)

    # Create a SimPy environment
    env = simpy.Environment()

    # Seed the random generator
    random.seed(args.seed)
    numpy.random.seed(args.seed)

    # Parse network, get NetworkX object ,ingress network list, and egress nodes list
    network, ing_nodes, eg_nodes = reader.read_network(args.network, node_cap=10, link_cap=10)

    # use dummy placement and schedule for running simulator without algorithm
    # TODO: make configurable via CLI
    sf_placement = dummy_data.triangle_placement
    schedule = dummy_data.triangle_schedule

    # Getting current SFC list, and the SF list of each SFC, and config
    sfc_list = reader.get_sfc(args.sf)
    sf_list = reader.get_sf(args.sf, args.sfr)
    config = reader.get_config(args.config)

    metrics = Metrics(network, sf_list)

    # Create the simulator parameters object with the provided args
    params = SimulatorParams(log, network, ing_nodes, eg_nodes, sfc_list, sf_list, config, metrics,
                             sf_placement=sf_placement, schedule=schedule)
    log.info(params)

    # Create a FlowSimulator object, pass the SimPy environment and params objects
    simulator = FlowSimulator(env, params)
    if 'trace_path' in config:
        trace_path = os.path.join(os.getcwd(), config['trace_path'])
        trace = reader.get_trace(trace_path)
        TraceProcessor(params, env, trace, simulator)
        log.info("Using trace " + config['trace_path'])

    # Start the simulation
    simulator.start()

    # Run the simpy environment for the specified duration
    env.run(until=args.duration)

    # Record endtime and running_time metrics
    end_time = time.time()
    metrics.running_time(start_time, end_time)

    # dump all metrics
    log.info(metrics.metrics)


# parse CLI args (when using simulator as stand-alone, not triggered through the interface)
def parse_args():
    parser = argparse.ArgumentParser(description="Coordination-Simulation tool")
    parser.add_argument('-d', '--duration', required=True, dest="duration", type=int,
                        help="The duration of the simulation (simulates milliseconds).")
    parser.add_argument('-sf', '--sf', required=True, dest="sf",
                        help="VNF file which contains the SFCs and their respective SFs and their properties.")
    parser.add_argument('-sfr', '--sfr', required=False, default='', dest='sfr',
                        help="Path which contains the SF resource consumption functions.")
    parser.add_argument('-n', '--network', required=True, dest='network',
                        help="The GraphML network file that specifies the nodes and edges of the network.")
    parser.add_argument('-c', '--config', required=True, dest='config', help="Path to the simulator config file.")
    parser.add_argument('-t', '--trace', required=False, dest='trace', default=None,
                        help="Provide a CSV trace file to configure the traffic the simulator is generating.")
    parser.add_argument('-s', '--seed', required=False, default=random.randint(0, 9999), dest='seed', type=int,
                        help="Random seed")
    return parser.parse_args()


if __name__ == '__main__':
    main()
