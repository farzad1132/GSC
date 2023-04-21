import csv
import importlib
import logging
import math
from collections import defaultdict
from typing import List

import networkx as nx
import numpy as np
import yaml
from geopy.distance import distance as dist

log = logging.getLogger(__name__)

# Disclaimer: Some snippets of the following file were imported/modified from B-JointSP on GitHub.
# Original code can be found on https://github.com/CN-UPB/B-JointSP

"""
Network parsing module.
- Reads and parses network files into NetworkX.
- Reads and parses network yaml files and gets placement and SFC and SFs.
"""


def get_trace(trace_file):
    """
    Parse the trace file that the simulator will use to generate traffic.
    """
    with open(trace_file) as f:
        trace_rows = csv.DictReader(f)
        traces = []
        for row in trace_rows:
            traces.append(dict(row))
    return traces


def get_config(config_file):
    """
    Parse simulator config params in specified yaml file and return as Python dict
    """
    # TODO: specify defaults as fall back if param is not set in config
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def get_sfc(sfc_file):
    """
    Get the list of SFCs from the yaml data.
    """
    with open(sfc_file) as yaml_stream:
        sfc_data = yaml.load(yaml_stream, Loader=yaml.FullLoader)

    sfc_list = defaultdict(None)
    for sfc_name, sfc_sf in sfc_data['sfc_list'].items():
        sfc_list[sfc_name] = sfc_sf
    return sfc_list


def load_resource_function(name, path):
    try:
        spec = importlib.util.spec_from_file_location(name, path + '/' + name + '.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception:
        raise Exception(f'Cannot load file "{name}.py" from specified location "{path}".')

    try:
        return getattr(module, 'resource_function')
    except Exception:
        raise Exception(f'There is no "resource_function" defined in file "{name}.py."')


def get_sf(sf_file, resource_functions_path=''):
    """
    Get the list of SFs and their properties from the yaml data.
    """
    with open(sf_file) as yaml_stream:
        sf_data = yaml.load(yaml_stream, Loader=yaml.FullLoader)

    # Configurable default mean and stddev defaults
    default_processing_delay_mean = 1.0
    default_processing_delay_stdev = 1.0
    default_startup_delay = 0.0

    def default_resource_function(x):
        return x

    sf_list = defaultdict(None)
    for sf_name, sf_details in sf_data['sf_list'].items():
        sf_list[sf_name] = sf_details
        # Set defaults (currently processing delay mean and stdev)
        sf_list[sf_name]["processing_delay_mean"] = sf_list[sf_name].get("processing_delay_mean",
                                                                         default_processing_delay_mean)
        sf_list[sf_name]["processing_delay_stdev"] = sf_list[sf_name].get("processing_delay_stdev",
                                                                          default_processing_delay_stdev)
        sf_list[sf_name]["startup_delay"] = sf_list[sf_name].get("startup_delay",
                                                                 default_startup_delay)
        if 'resource_function_id' in sf_list[sf_name]:
            try:
                sf_list[sf_name]['resource_function'] = load_resource_function(sf_list[sf_name]['resource_function_id'],
                                                                               resource_functions_path)
            except Exception as ex:
                sf_list[sf_name]['resource_function_id'] = 'default'
                sf_list[sf_name]['resource_function'] = default_resource_function
                log.warning(f'{str(ex)} SF {sf_name} will use default resource function instead.')
        else:
            sf_list[sf_name]["resource_function_id"] = 'default'
            sf_list[sf_name]["resource_function"] = default_resource_function
            log.debug(f'No resource function specified for SF {sf_name}. Default resource function will be used.')
    return sf_list


def weight(edge_cap, edge_delay):
    """
    edge weight = 1 / (cap + 1/delay) => prefer high cap, use smaller delay as additional influence/tie breaker
    if cap = None, set it to 0 use edge_delay as weight
    """
    assert edge_delay is not None
    if edge_cap is None:
        return edge_delay
    if edge_cap == 0:
        return math.inf
    elif edge_delay == 0:
        return 0
    return 1 / (edge_cap + 1 / edge_delay)


def network_diameter(nx_network):
    """Return the network diameter, ie, delay of longest shortest path"""
    if 'shortest_paths' not in nx_network.graph:
        shortest_paths(nx_network)
    return max([path[1] for path in nx_network.graph['shortest_paths'].values()])


def shortest_paths(networkx_network):
    """
    finds the all pairs shortest paths using Johnson Algo
    sets a dictionary, keyed by source and target, of all pairs shortest paths with path_delays in the network as an
    attr.
    key: (src, dest) , value: ([nodes_on_the_shortest_path], path_delay)
    path delays are the sum of individual edge_delays of the edges in the shortest path from source to destination
    """
    # in-built implementation of Johnson Algo, just returns a list of shortest paths
    # returns a dict with : key: source, value: dict with key: dest and value: shortest path as list of nodes
    all_pair_shortest_paths = dict(nx.johnson(networkx_network, weight='weight'))
    # contains shortest paths with path_delays
    # key: (src, dest) , value: ([nodes_on_the_shortest_path], path_delay)
    shortest_paths_with_delays = {}
    for source, v in all_pair_shortest_paths.items():
        for destination, shortest_path_list in v.items():
            path_delay = 0
            # only if the source and destination are different, path_delays need to be calculated, otherwise 0
            if source != destination:
                # shortest_path_list only contains ordered nodes [node1,node2,node3....] in the shortest path
                # here we take ordered pair of nodes (src, dest) to cal. the path_delay of the edge between them
                for i in range(len(shortest_path_list) - 1):
                    path_delay += networkx_network[shortest_path_list[i]][shortest_path_list[i + 1]]['delay']
            shortest_paths_with_delays[(source, destination)] = (shortest_path_list, path_delay)
    networkx_network.graph['shortest_paths'] = shortest_paths_with_delays


def read_network(file, node_cap=None, link_cap=None, force_link_cap: float = None,
                force_node_cap: List[float] = None):
    """
    Read the GraphML file and return list of nodes and edges.
    """
    SPEED_OF_LIGHT = 299792458  # meter per second
    PROPAGATION_FACTOR = 0.77  # https://en.wikipedia.org/wiki/Propagation_delay

    if not file.endswith(".graphml"):
        raise ValueError("{} is not a GraphML file".format(file))
    graphml_network = nx.read_graphml(file, node_type=int)
    networkx_network = nx.Graph()

    #  Setting the nodes of the NetworkX Graph
    for n in graphml_network.nodes(data=True):
        node_id = "pop{}".format(n[0])
        cap = n[1].get("NodeCap", None)
        if cap is None:
            cap = node_cap
            log.warning("NodeCap not set in the GraphML file, now using default NodeCap for node: {}".format(n))
        if force_node_cap is not None:
            cap = np.random.randint(force_node_cap[0], force_node_cap[1])
        node_type = n[1].get("NodeType", "Normal")
        node_name = n[1].get("label", None)
        if cap is None:
            raise ValueError("No NodeCap. set for node{} in file {} (as cmd argument or in graphml)".format(n, file))
        # Adding a Node in the NetworkX Graph
        # {"id": node_id, "name": node_name, "type": node_type, "cap": cpu})
        # Type of node. For now it is either "Normal" or "Ingress"
        # Init 'remaining_resources' to the node capacity
        networkx_network.add_node(node_id, name=node_name, type=node_type, cap=cap, available_sf={},
                                  remaining_cap=cap)

    # set links
    # calculate link delay based on geo positions of nodes;

    for e in graphml_network.edges(data=True):
        # Check whether LinkDelay value is set, otherwise default to None
        source = "pop{}".format(e[0])
        target = "pop{}".format(e[1])
        link_delay = e[2].get("LinkDelay", None)
        # As edges are undirectional, only LinkFwdCap determines the available data rate
        link_fwd_cap = e[2].get("LinkFwdCap", link_cap)
        if e[2].get("LinkFwdCap") is None:
            log.warning(f"Link {(e[0], e[1])} has no capacity defined. Using the default capacity {link_cap} instead.")
        # Setting a default delay of 3 incase no delay specified in GraphML file
        if force_link_cap is not None:
            link_fwd_cap = force_link_cap
        # and we are unable to set it based on Geo location
        delay = 3
        if link_delay is None:
            n1 = graphml_network.nodes(data=True)[e[0]]
            n2 = graphml_network.nodes(data=True)[e[1]]
            n1_lat, n1_long = n1.get("Latitude", None), n1.get("Longitude", None)
            n2_lat, n2_long = n2.get("Latitude", None), n2.get("Longitude", None)
            if n1_lat is None or n1_long is None or n2_lat is None or n2_long is None:
                log.warning("Link Delay not set in the GraphML file and unable to calc based on Geo Location,"
                            "Now using default delay for edge: ({},{})".format(source, target))
            else:
                distance = dist((n1_lat, n1_long), (n2_lat, n2_long)).meters  # in meters
                # round delay to int using np.around for consistency with emulator
                delay = int(np.around((distance / SPEED_OF_LIGHT * 1000) * PROPAGATION_FACTOR))  # in milliseconds
        else:
            delay = link_delay

        # Adding the undirected edges for each link defined in the network.
        # delay = edge delay , cap = edge capacity
        networkx_network.add_edge(source, target, delay=delay, cap=link_fwd_cap, remaining_cap=link_fwd_cap)

    # setting the weight property for each edge in the NetworkX Graph
    # weight attribute is used to find the shortest paths
    for edge in networkx_network.edges.values():
        edge['weight'] = weight(edge['cap'], edge['delay'])
    # Setting the all-pairs shortest path in the NetworkX network as a graph attribute
    shortest_paths(networkx_network)

    # Filter ingress and egress (if any) nodes
    ing_nodes = []
    eg_nodes = []
    for node in networkx_network.nodes.items():
        if node[1]["type"] == "Ingress":
            ing_nodes.append(node)
        if node[1]["type"] == "Egress":
            eg_nodes.append(node[0])

    return networkx_network, ing_nodes, eg_nodes


def reset_cap(network):
    for node in network.nodes.keys():
        network.nodes[node]['remaining_cap'] = network.nodes[node]['cap']
        network.nodes[node]['available_sf'] = {}
    for edge in network.edges(data=True):
        edge[2]['remaining_cap'] = edge[2]['cap']
