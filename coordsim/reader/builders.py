from copy import deepcopy
from typing import Tuple

import networkx as nx

from coordsim.reader import reader


def network_builder(network_file: str, config: dict) \
    -> Tuple[nx.Graph, list, list]:
    """
        config is simulator's config file
    """
    new_config = deepcopy(config)
    new_config["force_link_cap"] = new_config.get("force_link_cap", None)

    if "force_node_cap" not in new_config:
        new_config["force_node_cap"] = None
    elif not isinstance(new_config["force_node_cap"], list):
        raise Exception("force_node_cap should be a tuple")
    
    network, ing_nodes, eg_nodes = reader.read_network(network_file,
                                                       force_link_cap=new_config["force_link_cap"],
                                                       force_node_cap=new_config["force_node_cap"])

    return network, ing_nodes, eg_nodes