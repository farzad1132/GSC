import networkx as nx
from matplotlib import pyplot as plt
#import seaborn as sns; sns.set()
from typing import List
import torch as th


def homo_to_hetero(G: nx.DiGraph, node_features: List[str], link_features: List[str]) -> nx.DiGraph:
    """This utility function converts a homogeneous nx.DiGraph to a heterogeneous nx.DiGraph
    """
    net = nx.DiGraph(G)
    from copy import deepcopy

    index = 0
    remove_edges = []
    add_nodes = []
    add_edges = []
    for u, v, data in net.edges(data=True):
        remove_edges.append((u, v))
        label = f"L{index}"
        index += 1
        features = []
        for feat in link_features:
            if not feat in data:
                raise Exception(f"Feature `{feat}` not found in link data")
            if isinstance(data[feat], list):
                features.extend(data[feat])
            else:
                features.append(data[feat])
        add_nodes.append((label, {"node_type": "link", "node_feature": th.tensor(features)}))
        add_edges.append((u, label, {"edge_type": "nl"}))
        add_edges.append((label, v, {"edge_type": "ln"}))
    
    for u, data in net.nodes(data=True):
        features = []
        for feat in node_features:
            if not feat in data:
                raise Exception(f"Feature `{feat}` not found in node data")
            if isinstance(data[feat], list):
                features.extend(data[feat])
            else:
                features.append(data[feat])
            del data[feat]
        net.add_node(u, node_type="node", node_feature=th.tensor(features))

    net = nx.DiGraph(net)
    net.remove_edges_from(remove_edges)
    net.add_nodes_from(add_nodes)
    net.add_edges_from(add_edges)
    return net

def assign_value_to_links(values: List, G: nx.DiGraph) -> nx.DiGraph:
    index = 0
    for _, data in G.nodes(data=True):
        if data["node_type"] == "link":
            data["value"] = values[index]
            index += 1
    return G

def hetero_to_homo(G: nx.DiGraph) -> nx.DiGraph:
    net = nx.DiGraph()

    from copy import deepcopy

    for u, data in G.nodes(data=True):
        if data["node_type"] == "node":
            net.add_node(u, **deepcopy(data))
        elif data["node_type"] == "link":
            src = list(G.in_edges(u))[0][0]
            dst = list(G.out_edges(u))[0][1]
            net.add_edge(src, dst, **deepcopy(data))
    
    return net

from deepsnap.hetero_graph import HeteroGraph
from torch_geometric.data import HeteroData, Batch


def snap_to_geom(graph: HeteroGraph) -> HeteroData:
    data = HeteroData()
    for key in graph.node_feature.keys():
        data[key].x = graph.node_feature[key]

    for key in graph.edge_index.keys():
        a, b, c = key
        data[a, b, c].edge_index = graph.edge_index[key]
    return data

net = nx.Graph()

net.add_node("1", node_type="node", w=1, x=[1, 0, 0])
net.add_node("2", node_type="node", w=2, x=[2, 0, 0])
net.add_node("3", node_type="node", w=3, x=[3, 0, 0])
net.add_node("4", node_type="node", w=4, x=[4, 0, 0])

net.add_edge("1", "2", t=1)
net.add_edge("2", "3", t=2)
net.add_edge("2", "4", t=3)
net.add_edge("3", "4", t=4)

net.add_edge("1", "1", t=5)
net.add_edge("2", "2", t=6)
net.add_edge("3", "3", t=7)
net.add_edge("4", "4", t=8)

net = nx.to_directed(net)

""" nx.draw(net, with_labels=True)
plt.show() """

net = homo_to_hetero(net, node_features=["w", "x"], link_features=["t"])

color_map = []
for _, data in net.nodes(data=True):
    if "node_type" in data:
        type = data["node_type"]
        color_map.append("blue" if type == "node" else "green")
    else:
        color_map.append("blue")
nx.draw(net, with_labels=True, node_color=color_map)
plt.show()

het = HeteroGraph(net)
x = het.node_feature
edge_index = het.edge_index
nl = edge_index[("node", "nl", "link")]
ln = edge_index[("link", "ln", "node")]

ptr = th.zeros(12).to(th.int64)
ptr[0] = 1
ptr[3] = 1
index = (ptr == 1).nonzero()

#to_node_mask = ln[0, :] == index[:, 0]
to_node_mask = th.searchsorted(ln[0, :], index[:, 0])
#from_node_mask = nl[1, :] == index[:, 0]
from_node_mask = th.searchsorted(nl[1, :], index[:, 0])

to_node_index = ln[1, to_node_mask]
from_node_index = nl[0, from_node_mask]

to_node = x["node"][to_node_index]
from_node = x["node"][from_node_index]

exit(0)
data = snap_to_geom(het)
lis = Batch.from_data_list([data, data])

net = assign_value_to_links([i for i in range(12)], net)
net = hetero_to_homo(net)
print()
""" color_map = []
for _, data in net.nodes(data=True):
    if "node_type" in data:
        type = data["node_type"]
        color_map.append("blue" if type == "node" else "green")
    else:
        color_map.append("blue")
nx.draw(net, with_labels=True, node_color=color_map)
plt.show() """

net = hetero_to_homo(net)
""" nx.draw(net, with_labels=True)
plt.show() """