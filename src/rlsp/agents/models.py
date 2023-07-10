from typing import List, Optional

import numpy as np
import torch as th
from deepsnap.hetero_graph import HeteroGraph
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import (MLP, BatchNorm, HeteroConv, MessagePassing,
                                global_mean_pool, GCNConv)
from torch_geometric.nn.aggr import Aggregation, MultiAggregation

from src.rlsp.agents.agent_helper import AgentHelper

class GNNEmbedder(nn.Module):
    """
        This graph embedder module 
    """
    def __init__(self, input_dim, hidden_dim: List[int], num_layers, dropout: float = 0.5):
        super().__init__()

        if not isinstance(hidden_dim, list):
            raise Exception("hidden_dim should be a list of int")
        assert num_layers == len(hidden_dim), "Size if hidden_dim should be equal to number of layers"


        # A list of 1D batch normalization layers
        #self.bns = None

        self.convs = nn.ModuleList([GCNConv(input_dim, hidden_dim[0])])
        if num_layers > 1:
            self.convs.extend([GCNConv(hidden_dim[i], hidden_dim[i+1]) for i in range(num_layers-1)])

        #self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers-1)])
        self.num_layers = num_layers

        # Probability of an element to be zeroed
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        """ for bn in self.bns:
            bn.reset_parameters() """

    def forward(self, x, adj_t, batch):
        out = None

        for layer in range(self.num_layers):
            if layer != self.num_layers -1:
                x = self.convs[layer].forward(x, adj_t)
                #x = self.bns[layer].forward(x)
                x = nn.functional.relu(x)
                #x = nn.functional.dropout(x, p=self.dropout, training=self.training)
            else:
                x = self.convs[layer].forward(x, adj_t)
                out = global_mean_pool(x, batch=batch)

        return out

class QNetwork(nn.Module):
    def __init__(self, agent_helper: AgentHelper):
        super().__init__()
        self.agent_helper = agent_helper
        hidden_layers = agent_helper.config["critic_hidden_layer_nodes"]
        obs_space = agent_helper.env.observation_space
        action_space = agent_helper.env.action_space

        ## Feature extractor
        if agent_helper.config["critic_feature_size"] is not None:
            feature_size = int(agent_helper.config["critic_feature_size"])
        else:
            feature_size = np.array(obs_space.shape).prod()
        
        if self.agent_helper.config["graph_mode"] is True and \
            agent_helper.config["critic_feature_size"] is not None:
            self.feature = GNNEmbedder(2, [feature_size], 1)
            self.graph_mode = True
        else:
            self.graph_mode = False

        self.critic = nn.ModuleList()
        
        if len(hidden_layers) > 0:
            self.critic.append(nn.Linear(feature_size \
                             + np.prod(action_space.shape), hidden_layers[0]))
            self.critic.append(nn.ReLU())
        if len(hidden_layers) >= 2:
            for i in range(len(hidden_layers)-1):
                self.critic.append(hidden_layers[i], hidden_layers[i+1])
                self.critic.append(nn.ReLU()) 
        self.critic.append(nn.Linear(hidden_layers[-1], 1))
        self.critic = nn.Sequential(*self.critic)


    def forward(self, x, a):
        if self.graph_mode:
            x = self.feature(x.x, x.edge_index, x.batch)
        x = th.cat([x, a], 1)
        return self.critic(x)


class Actor(nn.Module):
    def __init__(self, agent_helper: AgentHelper):
        super().__init__()
        self.agent_helper = agent_helper
        hidden_layers = agent_helper.config["actor_hidden_layer_nodes"]
        obs_space = agent_helper.env.observation_space
        action_space = agent_helper.env.action_space
        self.before_softmax = nn.ModuleList()

        ## Feature extractor
        feature_size = 22
        self.feature = GNNEmbedder(2, [feature_size], 1)
        
        if len(hidden_layers) > 0:
            self.before_softmax.append(nn.Linear(feature_size, hidden_layers[0]))
            self.before_softmax.append(nn.ReLU())
        if len(hidden_layers) >= 2:
            for i in range(len(hidden_layers)-1):
                self.before_softmax.append(hidden_layers[i], hidden_layers[i+1])
                self.before_softmax.append(nn.ReLU()) 
        self.before_softmax.append(nn.Linear(hidden_layers[-1], np.prod(action_space.shape)))
        self.before_softmax = nn.Sequential(*self.before_softmax)

        self.low = action_space.low
        self.high = action_space.high
        self.num_nodes = agent_helper.env.env_limits.MAX_NODE_COUNT
        self.num_softmax = self.num_nodes*agent_helper.env.env_limits.MAX_SERVICE_FUNCTION_COUNT
        self.softmax_layers = [nn.Softmax(1) for _ in range(self.num_softmax)]
    
    def scale_action(self, action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [low, high] to [-1, 1]
        (no need for symmetric action space)

        :param action: Action to scale
        :return: Scaled action
        """
        return 2.0 * ((action - self.low) / (self.high - self.low)) - 1.0

    def unscale_action(self, scaled_action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [-1, 1] to [low, high]
        (no need for symmetric action space)

        :param scaled_action: Action to un-scale
        """
        return self.low + (0.5 * (scaled_action + 1.0) * (self.high - self.low))

    def forward(self, x):
        if self.agent_helper.config["graph_mode"]:
            x = self.feature(x.x, x.edge_index, x.batch)
        x = self.before_softmax(x)
        y = [self.softmax_layers[i](x[:, i*self.num_nodes:(i+1)*self.num_nodes]) for i in range(self.num_softmax)]
        x = th.concat(y, 1)
        return x

class BaseGCN(MessagePassing):
    def __init__(self, in_channel: int, out_channel: int, aggr: Aggregation, update: str = None,
                 num_iter: int = None, message_str: str = None, tar_in_channel: int = None):
        super().__init__(aggr=aggr)
        self.update_str = update
        self.message_str = message_str
        if tar_in_channel is None:
            self.tar_in_channel = in_channel
        else:
            self.tar_in_channel = tar_in_channel
        self.in_channels = in_channel
        self.out_channels = out_channel
        if update is not None:
            if update == "mlp":
                from torch_geometric.nn import MLP
                self.fn_update = MLP(in_channels=out_channel+in_channel,
                                    out_channels=out_channel,
                                    num_layers=1,
                                    norm=None)
            
            elif update == "rnn":
                if num_iter is None:
                    raise Exception("with update set to 'rnn', 'num_iter' should be set")
                self.num_iter = num_iter
                self.counter = 0
                from torch.nn import GRU
                self.fn_update = GRU(out_channel+in_channel, out_channel, batch_first=True)
            else:
                raise Exception(f"unsupported update function '{update}'")
        if self.message_str is not None:
            if self.message_str == "mlp":
                from torch_geometric.nn import MLP
                self.fn_message = MLP(in_channels=in_channel+self.tar_in_channel,
                                      out_channels=in_channel,
                                      num_layers=1,
                                      norm=None)
            else:
                raise Exception("message_str can only be 'mlp'")
    
    def message(self, x_j: th.Tensor, x_i: th.Tensor) -> th.Tensor:
        if self.message_str == "mlp":
            x = th.cat([x_j, x_i], dim=1)
            return self.fn_message(x)
        else:
            return x_j


    def forward(self, x, edge_index):
        return self.propagate(edge_index=edge_index, x=x)
    
    def update(self, inputs: th.Tensor, x: th.Tensor) -> th.Tensor:
        if self.update_str is None:
            return inputs
        elif self.update_str == "mlp":
            x = th.cat([inputs, x[1]], dim=1)
            return self.fn_update(x)
        elif self.update_str == "rnn":
            batch_size = inputs.size()[0]
            if self.counter == 0:
                self.hn = th.zeros((1, batch_size, self.out_channels))
            x = th.cat([inputs, x[1]], dim=1).view((batch_size, 1, self.out_channels+self.in_channels))
            output, self.hn = self.fn_update(x, self.hn)
            self.counter += 1
            if self.counter == self.num_iter:
                self.counter = 0
            return output.squeeze()
    
class CascadeMLPAggregation(Aggregation):
    def __init__(self, list_aggr: List[str], in_channels, out_channels) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.agg1 = MultiAggregation(list_aggr)

        from torch_geometric.nn import MLP
        self.mlp = MLP(in_channels=len(list_aggr)*in_channels,
                    out_channels=out_channels,
                    num_layers=1,
                    norm=None)
    
    def reset_parameters(self):
        self.mlp.reset_parameters()

    def forward(self, x: th.Tensor, index: Optional[th.Tensor] = None,
                ptr: Optional[th.Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> th.Tensor:
        x = self.agg1(x, index, ptr, dim_size, dim)
        return self.mlp(x)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')

class HeteroGNN(nn.Module):
    def __init__(self, hidden_dim: int, num_iter: int, aggr: str, pool: str,
                message: str = None, update: str = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pool = pool
        self.convs = th.nn.ModuleList()
        for i in range(num_iter):
            conv = HeteroConv({
                ("node", "nl", "link"): BaseGCN(hidden_dim, hidden_dim,
                                        aggr=aggr,
                                        update=update,
                                        message_str=message),
                ("link", "ln", "node"): BaseGCN(hidden_dim, hidden_dim,
                                        aggr=aggr,
                                        update=update,
                                        message_str=message)
            }, aggr="sum")
            self.convs.append(conv)
    
    def local_pool(self, ptr, x, edge_index_dict):
        index = (ptr == 1).nonzero()
        link_feat = x["link"][index[:, 0], :]

        nl = edge_index_dict[("node", "nl", "link")]
        ln = edge_index_dict[("link", "ln", "node")]
        
        sorted_ln, ln_indices = th.sort(ln[0, :])
        to_node_mask = th.searchsorted(sorted_ln, index[:, 0])
        to_node_mask = ln_indices[to_node_mask]

        sorted_nl, nl_indices = th.sort(nl[1, :])
        from_node_mask = th.searchsorted(sorted_nl, index[:, 0])
        from_node_mask = nl_indices[from_node_mask]

        to_node_index = ln[1, to_node_mask]
        from_node_index = nl[0, from_node_mask]

        to_node = x["node"][to_node_index]
        from_node = x["node"][from_node_index]

        return th.cat([link_feat, to_node, from_node], dim=1)
    
    def forward(self, x_dict, edge_index_dict, batch_dict):
        ptr = x_dict["link"][:, -1]
        x_dict = {key: th.nn.functional.pad(value, pad=(0, self.hidden_dim-value.size()[1]),
                                            mode='constant', value=0) for key, value in x_dict.items()}
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        if self.pool == "global":
            return global_mean_pool(x_dict["link"], batch_dict)
        elif self.pool == "local":
            return self.local_pool(ptr, x_dict, edge_index_dict)

class GSCActor(nn.Module):
    def __init__(self, agent_helper: AgentHelper):
        super().__init__()

        self.agent_helper = agent_helper
        hidden_layers: list = agent_helper.config["actor_readout_layers"]
        feature_size = agent_helper.config["GNN_features"]
        embedder_layers = agent_helper.config["GNN_layers"]
        action_space = agent_helper.env.action_space

        readout_layers = hidden_layers.copy()
        readout_layers.insert(0, feature_size*3)
        readout_layers.append(np.prod(action_space.shape))

        message = self.agent_helper.config["GNN_message"]
        if message == "None":
            message = None
        aggr = self.agent_helper.config["GNN_aggr"]
        update = self.agent_helper.config["GNN_update"]
        if update == "None":
            update = None
        self.embedder = HeteroGNN(feature_size, embedder_layers, aggr=aggr,
                                message=message, update=update, pool="local")

        self.readout = MLP(
            channel_list=readout_layers,
            norm=None
        )

        self.low = action_space.low
        self.high = action_space.high
    
    def scale_action(self, action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [low, high] to [-1, 1]
        (no need for symmetric action space)

        :param action: Action to scale
        :return: Scaled action
        """
        return 2.0 * ((action - self.low) / (self.high - self.low)) - 1.0

    def unscale_action(self, scaled_action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [-1, 1] to [low, high]
        (no need for symmetric action space)

        :param scaled_action: Action to un-scale
        """
        return self.low + (0.5 * (scaled_action + 1.0) * (self.high - self.low))

    def forward(self, x):
        if self.agent_helper.config["graph_mode"]:
            if isinstance(x, Batch):
                x = self.embedder(x.x_dict, x.edge_index_dict, x.batch_dict["link"])
            elif isinstance(x, HeteroGraph):
                x = self.embedder(x.node_feature, x.edge_index,
                                th.zeros((x.node_feature["link"].size()[0],), dtype=th.int64))
            else:
                raise Exception("unknown input type")
        return self.readout(x)
    

class GSCCritic(nn.Module):
    def __init__(self, agent_helper: AgentHelper):
        super().__init__()
        self.agent_helper = agent_helper
        hidden_layers: list = agent_helper.config["critic_readout_layers"]
        feature_size = agent_helper.config["GNN_features"]
        embedder_layers = agent_helper.config["GNN_layers"]

        readout_layers = hidden_layers.copy()
        readout_layers.insert(0, feature_size)
        readout_layers.append(1)

        message = self.agent_helper.config["GNN_message"]
        if message == "None":
            message = None
        aggr = self.agent_helper.config["GNN_aggr"]
        update = self.agent_helper.config["GNN_update"]
        if update == "None":
            update = None
        self.embedder = HeteroGNN(feature_size, embedder_layers, aggr=aggr,
                                message=message, update=update, pool="global")

        self.readout = MLP(
            channel_list=readout_layers,
            norm=None
        )

    def forward(self, x, a):
        x = self.embedder(x.x_dict, x.edge_index_dict, x.batch_dict["link"])
        return self.readout(x)