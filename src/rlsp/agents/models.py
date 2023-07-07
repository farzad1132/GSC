from typing import List, Optional

import numpy as np
import torch as th
from deepsnap.hetero_graph import HeteroGraph
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import (MLP, BatchNorm, HeteroConv, MessagePassing,
                                global_mean_pool)
from torch_geometric.nn.aggr import Aggregation, MultiAggregation

from src.rlsp.agents.agent_helper import AgentHelper


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
                raise Exception("unsupported update function")
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
    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.convs = th.nn.ModuleList()
        self.norms = th.nn.ModuleList()
        for i in range(num_layers):
            self.norms.append(
                th.nn.ModuleDict(
                    {
                        "link": BatchNorm(hidden_dim),
                        "node": BatchNorm(hidden_dim)
                    }
                ))
            conv = HeteroConv({
                ("node", "nl", "link"): BaseGCN(hidden_dim, hidden_dim,
                                        aggr="mean",
                                        update="mlp",
                                        message_str=None),
                ("link", "ln", "node"): BaseGCN(hidden_dim, hidden_dim,
                                        aggr="mean",
                                        update="mlp",
                                        message_str=None)
            }, aggr="sum")
            self.convs.append(conv)
    
    def forward(self, x_dict, edge_index_dict, batch_dict):
        x_dict = {key: th.nn.functional.pad(value, pad=(0, self.hidden_dim-value.size()[1]),
                                            mode='constant', value=0) for key, value in x_dict.items()}
        for conv, norm in zip(self.convs, self.norms):
            x_dict = conv(x_dict, edge_index_dict)
            #x_dict = {key: norm[key](x) for key, x in x_dict.items()}
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        if batch_dict is None:
            return x_dict["link"]
        else:
            # TODO: check pooling (maybe another one be more effective)
            return global_mean_pool(x_dict["link"], batch_dict)

class GSCActor(nn.Module):
    def __init__(self, agent_helper: AgentHelper):
        super().__init__()

        self.agent_helper = agent_helper
        hidden_layers = agent_helper.config["actor_hidden_layer_nodes"]
        obs_space = agent_helper.env.observation_space
        action_space = agent_helper.env.action_space

        # TODO: config these from the config file
        feature_size = 20
        embedder_layers = 3
        readout_layers = len(hidden_layers)
        readout_hid_dim = hidden_layers[0]

        self.embedder = HeteroGNN(feature_size, embedder_layers)

        self.readout = MLP(
            in_channels=feature_size,
            out_channels=np.prod(action_space.shape),
            num_layers=readout_layers,
            hidden_channels=readout_hid_dim,
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
        hidden_layers = agent_helper.config["critic_hidden_layer_nodes"]
        obs_space = agent_helper.env.observation_space
        action_space = agent_helper.env.action_space

        # TODO: config these from the config file
        feature_size = 20
        embedder_layers = 3
        readout_layers = len(hidden_layers)
        readout_hid_dim = hidden_layers[0]

        self.embedder = HeteroGNN(feature_size, embedder_layers)

        self.readout = MLP(
            in_channels=feature_size,
            out_channels=1,
            num_layers=readout_layers,
            hidden_channels=readout_hid_dim,
            norm=None
        )

    def forward(self, x, a):
        x = self.embedder(x.x_dict, x.edge_index_dict, x.batch_dict["link"])
        return self.readout(x)