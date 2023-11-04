import numpy as np
import torch as th
import torch.nn as nn
from torch_geometric.nn import MLP, GATv2Conv
from torch_geometric.nn.pool import global_mean_pool

from src.rlsp.agents.agent_helper import AgentHelper


class GNNEmbedder(nn.Module):
    """
        This graph embedder module 
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, aggr: str, num_iter: int,
                dropout: float = 0.5):
        super().__init__()

        self.num_iter = num_iter
        self.num_layers = num_layers
        self.dropout = dropout

        self.encoder = GATv2Conv(input_dim, hidden_dim, aggr=aggr)

        self.process = nn.ModuleList([])
        for _ in range(num_layers-1):
            self.process.append(GATv2Conv(hidden_dim, hidden_dim, aggr=aggr))
        
        #self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])


    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        """ for bn in self.bns:
            bn.reset_parameters() """

    def forward(self, x, adj_t, batch):
        x = self.encoder(x, adj_t)
        #x = self.bns[0].forward(x)
        x = nn.functional.relu(x)
        
        if self.num_layers == 1:
            return global_mean_pool(x, batch=batch)

        for iter in range(self.num_iter):
            for i, conv in enumerate(self.process):
                x = conv(x, adj_t)
                #x = self.bns[i+1].forward(x)

                if i == self.num_layers-2 and iter == self.num_iter-1:
                    return global_mean_pool(x, batch=batch)
                else:
                    x = nn.functional.relu(x)

class QNetwork(nn.Module):
    def __init__(self, agent_helper: AgentHelper):
        super().__init__()
        self.graph_mode = agent_helper.config["graph_mode"]
        hidden_layers = list(agent_helper.config["critic_hidden_layer_nodes"])
        obs_space = agent_helper.env.observation_space
        action_space = agent_helper.env.action_space

        ## Feature extractor
        if self.graph_mode is True:
            feature_size = int(agent_helper.config["GNN_features"])
            num_layers = int(agent_helper.config["GNN_num_layers"])
            num_iter = int(agent_helper.config["GNN_num_iter"])
            aggr = agent_helper.config["GNN_aggr"]
            self.embedder = GNNEmbedder(
                input_dim=obs_space["nodes"].shape[-1],
                hidden_dim=feature_size,
                num_layers=num_layers,
                num_iter=num_iter,
                aggr=aggr)
        else:
            feature_size = np.array(obs_space.shape).prod()

        hidden_layers.insert(0, feature_size+2*np.prod(action_space.shape))
        hidden_layers.append(1)
        self.critic = MLP(
            channel_list=hidden_layers,
            norm=None
        )


    def forward(self, x, a):
        if self.graph_mode:
            mask = x.mask
            x = self.embedder(x.x, x.edge_index, x.batch)
            x = th.cat([x, mask], 1)
        x = th.cat([x, a], 1)
        return self.critic(x)


class Actor(nn.Module):
    def __init__(self, agent_helper: AgentHelper):
        super().__init__()
        self.graph_mode = agent_helper.config["graph_mode"]
        hidden_layers = list(agent_helper.config["actor_hidden_layer_nodes"])
        obs_space = agent_helper.env.observation_space
        action_space = agent_helper.env.action_space

        if self.graph_mode:
            feature_size = int(agent_helper.config["GNN_features"])
            num_layers = int(agent_helper.config["GNN_num_layers"])
            num_iter = int(agent_helper.config["GNN_num_iter"])
            aggr = agent_helper.config["GNN_aggr"]
            self.embedder = GNNEmbedder(
                input_dim=obs_space["nodes"].shape[-1],
                hidden_dim=feature_size,
                num_layers=num_layers,
                num_iter=num_iter,
                aggr=aggr)
        else:
            feature_size = np.array(obs_space.shape).prod()
        
        hidden_layers.insert(0, feature_size+np.prod(action_space.shape))
        hidden_layers.append(np.prod(action_space.shape))
        self.actor = MLP(
            channel_list=hidden_layers,
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
        if self.graph_mode:
            mask = x.mask
            x = self.embedder(x.x, x.edge_index, x.batch)
            x = th.concat([x, mask], 1)
        x = self.actor(x)
        if self.graph_mode:
            x = x * mask
        return x