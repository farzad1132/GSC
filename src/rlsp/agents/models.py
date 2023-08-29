import numpy as np
import torch as th
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv
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

        self.encoder = GATConv(input_dim, hidden_dim, aggr=aggr)

        self.process = nn.ModuleList([])
        for _ in range(num_layers-1):
            self.process.append(GATConv(hidden_dim, hidden_dim, aggr=aggr))
        
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

# TODO: This network doesn't use any feature extractor. See SB3 implementation for more insight
class QNetwork(nn.Module):
    def __init__(self, agent_helper: AgentHelper):
        super().__init__()
        self.agent_helper = agent_helper
        hidden_layers = agent_helper.config["critic_hidden_layer_nodes"]
        obs_space = agent_helper.env.observation_space
        action_space = agent_helper.env.action_space

        ## Feature extractor
        if self.agent_helper.config["graph_mode"] is True:
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
        if self.agent_helper.config["graph_mode"]:
            x = self.embedder(x.x, x.edge_index, x.batch)
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

        if self.agent_helper.config["graph_mode"]:
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
            x = self.embedder(x.x, x.edge_index, x.batch)
        x = self.before_softmax(x)
        y = [self.softmax_layers[i](x[:, i*self.num_nodes:(i+1)*self.num_nodes]) for i in range(self.num_softmax)]
        x = th.concat(y, 1)
        return x