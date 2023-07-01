from typing import List

import numpy as np
import torch as th
from torch import nn
from torch.nn import Linear as Lin
from torch.nn import ReLU, BatchNorm1d
from torch.nn import Sequential as Seq
from torch_geometric.nn import GCNConv, global_mean_pool, MetaLayer, NNConv
from torch_geometric.utils import scatter

from src.rlsp.agents.agent_helper import AgentHelper


class EdgeModel(th.nn.Module):
    def __init__(self, node_in: int, edge_in: int, edge_out: int):
        super().__init__()
        self.edge_mlp = Seq(Lin(2*node_in+edge_in, 64), BatchNorm1d(64), ReLU(), Lin(64, edge_out))

    def forward(self, src, dest, edge_attr, u, batch):
        # src, dest: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        out = th.cat([src, dest, edge_attr], 1)
        return self.edge_mlp(out)

class NodeModel(th.nn.Module):
    def __init__(self, node_in: int, edge_out: int, node_out: int):
        super().__init__()
        self.node_mlp_1 = Seq(Lin(edge_out+node_in, 64), BatchNorm1d(64), ReLU(), Lin(64, node_out))
        #self.node_mlp_2 = Seq(Lin(4+node_in, 32), BatchNorm1d(32), ReLU(), Lin(32, node_out))

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        out = th.cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter(out, col, dim=0, dim_size=x.size(0),
                      reduce='mean')
        return out
        """ out = th.cat([x, out], dim=1)
        return self.node_mlp_2(out) """

class GlobalModel(th.nn.Module):
    def __init__(self, node_out: int, global_out: int):
        super().__init__()
        self.global_mlp = Seq(Lin(node_out, 32), ReLU(), Lin(32, global_out))

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        

        # TODO: compare scatter and global_mean_pool
        #scatter(x, batch, dim=0, reduce='mean'),
        out = global_mean_pool(x, batch=batch)
        return self.global_mlp(out)

class NNConvEmbedder(nn.Module):
    def __init__(self, in_node: int, out_node: int, in_edge: int) -> None:
        super().__init__()

        self.embedder = NNConv(
            in_channels=in_node,
            out_channels=out_node,
            nn=nn.Sequential(Lin(in_edge, 64), BatchNorm1d(64), ReLU(), Lin(64, in_node*out_node))
        )
    
    def reset_parameters(self):
        self.embedder.reset_parameters()
    
    def forward(self, x, adj_t, edge_attr, batch):
        out = self.embedder(x=x, edge_index=adj_t, edge_attr=edge_attr)
        return global_mean_pool(out, batch=batch)

class NNConvCritic(nn.Module):
    def __init__(self, agent_helper: AgentHelper) -> None:
        super().__init__()
        #self.agent_helper = agent_helper
        hidden_layers = agent_helper.config["critic_hidden_layer_nodes"]
        #obs_space = agent_helper.env.observation_space
        n_actions = agent_helper.env.action_space.shape[0]
        node_in = agent_helper.env.env_limits.observation_space_len
        edge_in = agent_helper.env.env_limits.link_obs_space_len + 2 + n_actions

        feature_size = 20
        self.feature = NNConvEmbedder(
            in_node=node_in,
            out_node=feature_size,
            in_edge=edge_in
        )

        self.critic = nn.ModuleList()
        
        if len(hidden_layers) > 0:
            self.critic.append(nn.Linear(feature_size + n_actions, hidden_layers[0]))
            self.critic.append(nn.ReLU())
        if len(hidden_layers) >= 2:
            for i in range(len(hidden_layers)-1):
                self.critic.append(hidden_layers[i], hidden_layers[i+1])
                self.critic.append(nn.ReLU()) 
        self.critic.append(nn.Linear(hidden_layers[-1], 1))
        self.critic = nn.Sequential(*self.critic)
    
    def forward(self, x, a):
        x = self.feature(x.x, x.edge_index, x.edge_attr, x.batch)
        x = th.cat([x, a], dim=1)
        return self.critic(x)

class NNConvActor(nn.Module):
    def __init__(self, agent_helper: AgentHelper) -> None:
        super().__init__()
        hidden_layers = agent_helper.config["actor_hidden_layer_nodes"]
        #obs_space = agent_helper.env.observation_space
        action_space = agent_helper.env.action_space
        n_actions = agent_helper.env.action_space.shape[0]
        node_in = agent_helper.env.env_limits.observation_space_len
        edge_in = agent_helper.env.env_limits.link_obs_space_len + 2 + n_actions
        self.low = action_space.low
        self.high = action_space.high
        self.actor = nn.ModuleList()

        feature_size = 20
        self.feature = NNConvEmbedder(
            in_node=node_in,
            out_node=feature_size,
            in_edge=edge_in
        )

        if len(hidden_layers) > 0:
            self.actor.append(nn.Linear(feature_size, hidden_layers[0]))
            self.actor.append(nn.ReLU())
        if len(hidden_layers) >= 2:
            for i in range(len(hidden_layers)-1):
                self.actor.append(hidden_layers[i], hidden_layers[i+1])
                self.actor.append(nn.ReLU()) 
        self.actor.append(nn.Linear(hidden_layers[-1], np.prod(action_space.shape)))
        self.actor = nn.Sequential(*self.actor)
    
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
        x = self.feature(x.x, x.edge_index, x.edge_attr, x.batch)
        x = self.actor(x)
        return x

class GSCActor(nn.Module):
    def __init__(self, agent_helper: AgentHelper) -> None:
        super().__init__()
        n_actions = agent_helper.env.action_space.shape[0]
        node_in = agent_helper.env.env_limits.observation_space_len
        edge_in = agent_helper.env.env_limits.link_obs_space_len + 2 + n_actions
        self.meta_layer = MetaLayer(
            edge_model=EdgeModel(node_in=node_in, edge_in=edge_in, edge_out=10),
            node_model=NodeModel(node_in=node_in, edge_out=10, node_out=10),
            global_model=None)
        # GlobalModel(node_out=4, global_out=n_actions)
        self.readout = Seq(Lin(10, 64), BatchNorm1d(64), ReLU(), Lin(64, n_actions))
        self.low = agent_helper.env.action_space.low
        self.high = agent_helper.env.action_space.high
    
    def forward(self, x):
        out, _, _ = self.meta_layer(x.x, x.edge_index, x.edge_attr, None, x.batch)
        u = global_mean_pool(out, x.batch)
        u = self.readout(u)
        return u
    
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

class GSCCritic(nn.Module):
    def __init__(self, agent_helper: AgentHelper) -> None:
        super().__init__()
        n_actions = agent_helper.env.action_space.shape[0]
        node_in = agent_helper.env.env_limits.observation_space_len
        edge_in = agent_helper.env.env_limits.link_obs_space_len + 2 + n_actions
        self.meta_layer = MetaLayer(
            edge_model=EdgeModel(node_in=node_in, edge_in=edge_in, edge_out=4),
            node_model=NodeModel(node_in=node_in, edge_out=4, node_out=10),
            global_model=None)
        # GlobalModel(node_out=4, global_out=10)
        self.readout = Seq(Lin(10+n_actions, 64), BatchNorm1d(64), ReLU(), Lin(64, 1))

    def forward(self, x, a):
        """ index = (x.edge_attr[:, -1] == 1).nonzero().squeeze()
        x.edge_attr[index, 2:5] = a
        x.edge_attr[index, 5] = 1
        _, _, u = self.meta_layer(x.x, x.edge_index, x.edge_attr, None, x.batch) """
        out, _, _ = self.meta_layer(x.x, x.edge_index, x.edge_attr, None, x.batch)
        x = global_mean_pool(out, x.batch)
        x = th.cat([x, a], dim=1)
        u = self.readout(x)
        return u

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

# TODO: This network doesn't use any feature extractor. See SB3 implementation for more insight
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


