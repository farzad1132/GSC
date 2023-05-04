"""
RLSP utility functions module
"""
from typing import Dict

import gym
from torch_geometric.data import Batch, Data

from src.rlsp.agents.agent_helper import AgentHelper
from src.rlsp.envs.gym_env import AutoResetWithSeed, GymEnv


def create_environment(agent_helper):
    # not sure why, but simulator has to be loaded here (not at top) for proper logging

    agent_helper.result.env_config['seed'] = agent_helper.seed
    agent_helper.result.env_config['sim-seed'] = agent_helper.sim_seed
    agent_helper.result.env_config['network_file'] = agent_helper.network_path
    agent_helper.result.env_config['service_file'] = agent_helper.service_path
    agent_helper.result.env_config['sim_config_file'] = agent_helper.sim_config_path
    agent_helper.result.env_config['simulator_cls'] = "siminterface.Simulator"

    # Get the environment and extract the number of actions.
    """ env = gym.make(ENV_NAME,
                   agent_config=agent_helper.config,
                   simulator=create_simulator(agent_helper),
                   network_file=agent_helper.network_path,
                   service_file=agent_helper.service_path,
                   seed=agent_helper.seed,
                   sim_seed=agent_helper.sim_seed) """

    env = GymEnv(
        agent_config=agent_helper.config,
        simulator=create_simulator(agent_helper),
        network_file=agent_helper.network_path,
        service_file=agent_helper.service_path,
        seed=agent_helper.seed,
        sim_seed=agent_helper.sim_seed
    )
    env.num_envs = 1
    agent_helper.result.env_config['reward_fnc'] = str(env.reward_func_repr())
    #env = DummyVecEnv([lambda: env])
    #env = Monitor(env, agent_helper.config_dir)

    return env

def create_simulator(agent_helper):
    """Create a simulator object"""
    from siminterface.simulator import Simulator

    return Simulator(agent_helper.network_path, agent_helper.service_path, agent_helper.sim_config_path,
                     test_mode=agent_helper.test_mode, test_dir=agent_helper.config_dir)

def make_env(agent_helper: AgentHelper, seed, idx, capture_video, run_name):
    def thunk():
        env = create_environment(agent_helper)
        agent_helper.env = env
        env = AutoResetWithSeed(env, seed)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        #env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

def simple_make_env(agent_helper: AgentHelper):
    env = create_environment(agent_helper)
    agent_helper.env = env
    # TODO: check between default auto reset and auto reset with seed
    #env = gym.wrappers.AutoResetWrapper(env)
    env = AutoResetWithSeed(env, agent_helper.seed)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env

def torch_stack_to_graph_batch(obs: Dict) -> Batch:
    """
        This function converts stacked arrays to pytorch_geometric's Batch object
    """
    data_list = []
    for i in range(obs["adj"].shape[0]):
        data = Data(x=obs["nodes"][i, :], edge_index=obs["adj"][i, :, :], edge_attr=obs["edges"][i, :])
        data_list.append(data)
    return Batch.from_data_list(data_list)

def graph_to_dict(data: Data) -> Dict:
    """
        This function converts graph obs to dict obs to be stored in buffer
    """
    return {
        "nodes": data.x,
        "edges": data.edge_attr,
        "adj": data.edge_index
    }