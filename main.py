import pickle
import random
from typing import Callable

import click
import numpy as np
import torch as th

from src.rlsp.agents.main import DATETIME, create_simulator, setup, setup_files
from src.rlsp.agents.rlsp_torch_ddpg import TorchDDPG
from src.rlsp.agents.simple_ddpg import SimpleDDPG
from src.rlsp.envs.gym_env import GymEnv

from src.rlsp.utils.experiment_result import ExperimentResult; np.set_printoptions(suppress=True, precision=3)

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

def exp_decay(init: float = 1e-2, end: float = 1e-3, decay: float = 0.25):
    def func(progress: float):
        lr = end + (init-end)*np.exp(-(1-progress)/decay)
        return lr
    
    return func


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.argument('agent_config', type=click.Path(exists=True))
@click.argument('network', type=click.Path(exists=True))
@click.argument('service', type=click.Path(exists=True))
@click.argument('sim_config', type=click.Path(exists=True))
@click.argument('scheduler', type=click.Path(exists=True))
@click.argument('episodes', type=int)
@click.option('--seed', default=random.randint(1000, 9999),
              help="Specify the random seed for the environment and the learning agent.")
@click.option('-t', '--test', help="Name of the training run whose weights should be used for testing.")
@click.option('-w', '--weights', help="Continue training with the specified weights (similar to testing)")
@click.option('-a', '--append-test', is_flag=True, help="Append a test run of the previously trained agent.")
@click.option('-v', '--verbose', is_flag=True, help="Set console logger level to debug. (Default is INFO)")
@click.option('-b', '--best', is_flag=True, help="Test the best of the trained agents so far.")
@click.option('-ss', '--sim-seed', type=int, help="Set the simulator seed", default=None)
@click.option('-gs', '--gen-scenario', type=click.Path(exists=True),
              help="Diff. sim config file for additional scenario test", default=None)
def cli(agent_config, network, service, sim_config, scheduler, episodes, seed, test, weights, append_test, verbose, best,
        sim_seed, gen_scenario):
    """rlsp cli for learning and testing"""
    global logger

    # Setup agent helper class
    agent_helper = setup(agent_config, network, service, sim_config, scheduler, seed, episodes, weights, verbose, DATETIME, test,
                         append_test, best, sim_seed, gen_scenario)

    agent = SimpleDDPG(agent_helper)

    agent.train(episodes)

    # Saving actor and agent_helper
    th.save(agent.actor, agent_helper.config_dir+"trained_actor.pt")
    del agent_helper.env
    with open(agent_helper.config_dir+"agent_helper.obj", "wb") as f:
        pickle.dump(agent_helper, f)

    ### Testing ###
    if agent_helper.test == True:
        # if test after training (append_test) test for 1 episodes
        agent_helper.episodes = 1
        agent_helper.result = ExperimentResult(agent_helper.experiment_id)
        agent_helper.result.episodes = agent_helper.episodes
        agent_helper.test_mode = True
        setup_files(agent_helper)

    agent_helper.test_mode = True

    env = GymEnv(
        agent_config=agent_helper.config,
        scheduler_conf=agent_helper.schedule,
        network_file=agent_helper.network_path,
        service_file=agent_helper.service_path,
        seed=seed,
        sim_seed=sim_seed,
        agent_helper=agent_helper
    )

    obs, _ = env.reset(agent_helper.seed)
    for _ in range(agent_helper.episode_steps):
        action = agent.predict(obs)
        obs, _, _, _, _ = env.step(action.numpy().squeeze())

if __name__ == "__main__":
    agent_config = 'configs/config/agent/sample_agent.yaml'
    network = 'configs/networks/abilene/abilene-in4-rand-cap1-2.graphml'
    service = 'configs/service_functions/abc.yaml'
    sim_config = 'configs/config/simulator/sample_config.yaml'
    scheduler = 'configs/config/scheduler.yaml'
    # sim_config = 'configs/config/simulator/det-mmp-arrival7-3_det-size0_dur100_no_traffic_prediction.yaml'

    # training for 1 episode
    # cli([agent_config, network, service, sim_config, '1', '-v'])

    # testing for 4 episode
    # cli([agent_config, network, service, sim_config, '1', '-t', '2021-01-07_13-00-43_seed1234'])

    # training & testing for 1 episodes
    cli([agent_config, network, service, sim_config, scheduler, '40', '--append-test'])