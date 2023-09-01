import os
import pickle

import torch as th

from src.rlsp.agents.agent_helper import AgentHelper
from src.rlsp.agents.models import Actor
from src.rlsp.envs import GymEnv


def predict(obs, agent_helper, actor):
    if agent_helper.config["graph_mode"]:
        return actor(obs)
    else:
        return actor(th.tensor(obs, dtype=th.float32).view(1, -1))

dir = "results/sample_agent/scheduler/abc/sample_config/2023-09-01_08-44-57_seed1460"

with open(os.path.join(dir, "agent_helper.obj"), "rb") as f:
    agent_helper: AgentHelper = pickle.load(f)

actor: Actor = th.load(os.path.join(dir, "trained_actor.pt"))
actor.eval()

agent_helper.test_mode = True

env = GymEnv(
    agent_config=agent_helper.config,
    scheduler_conf=agent_helper.schedule,
    network_file=agent_helper.network_path,
    service_file=agent_helper.service_path,
    seed=agent_helper.seed,
    sim_seed=agent_helper.sim_seed,
    agent_helper=agent_helper
)

obs, _ = env.reset(agent_helper.seed)
for _ in range(agent_helper.episode_steps):
    action = predict(obs, agent_helper, actor)
    obs, _, _, _, _ = env.step(action.detach().numpy().squeeze())