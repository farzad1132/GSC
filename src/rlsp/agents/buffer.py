from typing import NamedTuple
import torch as th
import numpy as np
from torch_geometric.data import Data, Batch
from gym import spaces

class BufferSample(NamedTuple):
    observations: Batch
    actions: np.ndarray
    next_observations: Batch
    dones: th.Tensor
    rewards: th.Tensor


class GraphReplayBuffer:
    def __init__(self, buffer_size: int, action_space: spaces.Space,
                 device: str = "cpu", ) -> None:
        self.buffer_size = buffer_size
        self.device = device

        self.dones = np.ones(self.buffer_size, np.float32)
        self.rewards = np.zeros(self.buffer_size, dtype=np.float32)
        self.actions = np.zeros(self.buffer_size, dtype=th.Tensor)
        self.observations = np.zeros((self.buffer_size, 3), dtype=Data)
        self.next_observations = np.zeros((self.buffer_size, 3), dtype=Data)
    
        self.pos = 0
        self.full = False
    
    def add(
        self,
        obs: Data,
        next_obs: Data,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos
    ) -> None:
        
        self.dones[self.pos] = done
        self.rewards[self.pos] = reward
        self.actions[self.pos] = th.from_numpy(action)
        self.observations[self.pos] = obs.clone()
        self.next_observations[self.pos] = next_obs.clone()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.pos = 0
            self.full = True

    def sample(self, batch_size: int) -> BufferSample:
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        
        return BufferSample(
            observations=Batch.from_data_list(list(map(lambda x: Data.from_dict(dict(map(lambda y: y, x))), self.observations[batch_inds]))).to(self.device),
            actions=self.actions[batch_inds],
            next_observations=Batch.from_data_list(list(map(lambda x: Data.from_dict(dict(map(lambda y: y, x))), self.next_observations[batch_inds]))).to(self.device),
            dones=th.from_numpy(self.dones[batch_inds]).to(self.device),
            rewards=th.from_numpy(self.rewards[batch_inds]).to(self.device)
        )