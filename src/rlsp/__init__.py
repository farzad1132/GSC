# -*- coding: utf-8 -*-

from gymnasium.envs.registration import register

register(
    id='rlsp-env-v1',
    entry_point='rlsp.envs:GymEnv',
)
