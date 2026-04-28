#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 11:41:15 2018

@author: juangabriel
"""

import gymnasium as gym

# En Gymnasium el registro vive en gym.envs.registration.registry y es un dict
# {env_id: EnvSpec}. Iteramos sobre sus claves para obtener todos los IDs.
env_names = list(gym.envs.registry.keys())

for name in sorted(env_names):
    print(name)
