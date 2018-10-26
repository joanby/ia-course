#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 11:41:15 2018

@author: juangabriel
"""

from gym import envs

env_names = [env.id for env in envs.registry.all()]

for name in sorted(env_names):
    print(name)

