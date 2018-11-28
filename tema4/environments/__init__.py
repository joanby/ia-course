#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 15:25:14 2018

@author: juangabriel
"""

from gym.envs.registration import register

register(
        id = "CustomEnvironment-v0",
        entry_point = ":CustomEnvironment"
        )