#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 18:06:32 2018

@author: juangabriel
"""

import sys

import gymnasium as gym


def run_gym_environment(argv):
    # El primer parámetro de argv será el nombre del entorno a ejecutar.
    # En Gymnasium el modo de renderizado se fija al crear el entorno.
    environment = gym.make(argv[1], render_mode="human")
    observation, info = environment.reset()
    for _ in range(int(argv[2])):
        environment.render()
        action = environment.action_space.sample()
        observation, reward, terminated, truncated, info = environment.step(action)
        if terminated or truncated:
            observation, info = environment.reset()
    environment.close()


if __name__ == "__main__":
    run_gym_environment(sys.argv)
