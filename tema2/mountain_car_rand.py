#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 16:38:26 2018

@author: juangabriel
"""

import gym
environment = gym.make("MountainCar-v0")
MAX_NUM_EPISODES = 1000

for episode in range(MAX_NUM_EPISODES):
    done = False
    obs = environment.reset()
    total_reward = 0.0 ## Variable para guardar la recompensa total obtenida en cada episodio
    step = 0
    while not done:
        environment.render()
        action = environment.action_space.sample()## Acción aleatoria, que posteriormente reemplazaremos por la decisión de nuestro agente inteligente
        next_state, reward, done, info = environment.step(action)
        total_reward += reward
        step += 1
        obs = next_state
        
    print("\n Episodio número {} finalizado con {} iteraciones. Recompensa final={}".format(episode, step+1, total_reward))
    
environment.close()
