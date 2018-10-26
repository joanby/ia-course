#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 18:26:17 2018

@author: juangabriel
"""

import gym 

environment = gym.make("Qbert-v0") 
MAX_NUM_EPISODES = 10
MAX_STEPS_PER_EPISODE = 500

for episode in range(MAX_NUM_EPISODES):
    obs = environment.reset()
    for step in range(MAX_STEPS_PER_EPISODE):
        environment.render()
        action = environment.action_space.sample()## Tomamos una decisión aleatoria...
        next_state, reward, done, info = environment.step(action)
        obs = next_state
        
        if done is True:
            print("\n Episodio #{} terminado en {} steps.".format(episode, step+1))
            break
        
environment.close() # Cerramos la sesión de Open AI Gym