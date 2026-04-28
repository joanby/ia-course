#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 16:38:26 2018

@author: juangabriel
"""

import gymnasium as gym

# render_mode="human" muestra una ventana; usa "rgb_array" si lo quieres
# silencioso o si vas a grabar vídeo con RecordVideo.
environment = gym.make("MountainCar-v0", render_mode="human")
MAX_NUM_EPISODES = 1000

for episode in range(MAX_NUM_EPISODES):
    obs, info = environment.reset()
    total_reward = 0.0  # Recompensa total obtenida en cada episodio
    step = 0
    terminated, truncated = False, False
    while not (terminated or truncated):
        environment.render()
        # Acción aleatoria, que posteriormente reemplazaremos por la decisión
        # de nuestro agente inteligente.
        action = environment.action_space.sample()
        next_state, reward, terminated, truncated, info = environment.step(action)
        total_reward += reward
        step += 1
        obs = next_state

    print(
        "\n Episodio número {} finalizado con {} iteraciones. Recompensa final={}".format(
            episode, step + 1, total_reward
        )
    )

environment.close()
