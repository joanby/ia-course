#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 18:26:17 2018

@author: juangabriel
"""

import gymnasium as gym

# Los entornos de Atari ahora se registran a través de ale_py.
# Si lo tienes instalado (pip install "gymnasium[atari]" "ale-py"), lo registramos.
try:
    import ale_py

    gym.register_envs(ale_py)
except ImportError:
    pass

# El antiguo "Qbert-v0" ya no existe en Gymnasium. Usamos el ID moderno ALE/Qbert-v5.
environment = gym.make("ALE/Qbert-v5", render_mode="human")
MAX_NUM_EPISODES = 10
MAX_STEPS_PER_EPISODE = 500

for episode in range(MAX_NUM_EPISODES):
    obs, info = environment.reset()
    for step in range(MAX_STEPS_PER_EPISODE):
        environment.render()
        action = environment.action_space.sample()  # Tomamos una decisión aleatoria
        next_state, reward, terminated, truncated, info = environment.step(action)
        obs = next_state

        if terminated or truncated:
            print("\n Episodio #{} terminado en {} steps.".format(episode, step + 1))
            break

environment.close()  # Cerramos la sesión de Gymnasium
