# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import gymnasium as gym  # cargamos la librería de Gymnasium (fork mantenido de OpenAI Gym)

# Lanzamos una instancia del videojuego de la Montaña rusa con render visual
environment = gym.make("MountainCar-v0", render_mode="human")
observation, info = environment.reset()  # reset devuelve (obs, info) en Gymnasium
for _ in range(2000):  # Durante 2000 iteraciones (veces)
    environment.render()  # Pintamos en pantalla la acción
    action = environment.action_space.sample()  # Acción aleatoria del conjunto disponible
    # En Gymnasium step devuelve 5 valores: obs, reward, terminated, truncated, info
    observation, reward, terminated, truncated, info = environment.step(action)
    # observation -> Object
    # reward      -> Float
    # terminated  -> Boolean (el episodio terminó por una condición natural)
    # truncated   -> Boolean (el episodio se cortó por límite de tiempo u otra restricción)
    # info        -> Dictionary
    if terminated or truncated:
        observation, info = environment.reset()
environment.close()  # Cerramos la sesión de Gymnasium
