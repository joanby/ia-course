# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import gym # cargamos la librería de OpenAI Gym

environment = gym.make("MountainCar-v0") # Lanzamos una instancia del videojuego de la Montaña rusa
environment.reset() # Limpiamos y preparamos el entorno para tomar decisiones
for _ in range(2000): # Durante 2000 iteraciones (veces)
    environment.render() # Pintamos en pantalla la acción
    environment.step(environment.action_space.sample()) # Tomamos una decisión aleatoria del conjunto de las disponibles 
    #next_state -> Object
    #reward     -> Float
    #done       -> Boolean
    #info       -> Dictionary
environment.close() # Cerramos la sesión de Open AI Gym