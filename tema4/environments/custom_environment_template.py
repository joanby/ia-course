#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 15:11:49 2018

@author: juangabriel
"""

import gymnasium as gym

class CustomEnvironment(gym.Env):
    """
    Una plantilla personalizada para crear entornos compatibles con OpenAI Gym
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self):
        self.__version__ = "0.1"
        
        # Modificar el espacio de observaciones,con los mínimos y máximos que necesitemos en base a las necesidades del entorno
        self.observation_space = gym.spaces.Box(low = 0.0, high = 1.0, shape = (3,))
        
        # Modificar el espacio de acciones según las necesidades del entorno
        self.action_space = gym.spaces.Box(4)
        
        
    def step(self, action):
        """
        Ejecuta la acción determinada a cada paso para guiar al agente en el entorno.
        El método reset se ejecutará también al final de cada episodio
        : param action: La acción a ser ejecutada en el entorno en cuestión
        : return : (observation, reward, terminated, truncated, info)
            observation(object):
                Observación del entorno en el momento que se ejecuta la acción
            reward(float):
                Recompensa del entorno en base a la acción ejecutada
            terminated(bool):
                flag booleano para indicar si el episodio ha terminado por el
                propio problema (el agente ha muerto, ha llegado a la meta...)
            truncated(bool):
                flag booleano para indicar si el episodio lo ha cortado algo
                externo al problema (un límite de pasos, por ejemplo)
            info(dict):
                Un diccionario con información adicional sobre la acción ejecutada
        """
        
        #Implementar los pasos del método step aquí:
        #   - Calcular la recompensa basada en la acción
        #   - Calcular la observación siguiente
        #   - Configurar terminated a True si el episodio ha terminado por el propio
        #     problema, y truncated a True si lo ha cortado un límite externo
        #   - Opcionalmente, definir los valores a ser persistidos dentro del diccionario info
        # return(observation, reward, terminated, truncated, info)
        
    def reset(self, seed = None, options = None):
        """
        Reinicia las variables del entorno y devuelve la observación inicial
        : return : (observation, info)
            observation(object): 
                observación inicial después de haber configurado un nuevo episodio
            info(dict):
                Un diccionario con información adicional sobre el episodio nuevo
        """
        # Implementar el método reset aquí
        # return observation, info 
        
    def render(self, mode = 'human', close = False):
        """
        : param mode:
        : return :
        """
        return