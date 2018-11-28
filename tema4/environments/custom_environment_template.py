#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 15:11:49 2018

@author: juangabriel
"""

import gym

class CustomEnvironment(gym.Env):
    """
    Una plantilla personalizada para crear entornos compatibles con OpenAI Gym
    """
    
    metadata = {'render.modes': ['human']}
    
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
        : return : (observation, reward, done, info)
            observation(object):
                Observación del entorno en el momento que se ejecuta la acción
            reward(float):
                Recompensa del entorno en base a la acción ejecutada
            done(bool):
                flag booleano para indicar si el episodio ha terminado o no
            info(dict):
                Un diccionario con información adicional sobre la acción ejecutada
        """
        
        #Implementar los pasos del método step aquí:
        #   - Calcular la recompensa basada en la acción
        #   - Calcular la observación siguiente
        #   - Configurar done a True si el episodio ha terminado y a False en otro caso
        #   - Opcionalmente, definir los valores a ser persistidos dentro del diccionario info
        # return(observation, reward, done, info)
        
    def reset(self):
        """
        Reinicia las variables del entorno y devuelve la observación inicial
        : return : observation
            observation(object): 
                observación inicial después de haber configurado un nuevo episodio
        """
        # Implementar el método reset aquí
        # return observation 
        
    def render(self, mode = 'human', close = False):
        """
        : param mode:
        : return :
        """
        return