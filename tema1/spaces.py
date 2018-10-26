#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 13:15:19 2018

@author: juangabriel
"""

import gym
from gym.spaces import *
import sys

# Box -> R^n (x1,x2,x3,...,xn), xi [low, high]
#gym.spaces.Box(low = -10, high = 10, shape = (2,)) # (x,y), -10<x,y<10

# Discrete -> Números enteros entre 0 y n-1, {0,1,2,3,...,n-1}
#gym.spaces.Discrete(5) # {0,1,2,3,4}

#Dict -> Diccionario de espacios más complejos
#gym.spaces.Dict({
#            "position": gym.spaces.Discrete(3), #{0,1,2}
#            "velocity": gym.spaces.Discrete(2)  #{0,1}
#        })


# Multi Binary -> {T,F}^n (x1,x2,x3,...xn), xi {T,F}
# gym.spaces.MultiBinary(3)# (x,y,z), x,y,z = T|F

# Multi Discreto -> {a,a+1,a+2..., b}^m
#gym.spaces.MultiDiscrete([-10,10],[0,1])

# Tuple -> Producto de espacios simples
#gym.spaces.Tuple((gym.spaces.Discrete(3), gym.spaces.Discrete(2)))#{0,1,2}x{0,1}

# prng -> Random Seed


def print_spaces(space):
    print(space)
    if isinstance(space, Box):#Comprueba si el space subministrado es de tipo Box
        print("\n Cota inferior: ", space.low)
        print("\n Cota superior: ", space.high)
        
        
if __name__ == "__main__":
    environment = gym.make(sys.argv[1]) ## El usuario debe llamar al script con el nombre del entorno como parámetro
    print("Espacio de estados:")
    print_spaces(environment.observation_space)
    print("Espacio de acciones: ")
    print_spaces(environment.action_space)
    try:
        print("Descripción de las acciones: ", environment.unwrapped.get_action_meanings())
    except AttributeError:
        pass
        








