#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 10:47:19 2018

@author: juangabriel
"""

import torch

class SwallowActor(torch.nn.Module):
    
    def __init__(self, input_shape, output_shape, device = torch.device("cpu")):
        """
        Una red neorunal que producirá dos valores continuos (media y desviación típica) para cada uno de los valores de output_shape.
        Se utiliza para representar el papel del actor en A2C.
        :param input_shape: Forma de los datos de entrada (representan las observaciones del actor)
        :param output_shape: Forma de los datos de salida (representan las acciones que debe producir el actor)
        :param device: Donde se almacena y opera la red neuronal (CPU vs CUDA).
        """
        super(SwallowActor, self).__init__()
        self.device = device
        self.layer1 = torch.nn.Sequential(torch.nn.Linear(input_shape[0], 64),
                                          torch.nn.ReLU())
        self.layer2 = torch.nn.Sequential(torch.nn.Linear(64, 32),
                                          torch.nn.ReLU())
        self.actor_mu = torch.nn.Linear(32, output_shape)
        self.actor_sigma = torch.nn.Linear(32, output_shape)
        
    def forward(self, x):
        """
        Dada el nuevo dato que obtiene el actor, calculamos la media y la desviación típica
        :param x: observación
        :return: Media (mu) y Desviación Estándar (sigma) para una política gaussiana
        """
        x = x.to(self.device)
        x = self.layer1(x)
        x = self.layer2(x)
        mu = self.actor_mu(x)
        sigma = self.actor_sigma(x)
        return mu, sigma
        
        
        
class SwallowDiscreteActor(torch.nn.Module):
    def __init__(self, input_shape, output_shape, device = torch.device("cpu")):
        """
        Una red neorunal que utilizará una función logística para discriminar la acción del espacio de acciones discreto.
        Se utiliza para representar el papel del actor en A2C.
        :param input_shape: Forma de los datos de entrada (representan las observaciones del actor)
        :param output_shape: Forma de los datos de salida (representan las acciones que debe producir el actor)
        :param device: Donde se almacena y opera la red neuronal (CPU vs CUDA).
        """
        super(SwallowDiscreteActor, self).__init__()
        self.device = device
        self.layer1 = torch.nn.Sequential(torch.nn.Linear(input_shape[0], 64),
                                          torch.nn.ReLU())
        self.layer2 = torch.nn.Sequential(torch.nn.Linear(64, 32),
                                          torch.nn.ReLU())
        self.actor_logits = torch.nn.Linear(32, output_shape)
        
    def forward(self, x):
        """
        Dada el nuevo dato que obtiene el actor, calculamos la acción con la función logit
        :param x: observación
        :return: logits según la política del agente
        """
        x = x.to(self.device)
        x = self.layer1(x)
        x = self.layer2(x)
        logits = self.actor_logits(x)
        return logits
        
        
class SwallowCritic(torch.nn.Module):
    def __init__(self, input_shape, output_shape = 1, device = torch.device("cpu")):
        """
        Una red neorunal que produce un valor contínuo.
        Se utiliza para representar el papel del crítico en A2C.
        Estima el valor de la observación / estado actual
        :param input_shape: Forma de los datos de entrada (representan las observaciones del actor)
        :param output_shape: Forma de los datos de salida (representan el feedback que debe producir el crítico, suele ser un solo valor)
        :param device: Donde se almacena y opera la red neuronal (CPU vs CUDA).
        """
        super(SwallowCritic, self).__init__()
        self.device = device
        self.layer1 = torch.nn.Sequential(torch.nn.Linear(input_shape[0], 64),
                                          torch.nn.ReLU())
        self.layer2 = torch.nn.Sequential(torch.nn.Linear(64, 32),
                                          torch.nn.ReLU())
        self.critic = torch.nn.Linear(32, output_shape)
        
    def forward(self, x):
        """
        A partir de los datos de entrada, devolvemos el valor estimado de salida como críticos
        :param x: observación
        :return: logits según la política del agente
        """
        x = x.to(self.device)
        x = self.layer1(x)
        x = self.layer2(x)
        critic = self.critic(x)
        return critic
        
        
class SwallowActorCritic(torch.nn.Module):
    def __init__(self, input_shape, actor_shape, critic_shape, device = torch.device("cpu")):
        """
        Una red neorunal que se utiliza para representar tanto al actor como al crítico en el algoritmo A2C.
        :param input_shape: Forma de los datos de entrada (representan las observaciones del actor)
        :param actor_shape: Forma de los datos de salida del actor (representan las acciones que debe producir el actor)
        :param critic_shape: Forma de los datos de salida del crítico (representan el feedback que debe producir el crítico, suele ser un solo valor)
        :param device: Donde se almacena y opera la red neuronal (CPU vs CUDA).
        """
        super(SwallowActorCritic, self).__init__()
        self.device = device
        self.layer1 = torch.nn.Sequential(torch.nn.Linear(input_shape[0], 32),
                                          torch.nn.ReLU())
        self.layer2 = torch.nn.Sequential(torch.nn.Linear(32, 16),
                                          torch.nn.ReLU())
        self.actor_mu = torch.nn.Linear(16, actor_shape)
        self.actor_sigma = torch.nn.Linear(16, actor_shape)
        self.critic = torch.nn.Linear(16, critic_shape)
        
        
    def forward(self, x):
        """
        A partir de los datos de entrada, devolvemos el valor estimado de salida para el actor y el crítico
        :param x: observación
        :return: logits según la política del agente
        """
        x.require_grad_()
        x = x.to(self.device)
        x = self.layer1(x)
        x = self.layer2(x)
        actor_mu = self.actor_mu(x)
        actor_sigma = self.actor_sigma(x)
        critic = self.critic(x)
        return actor_mu, actor_sigma, critic