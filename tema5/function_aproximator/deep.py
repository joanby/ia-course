#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 11:17:49 2018

@author: juangabriel
"""


import torch

class DeepActor(torch.nn.Module):
    
    def __init__(self, input_shape, output_shape, device = torch.device("cpu")):
        """
        Una red neuronal convolucional profunda que producirá dos valores continuos (media y desviación típica) 
        para cada uno de los valores de output_shape. usando el algoritmo de CNN
        Se utiliza para representar el papel del actor en A2C.
        :param input_shape: Forma de los datos de entrada (representan las observaciones del actor)
        :param output_shape: Forma de los datos de salida (representan las acciones que debe producir el actor)
        :param device: Donde se almacena y opera la red neuronal (CPU vs CUDA).
        """
        super(DeepActor, self).__init__()
        self.device = device
        self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(input_shape[2], 32, 8, stride = 4, padding = 0),
                                          torch.nn.ReLU())
        self.layer2 = torch.nn.Sequential(torch.nn.Conv2d(32, 64, 3, stride = 2, padding = 0),
                                          torch.nn.ReLU())
        self.layer3 = torch.nn.Sequential(torch.nn.Conv2d(64, 64, 3, stride = 1, padding = 0),
                                          torch.nn.ReLU())
        self.layer4 = torch.nn.Sequential(torch.nn.Linear(64 * 7 * 7, 512),
                                          torch.nn.ReLU())
        self.actor_mu = torch.nn.Linear(512, output_shape)
        self.actor_sigma = torch.nn.Linear(512, output_shape)
        
    def forward(self, x):
        """
        Dada el nuevo dato que obtiene el actor, calculamos la media y la desviación típica
        :param x: observación
        :return: Media (mu) y Desviación Estándar (sigma) para una política gaussiana
        """
        x.require_grad_()
        x = x.to(self.device)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.shape[0], -1)
        x = self.layer4(x)
        mu = self.actor_mu(x)
        sigma = self.actor_sigma(x)
        return mu, sigma
        
        
        
class DeepDiscreteActor(torch.nn.Module):
    def __init__(self, input_shape, output_shape, device = torch.device("cpu")):
        """
        Una red neorunal convolucional profunda que utilizará una función logística para discriminar 
        la acción del espacio de acciones discreto usando una CNN.
        Se utiliza para representar el papel del actor en A2C.
        :param input_shape: Forma de los datos de entrada (representan las observaciones del actor)
        :param output_shape: Forma de los datos de salida (representan las acciones que debe producir el actor)
        :param device: Donde se almacena y opera la red neuronal (CPU vs CUDA).
        """
        super(DeepDiscreteActor, self).__init__()
        self.device = device
        self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(input_shape[2], 32, 8, stride = 4, padding = 0),
                                          torch.nn.ReLU())
        self.layer2 = torch.nn.Sequential(torch.nn.Conv2d(32, 64, 3, stride = 2, padding = 0),
                                          torch.nn.ReLU())
        self.layer3 = torch.nn.Sequential(torch.nn.Conv2d(64, 64, 3, stride = 1, padding = 0),
                                          torch.nn.ReLU())
        self.layer4 = torch.nn.Sequential(torch.nn.Linear(64 * 7 * 7, 512),
                                          torch.nn.ReLU())
        self.actor_logits = torch.nn.Linear(512, output_shape)
        
    def forward(self, x):
        """
        Dada el nuevo dato que obtiene el actor, calculamos la acción con la función logit
        :param x: observación
        :return: logits según la política del agente
        """
        x.require_grad_()
        x = x.to(self.device)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.shape[0], -1)
        x = self.layer4(x)
        logits = self.actor_logits(x)
        return logits
        
        
class DeepCritic(torch.nn.Module):
    def __init__(self, input_shape, output_shape = 1, device = torch.device("cpu")):
        """
        Una red neorunal convolucional profunda que produce un valor contínuo.
        Se utiliza para representar el papel del crítico en A2C.
        Estima el valor de la observación / estado actual
        :param input_shape: Forma de los datos de entrada (representan las observaciones del actor)
        :param output_shape: Forma de los datos de salida (representan el feedback que debe producir el crítico, suele ser un solo valor)
        :param device: Donde se almacena y opera la red neuronal (CPU vs CUDA).
        """
        super(DeepCritic, self).__init__()
        self.device = device
        self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(input_shape[2], 32, 8, stride = 4, padding = 0),
                                          torch.nn.ReLU())
        self.layer2 = torch.nn.Sequential(torch.nn.Conv2d(32, 64, 3, stride = 2, padding = 0),
                                          torch.nn.ReLU())
        self.layer3 = torch.nn.Sequential(torch.nn.Conv2d(64, 64, 3, stride = 1, padding = 0),
                                          torch.nn.ReLU())
        self.layer4 = torch.nn.Sequential(torch.nn.Linear(64 * 7 * 7, 512),
                                          torch.nn.ReLU())
        self.critic = torch.nn.Linear(512, output_shape)
        
    def forward(self, x):
        """
        A partir de los datos de entrada, devolvemos el valor estimado de salida como críticos
        :param x: observación
        :return: logits según la política del agente
        """
        x.require_grad_()
        x = x.to(self.device)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.shape[0], -1)
        x = self.layer4(x)
        critic = self.critic(x)
        return critic
        
        
class DeepActorCritic(torch.nn.Module):
    def __init__(self, input_shape, actor_shape, critic_shape, device = torch.device("cpu")):
        """
        Una red neorunal convolucional profunda que se utiliza para representar tanto al actor como al crítico en el algoritmo A2C.
        :param input_shape: Forma de los datos de entrada (representan las observaciones del actor)
        :param actor_shape: Forma de los datos de salida del actor (representan las acciones que debe producir el actor)
        :param critic_shape: Forma de los datos de salida del crítico (representan el feedback que debe producir el crítico, suele ser un solo valor)
        :param device: Donde se almacena y opera la red neuronal (CPU vs CUDA).
        """
        super(DeepActorCritic, self).__init__()
        self.device = device
        self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(input_shape[2], 32, 8, stride = 4, padding = 0),
                                          torch.nn.ReLU())
        self.layer2 = torch.nn.Sequential(torch.nn.Conv2d(32, 64, 3, stride = 2, padding = 0),
                                          torch.nn.ReLU())
        self.layer3 = torch.nn.Sequential(torch.nn.Conv2d(64, 64, 3, stride = 1, padding = 0),
                                          torch.nn.ReLU())
        self.layer4 = torch.nn.Sequential(torch.nn.Linear(64 * 7 * 7, 512),
                                          torch.nn.ReLU())
        self.actor_mu = torch.nn.Linear(512, actor_shape)
        self.actor_sigma = torch.nn.Linear(512, actor_shape)
        self.critic = torch.nn.Linear(512, critic_shape)
        
        
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
        x = self.layer3(x)
        x = x.view(x.shape[0], -1)
        x = self.layer4(x)
        actor_mu = self.actor_mu(x)
        actor_sigma = self.actor_sigma(x)
        critic = self.critic(x)
        return actor_mu, actor_sigma, critic