#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 15:51:17 2018

@author: juangabriel
"""

import gym
import cv2
import numpy as np

class ResizeReshapeFrames(gym.ObservationWrapper):
    def __init__(self, environment):
        super(ResizeReshapeFrames, self).__init__(environment)
        if len(self.observation_space.shape) == 3:
            self.desired_width = 84
            self.desired_height = 84
            self.desired_channels = self.observation_space.shape[2]
            #La imagen viene en C x H x W
            self.observation_space = gym.spaces.Box(0,255, (self.desired_channels, self.desired_height, self.desired_width), dtype = np.uint8)
            
            
    def observation(self, obs):
        if len(obs.shape) == 3:
            obs = cv2.resize(obs, (self.desired_width, self.desired_height))
            if obs.shape[2] < obs.shape[0]:
                obs = np.reshape(obs, (obs.shape[2], obs.shape[1], obs.shape[0]))
        return obs