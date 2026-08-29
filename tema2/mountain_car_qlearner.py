#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 10:27:43 2018

@author: juangabriel
"""

import gymnasium as gym
import numpy as np


# EPISILON_MIN : vamos aprendiendo, mientras el incremento de aprendizaje sea superior a dicho valor
# MAX_NUM_EPISONES : número máximo de iteraciones que estamos dispuestos a realizar
# STEPS_PER_EPISODE: número máximo de pasos a realizar en cada episodio
# ALPHA: ratio de aprendizaje del agente
# GAMMA: factor de descuento del agente
# NUM_DISCRETE_BINS: número de divisiones en el caso de discretizar el espacio de estados continuo.

MAX_NUM_EPISODES = 50000
STEPS_PER_EPISODE = 200
EPSILON_MIN = 0.005
max_num_steps = MAX_NUM_EPISODES * STEPS_PER_EPISODE
EPSILON_DECAY = 500 * EPSILON_MIN / max_num_steps
ALPHA = 0.05
GAMMA = 0.98
NUM_DISCRETE_BINS = 30


# QLearner Class
# __init__(self, environment)
# discretize(self, obs) [-2,2] -> [-2,-1], [-1,0], [0,1], [1,2]
# get_action(self, obs)
# learn(self, obs, action, reward, next_obs)


class QLearner(object):
    def __init__(self, environment):
        self.obs_shape = environment.observation_space.shape
        self.obs_high = environment.observation_space.high
        self.obs_low = environment.observation_space.low
        self.obs_bins = NUM_DISCRETE_BINS
        self.bin_width = (self.obs_high - self.obs_low) / self.obs_bins

        self.action_shape = environment.action_space.n
        self.Q = np.zeros(
            (self.obs_bins + 1, self.obs_bins + 1, self.action_shape)
        )  # matriz de 31 x 31 x 3
        self.alpha = ALPHA
        self.gamma = GAMMA
        self.epsilon = 1.0

    def discretize(self, obs):
        return tuple(((obs - self.obs_low) / self.bin_width).astype(int))

    def get_action(self, obs):
        discrete_obs = self.discretize(obs)
        # Selección de la acción en base a Epsilon-Greedy
        if self.epsilon > EPSILON_MIN:
            self.epsilon -= EPSILON_DECAY
        if np.random.random() > self.epsilon:  # Con prob 1-epsilon, mejor acción
            return np.argmax(self.Q[discrete_obs])
        # Con probabilidad epsilon, elegimos al azar
        return np.random.choice([a for a in range(self.action_shape)])

    def learn(self, obs, action, reward, next_obs):
        discrete_obs = self.discretize(obs)
        discrete_next_obs = self.discretize(next_obs)
        self.Q[discrete_obs][action] += self.alpha * (
            reward
            + self.gamma * np.max(self.Q[discrete_next_obs])
            - self.Q[discrete_obs][action]
        )


def train(agent, environment):
    """Entrena al agente Q-learning sobre el entorno dado."""
    best_reward = -float("inf")
    for episode in range(MAX_NUM_EPISODES):
        obs, info = environment.reset()
        total_reward = 0.0
        terminated, truncated = False, False
        while not (terminated or truncated):
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = environment.step(action)
            agent.learn(obs, action, reward, next_obs)
            obs = next_obs
            total_reward += reward
        if total_reward > best_reward:
            best_reward = total_reward
        print(
            "Episodio número {} con recompensa: {}, mejor recompensa: {}, epsilon: {}".format(
                episode, total_reward, best_reward, agent.epsilon
            )
        )

    # De todas las políticas obtenidas en entrenamiento devolvemos la mejor.
    return np.argmax(agent.Q, axis=2)


def test(agent, environment, policy):
    obs, info = environment.reset()
    total_reward = 0.0
    terminated, truncated = False, False
    while not (terminated or truncated):
        action = policy[agent.discretize(obs)]
        next_obs, reward, terminated, truncated, info = environment.step(action)
        obs = next_obs
        total_reward += reward
    return total_reward


if __name__ == "__main__":
    environment = gym.make("MountainCar-v0")
    agent = QLearner(environment)
    learned_policy = train(agent, environment)

    # Para grabar vídeo del agente entrenado, RecordVideo necesita render_mode="rgb_array"
    monitor_path = "./monitor_output"
    record_env = gym.make("MountainCar-v0", render_mode="rgb_array")
    record_env = gym.wrappers.RecordVideo(
        record_env, video_folder=monitor_path, episode_trigger=lambda i: True
    )
    for _ in range(1000):
        test(agent, record_env, learned_policy)
    record_env.close()
    environment.close()
