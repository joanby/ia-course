#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 15:20:42 2018

@author: juangabriel

Wrappers Atari migrados a la API de Gymnasium (>=0.26).

Cambios principales respecto a la versión original (gym v0.21):

* ``import gym`` -> ``import gymnasium as gym``.
* ``atari_py`` ha sido reemplazado por ``ale-py`` y los entornos Atari se
  registran ahora como ``ALE/<Nombre>-v5``. Mantenemos compatibilidad con los
  IDs heredados ``*NoFrameskip-v4`` siempre que se haya hecho
  ``import ale_py; gym.register_envs(ale_py)`` en el script principal.
* ``Wrapper.reset`` ahora devuelve ``(obs, info)`` y acepta los kwargs
  ``seed`` y ``options``.
* ``Wrapper.step`` ahora devuelve ``(obs, reward, terminated, truncated, info)``.
* La clave ``info["ale.lives"]`` pasó a llamarse ``info["lives"]`` en
  ale-py >= 0.8. Se consulta de forma robusta con un fallback.
"""

from __future__ import annotations

import random
import re
from collections import deque

import cv2
import gymnasium as gym
import numpy as np
from gymnasium.spaces.box import Box

try:  # ale_py registra los entornos ALE/ y los IDs heredados *NoFrameskip-v4.
    import ale_py

    gym.register_envs(ale_py)
except ImportError:  # pragma: no cover - opcional
    ale_py = None  # type: ignore[assignment]


def get_games_list():
    """Devuelve los nombres de los juegos Atari registrados (snake_case).

    En versiones modernas de Gymnasium los entornos Atari se registran como
    ``ALE/<Nombre>-v5``. Esta función mantiene la API original (que antes
    delegaba en ``atari_py.list_games()``) iterando sobre el registro de
    Gymnasium y normalizando el nombre a snake_case.
    """
    games: list[str] = []
    for env_id in gym.envs.registry.keys():
        if not env_id.startswith("ALE/"):
            continue
        name = env_id.split("/", 1)[1].split("-")[0]
        snake = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
        games.append(snake)
    return games


def _is_atari(env_id: str) -> bool:
    return env_id.startswith("ALE/") or "NoFrameskip" in env_id


def make_env(env_id, env_conf):
    # Atari requiere render_mode="rgb_array" para que los wrappers de
    # observación puedan procesar los frames como arrays numpy.
    env = gym.make(env_id, render_mode="rgb_array")

    if "NoFrameskip" in env_id:
        assert "NoFrameskip" in env.spec.id
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=env_conf["skip_rate"])

    if env_conf["episodic_life"]:
        env = EpisodicLifeEnv(env)

    try:
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
    except AttributeError:
        pass

    env = AtariRescale(env, env_conf["useful_region"])

    if env_conf["normalize_observation"]:
        env = NormalizedEnv(env)

    env = FrameStack(env, env_conf["num_frames_to_stack"])

    if env_conf["clip_reward"]:
        env = ClipReward(env)

    return env


def process_frame_84(frame, conf):
    frame = frame[conf["crop1"]:conf["crop2"] + 160, :160]
    frame = frame.mean(2)
    frame = frame.astype(np.float32)
    frame *= 1.0 / 255.0
    frame = cv2.resize(frame, (84, conf["dimension2"]))
    frame = cv2.resize(frame, (84, 84))
    frame = np.reshape(frame, [1, 84, 84])
    return frame


class AtariRescale(gym.ObservationWrapper):
    def __init__(self, env, env_conf):
        super().__init__(env)
        self.observation_space = Box(0, 255, [1, 84, 84], dtype=np.uint8)
        self.conf = env_conf

    def observation(self, observation):
        return process_frame_84(observation, self.conf)


class NormalizedEnv(gym.ObservationWrapper):
    def __init__(self, env=None):
        super().__init__(env)
        self.mean = 0
        self.std = 0
        self.alpha = 0.9999
        self.num_steps = 0

    def observation(self, observation):
        self.num_steps += 1
        self.mean = self.mean * self.alpha + observation.mean() * (1 - self.alpha)
        self.std = self.std * self.alpha + observation.std() * (1 - self.alpha)

        unbiased_mean = self.mean / (1 - pow(self.alpha, self.num_steps))
        unbiased_std = self.std / (1 - pow(self.alpha, self.num_steps))

        return (observation - unbiased_mean) / (unbiased_std + 1e-8)


class ClipReward(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        return np.sign(reward)


def _get_lives(info: dict) -> int:
    """Lee el número de vidas restantes del dict ``info`` de un Atari env.

    En ale-py >= 0.8 la clave es ``"lives"``; las versiones antiguas la
    exponían como ``"ale.lives"``.
    """
    if "lives" in info:
        return info["lives"]
    return info.get("ale.lives", 0)


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        super().__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        noops = random.randrange(1, self.noop_max + 1)
        assert noops > 0
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset()
        return obs, info

    def step(self, action):
        return self.env.step(action)


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        obs, _, terminated, truncated, info = self.env.step(1)
        if terminated or truncated:
            obs, info = self.env.reset()
        obs, _, terminated, truncated, info = self.env.step(2)
        if terminated or truncated:
            obs, info = self.env.reset()
        return obs, info

    def step(self, action):
        return self.env.step(action)


class EpisodicLifeEnv(gym.Wrapper):
    """Convierte cada vida perdida en un "episodio" de cara al agente.

    Conserva la lógica original (el flag ``has_really_died`` se activa al
    perder una vida sin haber acabado realmente la partida; cuando vale
    ``True`` el siguiente ``reset`` no toca el entorno interno, sino que
    avanza un paso NOOP para mantener el estado del juego).
    """

    def __init__(self, env):
        super().__init__(env)
        self.lives = 0
        self.has_really_died = False

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Reseteamos el flag por defecto en cada paso. Solo lo activamos si
        # acabamos de perder una vida pero aún quedan vidas.
        self.has_really_died = False
        lives = _get_lives(info)
        if lives < self.lives and lives > 0:
            terminated = True  # señalamos fin de episodio al agente
            self.has_really_died = True  # ... pero el juego interno sigue
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        if not self.has_really_died:
            # Inicio fresco o partida completamente acabada -> reset real.
            obs, info = self.env.reset(seed=seed, options=options)
            self.lives = 0
        else:
            # Vida perdida pero el juego sigue: NOOP para mantener el estado
            # interno sin tirar la partida.
            obs, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                obs, info = self.env.reset(seed=seed, options=options)
            self.lives = _get_lives(info)
        return obs, info


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        super().__init__(env)
        self._obs_buffer = deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        info: dict = {}
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if terminated or truncated:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        self._obs_buffer.clear()
        obs, info = self.env.reset(seed=seed, options=options)
        self._obs_buffer.append(obs)
        return obs, info


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shape = env.observation_space.shape
        self.observation_space = Box(
            low=0,
            high=255,
            shape=(shape[0] * k, shape[1], shape[2]),
            dtype=np.uint8,
        )

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        for _ in range(self.k):
            self.frames.append(obs)
        return self.get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self.get_obs(), reward, terminated, truncated, info

    def get_obs(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class LazyFrames(object):
    def __init__(self, frames):
        self.frames = frames
        self.out = None

    def _force(self):
        if self.out is None:
            self.out = np.concatenate(self.frames, axis=0)
            self.frames = None
        return self.out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]
