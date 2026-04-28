#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 15:11:49 2018

@author: juangabriel

Plantilla de entorno personalizado adaptada a la API de Gymnasium (>=0.26).

Diferencias clave respecto a la versión original (gym v0.21):

* ``import gym`` -> ``import gymnasium as gym``.
* ``metadata`` ahora usa la clave ``"render_modes"`` (en plural) y normalmente
  incluye también ``"render_fps"``.
* ``reset(self, *, seed=None, options=None)`` debe llamar a
  ``super().reset(seed=seed)`` para inicializar el RNG y devolver
  ``(observation, info)``.
* ``step(self, action)`` debe devolver
  ``(observation, reward, terminated, truncated, info)``.
* El ``render_mode`` se fija al construir el entorno (ya no se pasa a
  ``render``).
"""

import gymnasium as gym
import numpy as np


class CustomEnvironment(gym.Env):
    """Plantilla mínima de entorno personalizado compatible con Gymnasium."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: str | None = None):
        self.__version__ = "0.2"
        self.render_mode = render_mode

        # Modificar el espacio de observaciones según las necesidades del
        # entorno (mínimos, máximos, tipo).
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(3,), dtype=np.float32
        )

        # Modificar el espacio de acciones según el problema. El antiguo
        # ``Box(4)`` ya no es válido en Gymnasium: hay que dar low/high y
        # shape explícitos. Aquí dejamos Discrete(4) como ejemplo razonable.
        self.action_space = gym.spaces.Discrete(4)

    def step(self, action):
        """Ejecuta la acción y devuelve la tupla esperada por Gymnasium.

        :param action: La acción a ejecutar en el entorno
        :return: ``(observation, reward, terminated, truncated, info)``
            observation (object): Observación tras aplicar la acción.
            reward (float): Recompensa obtenida.
            terminated (bool): True si el episodio acabó por una condición
                natural del MDP (objetivo alcanzado, fallo, etc.).
            truncated (bool): True si el episodio se cortó por una
                restricción externa (límite de pasos/tiempo, etc.).
            info (dict): Información de depuración adicional.
        """
        # Implementa aquí la lógica del entorno:
        #   - Calcular la recompensa basada en la acción.
        #   - Calcular la observación siguiente.
        #   - Decidir terminated y truncated por separado.
        #   - Rellenar opcionalmente el dict info.
        # Importante: usa ``self.np_random`` (PCG seeded por reset) en lugar
        # de ``np.random`` o ``observation_space.sample()`` para que el
        # entorno sea reproducible al pasar ``seed`` a ``reset``.
        observation = self.np_random.uniform(0.0, 1.0, size=(3,)).astype(np.float32)
        reward = 0.0
        terminated = False
        truncated = False
        info: dict = {}
        return observation, reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Reinicia el entorno y devuelve ``(observation, info)``.

        Hay que llamar siempre a ``super().reset(seed=seed)`` para que el RNG
        ``self.np_random`` de Gymnasium se inicialice correctamente cuando
        se pasa una semilla.
        """
        super().reset(seed=seed)
        observation = self.np_random.uniform(0.0, 1.0, size=(3,)).astype(np.float32)
        info: dict = {}
        return observation, info

    def render(self):
        """En Gymnasium ``render`` ya no recibe ``mode``; se usa el
        ``render_mode`` fijado en ``__init__``."""
        if self.render_mode == "rgb_array":
            # Devuelve un frame RGB (H, W, 3) si quieres soportar grabación.
            return None
        # Modo "human": pinta en pantalla / consola si procede.
        return None

    def close(self):
        return None
