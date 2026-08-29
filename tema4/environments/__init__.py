#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 15:25:14 2018

@author: juangabriel
"""

from gymnasium.envs.registration import register

# El ``entry_point`` debe ser ``"<modulo>:<Clase>"`` resoluble como import
# Python. En el repo original aparecía sin módulo (``":CustomEnvironment"``),
# que no funciona en Gymnasium. Apuntamos al módulo de la plantilla.
register(
    id="CustomEnvironment-v0",
    entry_point=(
        "tema4.environments.custom_environment_template:CustomEnvironment"
    ),
)
