# Migración de OpenAI Gym v0.21 a Gymnasium (>= 0.26)

El código original del curso se escribió contra **OpenAI Gym v0.21**.
A partir de Gym v0.26 la API rompió compatibilidad y, poco después, el
proyecto se bifurcó en **[Gymnasium](https://gymnasium.farama.org/)**, que es
el fork mantenido oficialmente por la Farama Foundation. Todo el repositorio
(`tema1` a `tema5`) ha sido migrado a Gymnasium.

Este documento resume los cambios y cómo afectan a los ejemplos del curso.

## Resumen ejecutivo

| Componente | Antes (gym v0.21) | Ahora (Gymnasium) |
| --- | --- | --- |
| Importación | `import gym` | `import gymnasium as gym` |
| `reset()` | `obs = env.reset()` | `obs, info = env.reset(seed=...)` |
| `step()` | `obs, reward, done, info = env.step(a)` | `obs, reward, terminated, truncated, info = env.step(a)` |
| Fin de episodio | `while not done:` | `while not (terminated or truncated):` |
| Render | `env.render(mode="human")` | `gym.make(id, render_mode="human")` y `env.render()` |
| Semilla | `env.seed(42)` | `env.reset(seed=42)` |
| Vídeo | `gym.wrappers.Monitor` | `gymnasium.wrappers.RecordVideo` |
| Atari | `import atari_py` + IDs `*NoFrameskip-v4` | `import ale_py; gym.register_envs(ale_py)` + IDs `ALE/<Juego>-v5` |
| Robótica | en el paquete `gym` | paquete aparte `gymnasium-robotics` |
| `CartPole-v0` | disponible | retirado, usa `CartPole-v1` |
| `Qbert-v0` | disponible | usa `ALE/Qbert-v5` |

## ¿Por qué cambió la API?

* **`done` → `terminated` / `truncated`.** El antiguo `done` mezclaba dos cosas
  muy distintas: el episodio terminó por una condición natural del MDP
  (objetivo alcanzado, agente muerto, etc.) o el episodio se cortó por una
  restricción externa (límite de pasos del wrapper `TimeLimit`). Esta
  ambigüedad llevaba a calcular mal el *bootstrap* del valor en algoritmos
  como Q-learning o Actor-Critic. Con la nueva API:
  * `terminated == True` ⇒ no hay valor futuro, el target es solo `reward`.
  * `truncated  == True` ⇒ sí hay valor futuro, hay que estimarlo con la red
    o tabla Q (`reward + γ · V(s')`).
* **`reset(seed=...)`.** Algunos entornos (en especial juegos emulados) solo
  pueden reinicializar su RNG al principio de un episodio. Centralizar la
  semilla en `reset` evita estados inconsistentes a media partida.
* **`render_mode` fijado en `gym.make`.** Permite optimizar el pipeline de
  renderizado (rgb_array vs ventana real) y elimina cambios de modo en
  caliente que algunos entornos no soportaban.
* **Atari movido a `ale-py`.** El paquete `atari_py` ya no se mantiene; los
  ROMs y la interfaz viven ahora en `ale-py`, y los entornos se registran
  como `ALE/<Juego>-v5`.

## Instalación rápida

```bash
pip install -r requirements.txt
```

`requirements.txt` instala Gymnasium con los extras `box2d`, `atari` y
`classic-control`, además de `ale-py`, `shimmy` (para correr entornos
heredados gym v0.21/v0.26), PyTorch, TensorBoardX y utilidades de notebook.

## Cambios concretos en este repositorio

### `tema1`
* `gym_step.py` ahora usa `ALE/Qbert-v5` y registra los entornos ALE/* a
  través de `ale_py`.
* `gym_environments.py` itera sobre `gym.envs.registry.keys()` (en
  Gymnasium el registro es un `dict`, no se usa `.all()`).
* `run_gym_environment.py` y `gym_install_test.py` usan la nueva tupla de 5
  elementos en `step` y desempaquetan `(obs, info)` en `reset`.

### `tema2`
* `mountain_car_rand.py` y `mountain_car_qlearner.py` usan
  `terminated`/`truncated` en sus bucles de entrenamiento y prueba.
* La grabación de vídeo usa `gymnasium.wrappers.RecordVideo` con un entorno
  creado en `render_mode="rgb_array"`.

### `tema3`
* `environments/atari.py`: todos los wrappers (`NoopResetEnv`,
  `FireResetEnv`, `EpisodicLifeEnv`, `MaxAndSkipEnv`, `FrameStack`,
  `AtariRescale`, `NormalizedEnv`, `ClipReward`) implementan ahora las
  firmas nuevas de `reset(*, seed=None, options=None) -> (obs, info)` y
  `step(action) -> (obs, reward, terminated, truncated, info)`.
* La función `get_games_list()` antes delegaba en `atari_py.list_games()`;
  ahora deriva la lista del registro de Gymnasium filtrando los IDs
  `ALE/*` y normalizándolos a `snake_case`.
* La clave heredada `info["ale.lives"]` ya no existe en `ale-py >= 0.8`;
  el helper `_get_lives(info)` consulta primero `info["lives"]`.
* `SwallowQLearner.py` cambió de `CartPole-v0` (retirado) a `CartPole-v1`
  y actualizó el bucle de entrenamiento.
* `DeepQLearner.py` actualizado al `step` de 5 valores y mantiene un
  `done = terminated or truncated` para alimentar el `Experience` replay.

### `tema4`
* `custom_environment_template.py` ahora hereda de `gym.Env` con la nueva
  API: `metadata = {"render_modes": [...], "render_fps": ...}`,
  `reset(*, seed=None, options=None)` (con `super().reset(seed=seed)`),
  `step` que devuelve cinco valores, y `render()` sin argumento `mode`.
  El `gym.spaces.Box(4)` original era inválido y se ha sustituido por
  `gym.spaces.Discrete(4)` como ejemplo.
* `__init__.py` corrige el `entry_point` (antes era `":CustomEnvironment"`,
  que no se podía importar) apuntando al módulo de la plantilla.

### `tema5`
* `a2c.py` usa Gymnasium, registra los entornos ALE/* y selecciona el
  `render_mode` (`"human"` con `--render`, si no `"rgb_array"`). El bucle
  de entrenamiento maneja `terminated`/`truncated` y mantiene el `done`
  consolidado para la lógica de aprendizaje.
* `environments/atari.py` y `environments/utils.py` están sincronizados con
  los de `tema3`.

## ¿Y si necesito ejecutar código antiguo sin migrarlo?

Para entornos legacy puedes usar [shimmy](https://shimmy.farama.org/):

```python
import gymnasium
import shimmy  # noqa: F401  (con su `register_envs` ya hecho al importar)

# Entornos estilo gym v0.21
env = gymnasium.make("GymV21Environment-v0", env_id="CartPole-v1")
# Entornos estilo gym v0.26
env = gymnasium.make("GymV26Environment-v0", env_id="CartPole-v1")
```

También tienes los conversores
`gymnasium.utils.step_api_compatibility.convert_to_terminated_truncated_step_api`
y `convert_to_done_step_api` para envolver entornos personalizados.

## Validar tu propio entorno

Si modificas la plantilla de `tema4` o creas tu propio entorno, valida que
respeta el contrato de Gymnasium con:

```python
from gymnasium.utils.env_checker import check_env

check_env(MyEnv())
```
