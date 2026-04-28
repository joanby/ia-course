# Guía de instalación con `uv`

Este curso usa [Gymnasium](https://gymnasium.farama.org/) y PyTorch. La forma
recomendada de prepararlo es con [`uv`](https://github.com/astral-sh/uv), un
gestor de paquetes / entornos virtuales en Rust mucho más rápido que
`pip`/`venv` (10–100×). `uv` reemplaza a `python -m venv`, `pip`, `pip-tools`
y `pyenv` con un único binario.

> **TL;DR**
>
> ```bash
> curl -LsSf https://astral.sh/uv/install.sh | sh    # instala uv
> uv venv --python 3.12 .venv                        # crea el entorno
> source .venv/bin/activate                          # actívalo
> uv pip install -r requirements.txt                 # instala dependencias
> python tema1/gym_install_test.py                   # probar :)
> ```

---

## 1. Instalar `uv`

### macOS / Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

El script copia el binario a `~/.local/bin/uv` (o `~/.cargo/bin/uv`) y añade
ese directorio a tu `PATH` editando tu shell rc (`~/.zshrc`, `~/.bashrc`,
etc.). Abre una terminal nueva o ejecuta `source ~/.zshrc` para recargar el
PATH.

### Windows (PowerShell)

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Otras vías

| Vía              | Comando |
| ---------------- | ------- |
| Homebrew (macOS) | `brew install uv` |
| pipx             | `pipx install uv` |
| pip              | `pip install uv` |
| Cargo            | `cargo install --git https://github.com/astral-sh/uv uv` |

### Verificar

```bash
uv --version
# uv 0.5.x (xxxxxxxxxx)
```

Para mantenerlo al día:

```bash
uv self update
```

---

## 2. Clonar el repo y entrar en el directorio

```bash
git clone https://github.com/joanby/ia-course.git
cd ia-course
```

(Si ya lo tienes clonado, simplemente `cd` a la carpeta del repo.)

---

## 3. Crear un entorno virtual

`uv` crea entornos virtuales con `uv venv`. Por defecto usa la versión de
Python que tengas como `python3` en el sistema, pero puedes pedir una
concreta — y si no la tienes instalada, `uv` se la descarga sola.

```bash
uv venv --python 3.12 .venv
```

> Gymnasium soporta Python 3.10 – 3.13. Para Atari/Box2D recomiendo 3.11 o 3.12.

Salida típica:

```
Using CPython 3.12.5
Creating virtualenv at: .venv
Activate with: source .venv/bin/activate
```

Activa el entorno:

```bash
# bash / zsh (macOS, Linux)
source .venv/bin/activate

# fish
source .venv/bin/activate.fish

# PowerShell (Windows)
.venv\Scripts\Activate.ps1
```

Sabrás que está activo porque tu prompt empieza por `(.venv)`.

> **Tip**: con `uv` no es estrictamente necesario activar el venv. Puedes
> ejecutar comandos dentro de él con `uv run <cmd>` (p. ej. `uv run python
> tema2/mountain_car_rand.py`). Activarlo es solo más ergonómico cuando vas
> a trabajar interactivamente.

---

## 4. Instalar las dependencias del curso

Con el venv activo (o usando `uv run` por delante):

```bash
uv pip install -r requirements.txt
```

Esto instala, según [`requirements.txt`](requirements.txt):

- `gymnasium[box2d,atari,classic-control]` — el RL framework moderno.
- `ale-py` — backend Atari (sustituye al antiguo `atari_py`).
- `shimmy` — adaptadores para entornos `gym` v0.21/v0.26 antiguos.
- `torch`, `torchvision` — DL backend para los temas 3 y 5.
- `numpy`, `opencv-python`, `tensorboardX`, `tqdm` — utilidades.
- `pyvirtualdisplay`, `matplotlib` — visualización en Colab y entornos sin pantalla.

`uv` instala todo en paralelo y resuelve dependencias mucho más rápido que
`pip`. La primera vez tarda unos segundos; las siguientes se beneficia de la
caché global (`~/.cache/uv/`).

> **PyTorch + GPU.** El `torch` que viene en `requirements.txt` es la versión
> CPU/CUDA por defecto de PyPI. Si quieres una build concreta de CUDA (p. ej.
> CUDA 12.4), instálala desde el índice oficial *después* del paso anterior:
>
> ```bash
> uv pip install --index-url https://download.pytorch.org/whl/cu124 \
>                torch torchvision
> ```

### Atari ROMs

A partir de `ale-py >= 0.8` los ROMs vienen empaquetados con la librería; no
hay que correr `AutoROM` ni aceptar licencias por separado.

### MuJoCo (opcional, para los entornos avanzados de tema5)

Si quieres MuJoCo:

```bash
uv pip install "gymnasium[mujoco]"
```

### Robotics (opcional)

Para `FetchReach-v3` y similares:

```bash
uv pip install gymnasium-robotics
```

---

## 5. Validar la instalación

```bash
python -c "import gymnasium as gym; import ale_py; gym.register_envs(ale_py); \
           env = gym.make('CartPole-v1'); obs, info = env.reset(seed=42); print(obs, info)"
```

Salida esperada:

```
[ 0.0273956 -0.00611216  0.03585979  0.0197368 ] {}
```

Prueba un script del curso:

```bash
python tema2/mountain_car_rand.py
```

Debería abrir una ventana con el `MountainCar-v0`. Ciérrala con `Ctrl+C`.

Si no tienes pantalla (servidor remoto, WSL sin X, etc.) lanza primero un
display virtual:

```bash
uv pip install pyvirtualdisplay
xvfb-run -a python tema2/mountain_car_rand.py
```

---

## 6. Jupyter / notebooks

Para abrir los `.ipynb` del curso localmente:

```bash
uv pip install jupyterlab
jupyter lab
```

O en Colab — los notebooks ya tienen un `!pip install gymnasium[box2d,atari]`
en sus primeras celdas, así que se autoinstalan.

---

## 7. Comandos `uv` útiles

| Acción | Comando |
| ------ | ------- |
| Ver paquetes instalados | `uv pip list` |
| Mostrar dependencias de un paquete | `uv pip show gymnasium` |
| Actualizar un paquete | `uv pip install --upgrade gymnasium` |
| Congelar el venv a un lock | `uv pip freeze > requirements.lock` |
| Reinstalar todo desde cero | `rm -rf .venv && uv venv --python 3.12 .venv && uv pip install -r requirements.txt` |
| Ejecutar sin activar el venv | `uv run python script.py` |
| Vaciar la caché global | `uv cache clean` |

---

## 8. Resolución de problemas

### `uv: command not found`

Tu shell no ha recargado el PATH. Cierra y abre una terminal nueva, o haz
`source ~/.zshrc` (o `~/.bashrc`).

### `error: Failed to build wheel for box2d-py`

`box2d` necesita un compilador C++. En macOS:

```bash
xcode-select --install
```

En Ubuntu/Debian:

```bash
sudo apt-get install -y build-essential swig
```

Luego repite `uv pip install -r requirements.txt`.

### `ModuleNotFoundError: No module named 'gymnasium'`

Estás usando un Python distinto al del venv. Verifica con:

```bash
which python && python -c "import sys; print(sys.executable)"
```

Si no apunta a `.venv/bin/python`, vuelve a activarlo (`source
.venv/bin/activate`) o usa `uv run python ...`.

### El render se queda en negro / no se abre la ventana (Linux/WSL)

Necesitas un servidor X o un display virtual:

```bash
sudo apt-get install -y xvfb
xvfb-run -a python tema2/mountain_car_rand.py
```

En notebooks, la celda `from pyvirtualdisplay import Display; Display(...)`
ya cubre este caso.

---

## 9. ¿Y si quiero seguir usando `pip`/`conda`?

Funciona, pero es más lento y tienes que gestionar tú las versiones de
Python. Equivalentes:

```bash
# pip + venv
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# conda
conda create -n ia-course python=3.12
conda activate ia-course
pip install -r requirements.txt
```

Para cambios en la API entre el `gym` v0.21 original del curso y la
`gymnasium` actual, mira [`MIGRATION.md`](MIGRATION.md).
