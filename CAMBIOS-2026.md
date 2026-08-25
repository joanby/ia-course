# Cambios 2026

Este documento explica qué se ha tocado en el código del *Curso Completo de
Inteligencia Artificial con Python* para que vuelva a ejecutarse en 2026, y
por qué. No es un diff: es la explicación de qué decidió cada librería entre
2019 (cuando se grabó el curso) y ahora, y qué se ha hecho al respecto.

Todo lo que hay en esta rama son las mismas clases del vídeo. La mayor parte
de lo que sigue cambia **cómo se le pide algo a una librería**, no lo que
obtienes.

Hay, eso sí, una segunda mitad de este documento que no habla de librerías.
Habla de cosas que se rompieron solas dentro del propio repositorio: rutas
que apuntaban al Google Drive de una sesión de Colab de 2019, un fichero
duplicado esperando a desfasarse, y un typo de una letra que llevaba seis
años matando un proceso sin que nadie pudiera verlo. Esa parte es la que más
enseña, y por eso se cuenta con el mismo detalle que la otra.

## Versiones con las que se ha verificado

Verificado en **agosto de 2026**, en **macOS con Apple Silicon**:

| Paquete | Versión |
|---|---|
| Python | 3.13.15 |
| numpy | 2.5.2 |
| matplotlib | 3.11.1 |
| opencv (`cv2`) | 5.0.0 |
| pytorch | 2.13.0 |
| gymnasium | 1.3.0 |
| ale-py | 0.12.1 |
| tensorboardX | 2.6.5 |
| moviepy | 2.2.1 |
| pygame-ce | 2.5.8 |
| box2d | 2.3.10 (+ swig 4.5.0) |
| jupyter / nbconvert | 7.17.1 |

Cinco de esas líneas son **dependencias nuevas** que el curso original no
instalaba: `gymnasium` y `ale-py` sustituyen a `gym` y `atari_py`, y
`moviepy`, `pygame-ce` y `box2d` hacen falta por consecuencias concretas de
esa sustitución que se explican más abajo. El entorno completo está en
[`iacourse.yml`](iacourse.yml), en la raíz de esta rama.

Si usas otro sistema operativo o instalas versiones distintas de estas
librerías, es posible que algo de lo que se cuenta aquí ya no aplique tal
cual, para bien o para mal.

---

# Lo que rompieron las librerías

## 1. `gym` se quedó sin mantenimiento y su heredero es `gymnasium`

**Afecta a 25 de los 37 ficheros del curso** (16 `.py` + los 9 notebooks).
Es, con diferencia, el cambio central de esta actualización: todo lo demás
que hay en este documento es más pequeño o más local.

**Qué se rompía.** `import gym` — la primera línea de medio curso — moría
con `ModuleNotFoundError: No module named 'gym'`, sencillamente porque el
paquete ya no forma parte de ningún entorno moderno razonable.

**Antes de migrar se midió la alternativa, y funcionaba.** No damos esto por
supuesto: se probó de verdad congelar `gym` en su versión de la época
(`gym==0.25.2`) sobre Python 3.13, y **funciona**, con un único pin extra
(`numpy<2`, porque `gym` usa `np.bool8`, que NumPy 2 eliminó). Cero líneas
de código tocadas. Era la opción barata.

Se descartó de todas formas, y conviene que sepas por qué: **`gym` está sin
mantenimiento desde 2022** — lo dice el propio paquete por consola al
importarse — y su desarrollo continuó en otro sitio, bajo el nombre
`gymnasium`, mantenido por la Fundación Farama. Congelarlo habría dejado el
curso ejecutándose hoy a cambio de enseñarte una API muerta, con una
caducidad puesta y sin ningún camino de salida. Se prefirió pagar el coste
una vez.

**Qué cambió la librería, y por qué merece la pena entenderlo.** El grueso
de la migración no son los nombres: los identificadores de entorno que usa
el curso (`MountainCar-v0`, `CartPole-v0`, `Qbert-v0`,
`SeaquestNoFrameskip-v4`, `PongNoFrameskip-v4`) siguen existiendo con el
mismo nombre y se comprobó uno por uno. Lo que cambió es **el contrato de
`reset()` y `step()`**, las dos funciones que aparecen en todos los bucles
de entrenamiento del curso:

```python
obs = env.reset()                          →  obs, info = env.reset()
obs, reward, done, info = env.step(a)      →  obs, reward, terminated, truncated, info = env.step(a)
                                              done = terminated or truncated
```

`reset()` pasó a devolver una tupla y `step()` pasó de 4 valores a 5.

**Por qué `done` se partió en dos.** Esta es la parte interesante, y no es
burocracia de la librería: es una distinción real del aprendizaje por
refuerzo que la API vieja tenía escondida.

El `done` de siempre metía en un solo booleano dos situaciones que no se
parecen en nada:

- **El episodio se acabó de verdad**: el CartPole se cayó, el submarino de
  Seaquest se quedó sin vidas. El propio problema dice que ahí no hay futuro.
- **Al episodio lo cortó un reloj externo**: `MountainCar-v0` trae un límite
  de 200 pasos que no es parte del problema, es una decisión de quien montó
  el entorno para que las cosas terminen alguna vez.

Para el alumno que mira el bucle, ambas cosas eran `done = True`. Pero para
el agente que aprende, la diferencia es enorme. El agente estima cuánto vale
un estado por la recompensa que espera obtener **a partir de ahí**; cuando
un episodio termina de verdad, ese valor futuro es cero, y así hay que
enseñárselo. Cuando el episodio se corta por el reloj, el valor futuro **no
era cero** — el coche seguía subiendo la cuesta, simplemente no le dejamos
llegar. Si le enseñas al agente que ese estado valía cero, le estás mintiendo
sobre el problema, y el error se propaga hacia atrás por toda la estimación.

`gymnasium` hizo explícita esa distinción: `terminated` (el propio problema
dice que se acabó) y `truncated` (algo externo lo cortó).

**Qué se hace ahora.** El vídeo teclea `done`, así que **`done` se queda**:
en cada bucle se añade una línea, `done = terminated or truncated`, y el
resto del código sigue viendo exactamente el mismo booleano que ves en
pantalla. El coste es de una línea nueva por cada punto donde el curso llama
a `step()`; la ventaja es que quien quiera ir más allá tiene ya los dos
conceptos delante, con su nombre.

**Un fichero se llevó la peor parte: `environments/atari.py`** (existe por
duplicado, en tema3 y tema5). No es un fichero que use `gym`: es un fichero
que **reimplementa a mano cinco wrappers de Atari** (`NoopResetEnv`,
`FireResetEnv`, `EpisodicLifeEnv`, `MaxAndSkipEnv` y `FrameStack`), cada uno
con su propio `step()` y su propio `reset()`. Son diez métodos por fichero,
y en cada uno hay que migrar las dos direcciones: lo que se recibe del
wrapper de dentro y lo que se le devuelve al de fuera.

Curiosamente, los tres wrappers que **no** hubo que tocar
(`AtariRescale`, `NormalizedEnv`, `ClipReward`) son los que heredan de
`gym.ObservationWrapper`/`RewardWrapper` en vez de reimplementar el método:
la clase base de `gymnasium` ya resuelve el contrato nuevo por ellos. Es una
lección de diseño gratis — lo que delegas en la librería, la librería lo
migra por ti.

**Y un cambio menor de la misma familia:** `tema1/gym_environments.py` hacía
`envs.registry.all()` para listar los entornos disponibles. En `gymnasium`,
`registry` es un diccionario normal, así que la línea equivalente es
`envs.registry.values()`.

## 2. `atari_py` ya no se instala; `ale-py` sí, y trae las ROMs dentro

**Afecta a:** `tema3/environments/atari.py` y `tema5/environments/atari.py`
(y, por arrastre, a `DeepQLearner.py` y `a2c.py`, que llaman a su
`get_games_list()` nada más arrancar).

**Qué se rompía.** `import atari_py`, un import independiente del de `gym` y
que por tanto sobrevivía a arreglar el anterior. `atari_py` no tiene ruedas
(*wheels*) para ningún Python posterior al 3.7 — comprobado, no sospechado:
forzando solo binarios, el instalador responde que las únicas disponibles
son para `cp35`, `cp36` y `cp37`. Y compilarlo desde el código fuente exige
`cmake` y no garantiza nada, con el paquete abandonado desde 2020.

**Qué se hace ahora.** Se sustituye por **`ale-py`**, el paquete oficial del
Arcade Learning Environment que mantiene hoy la Fundación Farama. La única
línea de código que cambia es la que lista los juegos:
`atari_py.list_games()` pasa a `ale_py.roms.get_all_rom_ids()`, que devuelve
la misma convención de nombres (minúsculas, con `_` de separador:
`space_invaders`, no `spaceinvaders`), así que la línea de al lado que
consume ese listado no necesitó ni tocarse.

**La buena noticia, que hace unos años no lo era.** Entre 2020 y 2022, hacer
funcionar Atari implicaba un paso extra: descargar las ROMs por tu cuenta
con `AutoROM` y aceptar una licencia. Hoy **`ale-py` las trae incluidas en
el propio paquete**: no hay descarga aparte, ni licencia que aceptar, ni
carpeta que configurar. Se instala y `Seaquest` funciona.

## 3. `info['ale.lives']` pasó a llamarse `info['lives']` — sin dar ningún error

**Afecta a:** el wrapper `EpisodicLifeEnv` de `environments/atari.py`, en
tema3 y tema5.

Este merece su propio apartado, aunque sea un renombrado de una clave de
diccionario, porque es el tipo de cambio más peligroso que existe: **el que
no rompe nada**.

`EpisodicLifeEnv` hace algo muy concreto y muy útil para entrenar: le hace
creer al agente que **cada vida perdida es el final de un episodio**, en vez
de tratar la partida entera de Atari como un episodio único. Así el agente
recibe la señal de "esto ha ido mal" en el momento en que muere, no cinco
vidas después. Para saber cuándo ha perdido una vida, lee el contador que le
pasa el emulador dentro del diccionario `info`.

En el ALE moderno esa clave ya no lleva el prefijo `ale.`: es `info['lives']`.
Sin este cambio, el wrapper **no habría lanzado ninguna excepción**: en la
versión moderna de la librería la clave con el nombre viejo simplemente no
existe, el wrapper habría dejado de detectar la pérdida de vidas en
silencio, y el entrenamiento habría seguido corriendo, más lento en aprender
y sin una sola línea roja que lo explicara.

**Verificado ejecutando, no leyendo la documentación:** con el wrapper
migrado, 2.000 pasos aleatorios en `SeaquestNoFrameskip-v4` detectan la
primera pérdida de vida en el paso 68 (de 4 vidas a 3) y siguen
detectándolas de forma consistente el resto del episodio.

## 4. `gym.wrappers.Monitor` ya no existe: ahora es `RecordVideo`

**Afecta a:** `tema2/mountain_car_qlearner.py` y su notebook gemelo — la
clase donde se graban en vídeo los intentos del agente subiendo la montaña.

**Qué se rompía.** `gym.wrappers.Monitor` se retiró. Su sustituto es
`gym.wrappers.RecordVideo`, que hace lo mismo pero pide una cosa que
`Monitor` no pedía: **el entorno tiene que haberse creado con
`render_mode="rgb_array"`**. La razón es coherente con el resto de
`gymnasium`: el modo de renderizado dejó de ser algo que se decide sobre la
marcha y pasó a ser una propiedad del entorno, fijada al crearlo.

**Dos dependencias nuevas, y no son opcionales.** Grabar vídeo hoy arrastra
dos paquetes que `Monitor` no necesitaba:

- **`moviepy`**, que escribe el `.mp4`. Sin él, `RecordVideo` ni siquiera
  llega a construirse.
- **`pygame-ce`**, que dibuja los fotogramas. Esto sorprende: en los
  entornos de *classic control* (el `MountainCar` de esta clase), incluso
  `render_mode="rgb_array"` —que no abre ninguna ventana— pasa por pygame
  para pintar. No es solo cosa del modo `"human"`.

Las dos están en `iacourse.yml`, así que no tienes que hacer nada.

**Comprobado que graba de verdad, no solo que la llamada no falla.** Tras la
ejecución completa (50.000 episodios de entrenamiento más 1.000 de prueba,
152 segundos), los 10 `.mp4` generados en `monitor_output/` se leyeron con
`ffprobe`: todos entre 27 y 36 KB, con duración real de entre 3,7 y 4,8
segundos. Ficheros de vídeo válidos, no ficheros vacíos con la extensión
correcta. Los de la prueba se borraron después; no forman parte del repo.

## 5. Box2D, una dependencia nueva para `LunarLander`

`LunarLander` (el módulo lunar que hay que posar entre dos banderas, y uno
de los dos modelos entrenados que trae tema5) necesita hoy el motor de
físicas **Box2D**, que hay que instalar aparte. Con `gym` no hacía falta.

Se instala con `pip install "gymnasium[box2d]"`, que resuelve a
`box2d==2.3.10` y `swig==4.5.0`, ambos con rueda precompilada para
macOS/arm64 — no hay que compilar nada. Ya está incluido en `iacourse.yml`.

Solo hace falta si quieres usar `LunarLander-v3`. El resto del curso,
incluido el entorno que tema5 usa por defecto (`Pendulum-v1`), funciona sin
él.

## 6. Cuatro roturas más, todas escondidas detrás del `import gym`

Estas cuatro no se ven leyendo el código: aparecen solo cuando arreglas el
`import` y el fichero llega, por fin, a ejecutarse de verdad.

- **`torch.load()` ya no carga cualquier cosa** (`DeepQLearner.py`,
  `a2c.py`). Desde PyTorch 2.6, `weights_only=True` es el valor por defecto:
  una carga restringida que rechaza ficheros capaces de ejecutar código
  arbitrario al abrirse. Es un cambio de seguridad deliberado, y muy
  sensato — pero los checkpoints de 2018 del curso serializan escalares de
  numpy y no pasan por ese filtro: `_pickle.UnpicklingError: Weights only
  load failed`. Se ha puesto `weights_only=False` explícito, porque estos
  ficheros vienen del propio repositorio del curso y sabemos de dónde salen.
  Si algún día cargas un checkpoint que te has bajado de internet, ese es
  exactamente el caso en el que **no** debes hacer esto.

- **`isinstance(action_space.sample(), int)` dejó de detectar las acciones
  discretas** (`a2c.py`). El fichero decidía si el entorno tenía acciones
  discretas (Atari, CartPole) o continuas (Pendulum) preguntándole a una
  muestra del espacio de acciones si era un `int` de Python. Bajo
  `gymnasium`, `Discrete.sample()` devuelve un `numpy.int64`, que no lo es.
  Resultado: **todos** los entornos habrían entrado por la rama continua, y
  el fichero habría reventado con un `IndexError` en
  `action_space.shape[0]` — un error incomprensible, a mucha distancia de
  su verdadera causa. Arreglado preguntando por el tipo del **espacio**
  (`isinstance(env.action_space, gym.spaces.Discrete)`), que es la pregunta
  que se quería hacer desde el principio.

- **`mp.set_start_method("spawn")` reventaba con "context has already been
  set"** (`a2c.py`). La versión moderna de `tensorboardX` fija el método de
  arranque de `multiprocessing` al construir su `SummaryWriter`, algo que en
  2018 no hacía. La llamada del fichero, unas líneas más abajo, pide
  exactamente el mismo valor (`"spawn"`), pero pedirlo dos veces es un error
  aunque coincida. Arreglado con `force=True`: no cambia qué método se usa,
  solo permite volver a pedir el que ya está puesto.

- **A `tema5/parameters.json` le faltaba la clave `"clip_reward"`**, que
  `environments/atari.py` da por hecha. Su fichero gemelo de tema3 sí la
  tiene. Esta no la rompió ninguna librería —es un defecto propio del repo
  desde 2018— pero se descubrió aquí, porque `a2c.py` moría antes de llegar
  a leerla. Añadida con el mismo valor que su gemelo (`true`).

---

# Lo que rompió la propia historia del repositorio

Nada de lo que viene ahora lo causó una librería. Lo causamos nosotros, en
revisiones anteriores de este mismo repositorio, y lleva años ahí. Se cuenta
con detalle porque es lo que mejor enseña de toda esta actualización: casi
todo son fallos que **no dan ningún error**, y ese es justamente el motivo
de que nadie los viera.

## 7. Los dos modelos entrenados del curso no se cargaban nunca

Este es el más gordo, y el que más tiempo le ha costado a la gente sin que
nadie lo supiera.

El curso te **regala dos agentes ya entrenados** en `tema3/trained_models/`
(`DQL_Seaquest-v0.ptm` y `DQL_PongNoFrameskip-v4.ptm`). Están en el
repositorio, se descargan cuando clonas, y existen precisamente para que
puedas ver a un agente jugar sin tener que entrenarlo tú.

`tema3/parameters.json` tenía esto:

```json
"save_dir": "/content/drive/My Drive/",
"load_dir": "/content/drive/My Drive/",
```

Eso es una carpeta de Google Drive montada dentro de una sesión de Google
Colab. Existió durante unas horas de 2019, en la máquina virtual de quien
grabó la clase, y no ha vuelto a existir desde entonces en ningún ordenador
del mundo. Mientras tanto, los `.ptm` estaban ahí al lado, en el repositorio.

**Y ahora la parte que lo convierte en un problema serio.** La carga del
modelo vive dentro de un `try/except FileNotFoundError`. Así que no había
excepción, ni traza, ni salida distinta de cero. Lo que había era esto por
consola:

```
ERROR: no existe ningún modelo entrenado para este entorno. Empezamos desde cero
```

Es decir: el alumno leía que no existe ningún modelo entrenado —**cuando el
modelo estaba en su propio disco duro**— y se ponía a entrenar un agente de
Atari desde cero. Eso son horas o días de cómputo, dependiendo de la
máquina, para llegar a donde ya estaba antes de empezar.

Y como el proceso terminaba con código 0, **ninguna comprobación automática
lo habría detectado nunca**. Un verificador que solo mire el código de
salida ve un fichero verde. Hay que leer lo que imprime.

**Qué se hace ahora.** `save_dir` y `load_dir` apuntan a `trained_models/`,
la carpeta que existe de verdad, y `DeepQLearner.py` resuelve esa ruta
relativa al propio fichero (`os.path.dirname(os.path.abspath(__file__))`) en
vez de al directorio desde el que lo lances — así funciona igual si ejecutas
`python DeepQLearner.py` desde `tema3/` o desde cualquier otro sitio. Se
retiraron además dos variables `path = F"/content/drive/..."` que se
calculaban y no se usaban: restos muertos de aquella sesión de Colab.

`tema5/parameters.json` no tenía este problema: su ruta ya era relativa.

## 8. El entorno por defecto no coincidía con ningún modelo guardado

Arreglada la ruta, quedaba la otra mitad del mismo desencuentro, y esta
afecta a los dos temas.

`DeepQLearner.py` y `a2c.py` construyen el nombre del fichero de checkpoint
a partir del identificador del entorno. En los dos, el valor por defecto de
`--env` era `SeaquestNoFrameskip-v4`. Y en ninguno de los dos temas existe
un fichero guardado con ese nombre:

| Tema | Buscaba | Lo que hay guardado |
|---|---|---|
| tema3 | `DQL_SeaquestNoFrameskip-v4.ptm` | `DQL_Seaquest-v0.ptm`, `DQL_PongNoFrameskip-v4.ptm` |
| tema5 | `A2C_SeaquestNoFrameskip-v4.ptm` | `LunarLander`, `Pendulum` |

Con lo cual, incluso con la ruta correcta, **ejecutar el fichero a secas
seguía sin cargar nada** y volvía a caer en el mismo `try/except` silencioso
del punto anterior. Se ha cambiado el valor por defecto en los dos:

- **`tema3/DeepQLearner.py` → `Seaquest-v0`.** Es el mismo juego que ya
  sugería el valor anterior, solo con el sufijo que de verdad corresponde al
  fichero guardado: el cambio más pequeño posible. Se eligió sobre el otro
  candidato disponible (`PongNoFrameskip-v4`) porque su agente tiene
  recompensa media positiva (1,59, máxima 12,0) y se le ve sobrevivir y
  puntuar, mientras que el de Pong tiene una media de -20,2 sobre un mínimo
  posible de -21 — casi indistinguible de perder todas las partidas.

- **`tema5/a2c.py` → `Pendulum-v1`.** Se eligió sobre `LunarLander-v3`
  porque ejercita la política gaussiana continua
  (`multi_variate_gaussian_policy`), que es la aportación propia de A2C
  frente al Q-Learning discreto que ya has visto en los temas 2 y 3 — y
  porque, a diferencia de `LunarLander`, no necesita Box2D.

**Verificado ejecutando los dos sin ningún argumento**: ambos cargan ahora
un checkpoint real e imprimen su recompensa registrada antes de ponerse a
jugar, en vez de arrancar un entrenamiento desde cero.

## 9. Los identificadores de dos checkpoints de tema5 ya no existen

`tema5/trained_models/` guardaba `resultsA2C_LunarLander-v2.ptm` y
`resultsA2C_Pendulum-v0.ptm`. Esos dos identificadores de entorno ya no
existen — y no es como el aviso amable de "esta versión está anticuada" que
dan `CartPole-v0` o `Qbert-v0`: aquí `gym.make()` lanza directamente
`DeprecatedEnv` y te exige la versión nueva.

- `Pendulum-v0` → hoy es `Pendulum-v1`. Sin dependencias nuevas.
- `LunarLander-v2` → hoy es `LunarLander-v3`. Necesita Box2D (punto 5).

El problema es que **el nombre del fichero se deriva del mismo texto que
crea el entorno**: `"A2C_" + self.env_name + ".ptm"`. Un solo texto no puede
a la vez crear `Pendulum-v1` y nombrar un fichero `Pendulum-v0`.

Había dos salidas: separar "identificador del entorno" de "nombre del
fichero" —lo que exigía un parámetro nuevo y tocar `save()`, `load()` y la
construcción de `env_name`— o **renombrar los cuatro ficheros**. Se
renombraron: cero líneas de código tocadas.

```
resultsA2C_LunarLander-v2.ptm(.agent_params)  →  resultsA2C_LunarLander-v3.ptm(.agent_params)
resultsA2C_Pendulum-v0.ptm(.agent_params)     →  resultsA2C_Pendulum-v1.ptm(.agent_params)
```

**Los pesos se comprobaron ejecutando, no leyendo la documentación.** Un
checkpoint entrenado en una versión y cargado en otra puede fallar de forma
sutil si cambió la forma del espacio de observación o de acción; aquí no
cambió (`Pendulum`: `Box(3,)` y `Box(1,)`; `LunarLander`: `Box(8,)` y
`Discrete(4)`), y la primera capa lineal del actor habría reventado con un
error de dimensiones si no fuera así. No reventó, y no hubo que forzar nada:

- `--env Pendulum-v1 --test`: carga el checkpoint (media registrada
  -1419,02; máxima -1038,51) y completa un episodio real con recompensa
  entre -1450 y -1485 — el mismo orden de magnitud que sus propias
  estadísticas guardadas.
- `--env LunarLander-v3 --test`, 3 episodios: carga el checkpoint (media
  registrada -182,55; máxima -7,40) y produce episodios de entre -65 y -167.

Un apunte honesto sobre ese segundo agente: **`LunarLander` nunca estuvo
resuelto**. Se considera resuelto por encima de +200 de recompensa media, y
su mejor episodio registrado es -7,4. Es un agente a medio entrenar, y eso
ya era así en 2019 — los números que verás al ejecutarlo son coherentes con
eso, no con unos pesos rotos.

## 10. Un typo de 2018 que llevaba seis años siendo invisible

En `tema5/function_aproximator/deep.py` (cuatro veces) y en `swallow.py`
(una más) había esto:

```python
x.require_grad_()      # el método no existe; es requires_grad_
```

No es un cambio de PyTorch: **`require_grad_` no ha existido nunca**, en
ninguna versión. Es un typo del propio repositorio desde 2018. Y lanzaba
`AttributeError` en el primer paso de cada agente.

**¿Por qué no lo vio nadie en seis años?** Porque `a2c.py` lanza cada uno de
sus agentes en un proceso hijo con `multiprocessing.Process`, y luego hace:

```python
[p.join() for p in agent_procs]
```

`join()` espera a que el hijo termine. No comprueba **cómo** terminó. Si el
hijo revienta con una excepción, `join()` vuelve con toda normalidad, y el
proceso padre sale con **código 0**. Todo verde.

Así que ni el alumno lo veía —el mensaje del hijo se pierde entre el ruido
de la consola— ni ninguna herramienta automática podía verlo, porque todas
las herramientas automáticas miran el código de salida. Un fichero que
"funciona", durante seis años, sin haber entrenado jamás.

**Arreglado** en las cinco apariciones. Verificado leyendo la salida real,
no el código de retorno: antes, `a2c.py` moría en el primer
`action = self.get_action(obs)` con `AttributeError: 'Tensor' object has no
attribute 'require_grad_'`; después, completa episodios enteros con sus
recompensas registradas.

La regla que queda de aquí, y que va más allá de este curso: **si un fichero
usa `multiprocessing`, `subprocess` o hilos, su código de salida no prueba
absolutamente nada.** Hay que leer lo que imprime.

## 11. `params_manager_colab.py` era una copia exacta esperando a desfasarse

`tema3/utils/` tenía dos ficheros: `params_manager.py` y
`params_manager_colab.py`. El segundo existía porque su única diferencia era
una ruta absoluta de Colab escrita a mano
(`/content/ia-course/tema3/parameters.json`) donde el primero usaba una ruta
relativa. Y `DeepQLearner.ipynb` importaba el segundo mientras
`DeepQLearner.py` importaba el primero: **el notebook y el script no
ejecutaban el mismo código, y eso no lo detecta ninguna comparación con las
librerías de hoy.**

Al arreglar esa ruta, el fichero quedó **byte a byte idéntico** a su gemelo
(`diff` sin ninguna salida). A partir de ahí ya no defendía nada: era, sin
más, un segundo fichero con el mismo contenido, esperando a que alguien
tocara uno y se olvidara del otro.

Se ha retirado, y el notebook pasa a importar `params_manager` como todo lo
demás. Se comprobó con `grep` que nada más en el árbol lo referenciaba antes
de borrarlo.

## 12. Los 9 notebooks solo funcionaban dentro de Google Colab

Los nueve `.ipynb` del curso morían todos en la misma celda:

```
ModuleNotFoundError: No module named 'google.colab'
```

No era un problema de `gym`: era que estaban escritos para vivir
exclusivamente dentro de Colab. Cada uno empezaba clonándose el repositorio
entero (`!git clone`), montando Google Drive (`drive.mount`), instalando
paquetes con `!pip install`, y —en los de tema1, 2, 3 y 5— construyendo un
bloque entero de utilidades de vídeo (`pyvirtualdisplay`, `xvfb`,
`wrap_env`, `show_video`) que existe porque una máquina de Colab no tiene
pantalla. Fuera de Colab, todo eso es a la vez inútil y fatal.

**Qué se hace ahora**, siguiendo exactamente el mismo patrón que ya se
aplicó en el curso hermano *Deep Learning de A a la Z*:

- Todo el bloque de arranque se **colapsa en una sola celda condicionada**:

  ```python
  import sys
  if 'google.colab' in sys.modules:
      !git clone -b update-2026 https://github.com/joanby/ia-course.git
      %cd 'ia-course/temaN'
  ```

  En local es un no-op: no hace ninguna llamada de red y no cuesta nada. En
  Colab sigue clonando el repositorio y situándose en la carpeta correcta,
  igual que antes. **Los notebooks funcionan en los dos sitios.**

- Las llamadas a `wrap_env(...)` (del bloque de utilidades borrado) se
  sustituyen por la creación directa del entorno, `gym.make(...)`, igual que
  hace el `.py` gemelo. Las llamadas finales a `show_video()` se retiran: la
  función ya no existe, y en el único notebook que graba de verdad
  (`mountain_car_qlearner.ipynb`) el vídeo se sigue escribiendo a disco vía
  `RecordVideo`, solo que no se muestra incrustado.

- Los `sys.path.append('/content/ia-course/temaN/utils')` pasan a rutas
  relativas (`./utils/`, `./libs/`, `./environments/`).

- Las dos celdas `%tensorboard --logdir ...` de `DeepQLearner.ipynb`
  apuntaban a `/content/ia-course/tema3/logs` y `/content/logs/`. Ahora
  apuntan a `./logs`, que es donde el propio `parameters.json` dice que se
  escriben.

- Y reciben, donde toca, los mismos arreglos que sus `.py` gemelos: el
  contrato de `step()`/`reset()`, `Monitor` → `RecordVideo`, los valores por
  defecto de `--env`, y `weights_only=False`.

**Un hallazgo de propina, y del tipo que solo aparece al ejecutar.**
`tema2/mountain_car_qlearner.ipynb` usa `np.zeros`, `np.argmax` y
`np.random` en su clase `QLearner`… pero su **único** `import numpy as np`
vivía dentro del bloque de utilidades de Colab que se acababa de borrar. El
primer intento de verificación murió con `NameError: name 'np' is not
defined` en la línea que construye la tabla Q. Nada que ver con `gym`: un
import huérfano que quedó al descubierto al quitar el código muerto.
Arreglado, y revisado el mismo riesgo en los otros ocho.

---

# Lo que se ha quitado de la rama

Dos cosas que nunca debieron estar en el repositorio y que te descargas cada
vez que lo clonas:

- **`tema3/logs/` — 9 ficheros, 110 MB.** Tres directorios de ejecución de
  TensorBoard de una sesión de entrenamiento de **octubre de 2018**
  (`DQL_PongNoFrameskip-v418-10-23-16-05`, `DQL_Seaquest-v018-10-23-18-50`,
  `DQL_Seaquest-v018-10-24-08-37`), commiteados por accidente. Los tres
  ficheros `events.out.tfevents.*` se llevan 110.069.604 de esos 110.075.736
  bytes ellos solos. Ningún fichero del curso los lee: `logs/` es un destino de **escritura**,
  y `SummaryWriter` lo vuelve a crear él solo la primera vez que entrenas.
  Lo único que desaparece es el histórico de las ejecuciones de otra
  persona; tú verás las tuyas en `%tensorboard --logdir ./logs`.

- **`tema1/gym-master/` — 210 ficheros, 4,1 MB.** Una copia de la librería
  `gym` 0.10.5, descomprimida dentro del repositorio. Ningún import del
  curso la usaba: no está en el `sys.path` de nadie, y un `grep` sobre los
  37 ficheros del curso no la menciona ni una vez. Con el curso ya migrado a
  `gymnasium`, tener dentro una copia de la librería vieja solo podía
  inducir a error.

**Y ahora la parte honesta sobre lo que esto consigue.** Las dos siguen
existiendo en la rama `master`, y sus objetos siguen en el histórico de git.
Borrarlas en esta rama **alivia el disco que ocupa tu copia de trabajo
después del `checkout`, no lo que te descargas al clonar.** El `git clone`
sigue trayendo el histórico completo, con esos 114 MB dentro. Quitarlos de
verdad exigiría reescribir la historia del repositorio, que es otra decisión
y con otras consecuencias.

---

# Lo que puede verse distinto al vídeo

- **Todos los bucles de entrenamiento.** Donde el vídeo teclea
  `obs = env.reset()` y `obs, reward, done, info = env.step(action)`, aquí
  verás `obs, info = env.reset()`, cinco valores en el `step()`, y una línea
  nueva justo debajo: `done = terminated or truncated`. La variable `done`
  sigue existiendo y vale lo mismo; el resto del bucle es idéntico al de
  pantalla.

- **`import gym` es ahora `import gymnasium as gym`.** El alias es
  deliberado: a partir de esa línea, todas las demás (`gym.make`,
  `gym.spaces.Box`, `gym.Wrapper`…) se escriben exactamente igual que en el
  vídeo.

- **`DeepQLearner.py` y `a2c.py` arrancan con otro entorno por defecto**
  (`Seaquest-v0` y `Pendulum-v1`). Es el cambio del punto 8, y su efecto es
  el contrario de "distinto al vídeo": ejecutarlos a secas ahora enseña un
  agente ya entrenado jugando, que es lo que la clase quería enseñar.

- **La animación en pantalla, en varios ficheros.** En `gymnasium`, el modo
  de renderizado se fija al crear el entorno. Si un fichero llama a
  `.render()` sin que el entorno se haya creado con `render_mode="human"`,
  ya no se abre ninguna ventana: solo sale un aviso por consola. Se ha
  puesto `render_mode="human"` explícito en `tema1/gym_step.py` (ahí era
  obligatorio: con Atari, `.render()` sin modo no avisa, **revienta**). En
  `run_gym_environment.py` y `spaces.py` **no** se ha puesto, a propósito:
  esos dos reciben el identificador del entorno como argumento, así que no
  saben de antemano si les vas a pasar un `CartPole` o un juego de Atari, y
  fijar el modo por ti los rompería para la mitad de los casos. Si quieres
  ver la ventana, pásale `render_mode="human"` tú al `gym.make()`.

- **Los notebooks.** Su primera celda ya no es un `git clone` a pelo, sino
  el bloque condicionado del punto 12, y han perdido las utilidades de vídeo
  de Colab. Funcionan igual en Colab que antes y ahora también en local.

- **`a2c.py` avisa de `NaN or Inf found in input tensor`** si lo dejas
  entrenar sobre un checkpoint ya cargado. Se explica justo abajo.

---

# Lo que NO se ha verificado

Esta es la sección que más te conviene leer, porque es donde este documento
deja de prometer.

**`tema5/a2c.ipynb` no se ha podido ejecutar, y no se ha arreglado.** El
notebook define `DeepActorCriticAgent`, una subclase de
`multiprocessing.Process`, **dentro de una celda**. Y fuerza el método de
arranque `spawn`. Con `spawn`, el proceso hijo tiene que volver a importar
el módulo donde vive la clase para reconstruirla — y para un notebook ese
"módulo" es el lanzador interactivo de IPython, que no se puede reimportar.
Resultado:

```
AttributeError: Can't get attribute 'DeepActorCriticAgent' on <module '__main__' ...>
```

Tres cosas, en orden de importancia:

1. **Esto no lo ha causado esta actualización, ni tiene nada que ver con
   `gym`.** Es una limitación estructural de `multiprocessing` con clases
   definidas de forma interactiva, y se reprodujo con una clase de prueba de
   dos líneas sin nada de aprendizaje por refuerzo de por medio. El commit
   que añadió este notebook al repositorio, en 2018, se titula literalmente
   *"a2c colab, tensorboard not working"*: nunca hubo una versión de este
   fichero sin este problema.

2. **En Linux y en Colab, que es donde este notebook está pensado para
   correr, debería funcionar** — ahí `multiprocessing` arranca de otra
   forma y la clase se hereda sin reimportarla. Pero **no lo hemos
   comprobado**, y no vamos a decir que funciona sin haberlo visto. Esta
   verificación se ha hecho en macOS, donde probar el cambio de una palabra
   (`spawn` → `fork`) hace que el intérprete entero se caiga en cuanto
   PyTorch toca el backend Metal, que es precisamente el motivo por el que
   Python usa `spawn` en macOS.

3. **Su `.py` gemelo, `tema5/a2c.py`, sí está verificado y funciona.** Si
   estás en macOS y quieres ver A2C funcionando, ese es el fichero.

Y como este notebook cae exactamente en la trampa del punto 10 —`join()` sin
mirar el `exitcode`— **termina con código 0** aunque el hijo no haya
entrenado nada. Se cuenta aquí para que no lo confundas con una ejecución
correcta.

**Los entrenamientos largos se han verificado con una versión reducida.**
Igual que en los cursos hermanos, hay ficheros cuyo entrenamiento completo
se mide en horas o días y no es viable ejecutar en una revisión:

| Fichero | Lo que hace por defecto | Cómo se verificó |
|---|---|---|
| `tema3/DeepQLearner.py` | 1.000.000 de pasos de Atari | `parameters.json` reducido; se comprobó leyendo por consola que **el checkpoint real se carga** |
| `tema3/SwallowQLearner.py` | 100.000 episodios | copia reducida de 5 episodios |
| `tema1/gym_step.py` | 10 × 500 pasos con ventana en tiempo real | copia reducida de 1 episodio y 20 pasos |
| `tema5/a2c.py` | 25 episodios sobre el checkpoint cargado | ejecutado, pero sobre una **copia** fuera del repo, para no arriesgar el checkpoint que se distribuye |

Dicho sin maquillar: **sabemos que el código corre de principio a fin; no
hemos comprobado que un entrenamiento completo reproduzca los números del
vídeo.** Los cuatro sí se han ejecutado lo suficiente como para ver
recompensas reales y la carga del modelo correcto.

Sí se ejecutaron **completos**, sin reducir nada:
`tema2/mountain_car_rand.py` (1.000 episodios, 1,9 segundos) y
`tema2/mountain_car_qlearner.py` (50.000 episodios de entrenamiento, 1.000
de prueba y la grabación real de vídeo, 152 segundos, ejecutado tres veces
con el mismo resultado).

**`tema1/run_gym_environment.py` y `tema1/spaces.py`** reciben el entorno
por línea de comandos, así que ninguna herramienta automática puede
ejecutarlos sin decidir por ti qué entorno pasarles. Se verificaron a mano
con `MountainCar-v0`, `CartPole-v0` y `Qbert-v0`.

---

# Lo que sigue sin arreglar

Dos cosas, las dos conocidas y ninguna de las dos bloqueante:

- **La inestabilidad numérica de `a2c.py`.** Si lo dejas entrenar sobre un
  checkpoint ya cargado (que es el comportamiento por defecto), `tensorboardX`
  avisa por consola `NaN or Inf found in input tensor.` — 199 veces en una
  ejecución de 25 episodios. No es una traza de error: el proceso completa
  los 25 episodios y sale con código 0. Es un aviso de que a la función de
  registro le ha llegado algún valor no finito, probablemente en el cálculo
  de la pérdida o de la entropía de la política gaussiana continua.
  **Se comprobó que no corrompe el modelo**: se inspeccionaron uno a uno
  todos los tensores del `state_dict` guardado tras ese entrenamiento
  (`torch.isnan` / `torch.isinf`) y ninguno es `NaN` ni `Inf`.
  No se ha investigado su causa raíz: no es parte del contrato de
  `gym`/`gymnasium` y no impide que el fichero se ejecute. Pero conviene
  saber que **antes era imposible de ver**, porque con el typo del punto 10
  `a2c.py` nunca llegaba a entrenar de verdad.

- **El renderizado en pantalla de los ficheros genéricos**
  (`run_gym_environment.py`, `spaces.py`), por el motivo explicado en "lo
  que puede verse distinto al vídeo": no se decide por ti qué tipo de
  entorno vas a pasarles.

---

¿Dudas sobre algún cambio? Este documento intenta explicar el porqué, no
solo el qué — si algo no cuadra, dínoslo.
