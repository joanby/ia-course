# [Curso completo de Inteligencia Artificial con Python](https://www.udemy.com/curso-completo-de-inteligencia-artificial/?couponCode=GITHUB_PROMO)

**Última actualización**: Agosto de 2026

---

## 🆕 Estás en la rama `update-2026`

Esta es la versión del código **puesta al día y verificada en agosto de
2026**, con Python 3.13 y las librerías tal como están hoy. El curso se grabó
en **2018-2019** —su código entra en el repositorio entre septiembre y
noviembre de 2018—, sobre `gym` — la librería de OpenAI para entornos de aprendizaje
por refuerzo — y `gym` lleva sin mantenimiento desde 2022. Aquí el código
vuelve a ejecutarse de principio a fin.

**En qué se diferencia de `master`:**

- **`master` no ha cambiado y no va a cambiar**: conserva el código **tal y
  como se grabó el curso**. Si quieres ver exactamente lo que el profesor
  teclea en pantalla, es
  [ahí](https://github.com/joanby/ia-course/tree/master) donde está.
- **Esta rama migra `gym` a
  [`gymnasium`](https://gymnasium.farama.org/)**, su continuación mantenida
  por la Fundación Farama, y `atari_py` a `ale-py`. Los identificadores de
  entorno del curso (`MountainCar-v0`, `CartPole-v0`, `Qbert-v0`,
  `SeaquestNoFrameskip-v4`…) **no cambian de nombre**: lo que cambia es cómo
  se desempaqueta lo que devuelven `reset()` y `step()`.
- **Los dos agentes ya entrenados de `tema3` ahora se cargan de verdad.** En
  `master` no llegan a cargarse: el fichero de configuración apunta a una
  carpeta de Google Drive de una sesión de Colab de **septiembre de 2020**,
  escrita por un contribuidor externo en un *pull request* (antes de eso, en
  2018, la ruta era correcta). Así que ejecutar la clase arranca un
  entrenamiento desde cero de horas sin decir por qué. Está contado con
  detalle, con las fechas y los commits, en `CAMBIOS-2026.md`.
- **Los 9 notebooks están aquí, y 8 de los 9 se ejecutan en local.** En
  `master` solo funcionaban dentro de Colab: montaban Google Drive y se
  clonaban el repositorio a sí mismos. Ahora todo ese arranque vive en una
  única celda condicionada, que en Colab hace lo de siempre —clonar,
  situarse en la carpeta e instalar lo que Colab no trae— y en local no hace
  nada. La excepción es `tema5/a2c.ipynb`, que se explica más abajo.
- **La plantilla de `tema4` ya enseña el contrato de `gymnasium`.**
  `custom_environment_template.py` es el fichero del que partes para
  escribir tu propio entorno, y sus docstrings seguían documentando el
  `step()` de cuatro valores que ya no existe. Ahora documentan
  `(observation, reward, terminated, truncated, info)`, un `reset()` que
  devuelve `(observation, info)` y un `render()` sin parámetros, que es como
  lo pide `gymnasium`.
- **Esta rama no trae `tema1/gym-master/` ni `tema3/logs/`**: una copia
  vendorizada de `gym` 0.10.5 que ningún fichero del curso importaba, y 110 MB
  de registros de TensorBoard de una sesión de entrenamiento de octubre de
  2018. Las dos siguen en `master`.

**Qué ha cambiado exactamente y por qué**: está todo explicado, causa por
causa, en [`CAMBIOS-2026.md`](CAMBIOS-2026.md) — incluidas las versiones
exactas con las que se ha verificado, lo que puede verse distinto al vídeo y
lo que **no** se ha llegado a comprobar. Léelo antes de reportar como error
algo que ya esté documentado ahí.

---

## 🐍 Cómo montar el entorno

El curso usa un entorno conda. Con
[Miniforge](https://github.com/conda-forge/miniforge) o Anaconda instalado:

```bash
conda env create -f iacourse.yml
conda activate iacourse
```

Eso te deja Python 3.13 con numpy, matplotlib, OpenCV, PyTorch y Jupyter,
más `gymnasium` con sus extras `[box2d,classic-control]`, `ale-py` para los
juegos de Atari, `moviepy` para grabar los vídeos de los agentes, y
`tensorboard`/`tensorboardX` para ver las curvas de entrenamiento. Box2D y
`pygame-ce` no aparecen como líneas propias del `.yml`: llegan por esos dos
extras de `gymnasium`.

Dos cosas que antes daban guerra y ahora no: **las ROMs de Atari vienen
incluidas en `ale-py`** (no hay que ejecutar `AutoROM` ni aceptar ninguna
licencia), y **Box2D se instala con rueda precompilada**, sin compilar nada.

Las versiones exactas con las que se ha verificado esta rama están en la
tabla de
[`CAMBIOS-2026.md`](CAMBIOS-2026.md#versiones-con-las-que-se-ha-verificado).

---

## 📓 Los notebooks

Cada tema trae sus `.ipynb` al lado de los `.py`:

- **En local**, con el entorno de arriba: `jupyter notebook` desde la carpeta
  del tema y a correr. **Verificados 8 de los 9** — la excepción está justo
  aquí abajo.
- **En Google Colab**, igual que siempre: la primera celda detecta que está
  en Colab, se clona esta rama, se sitúa en la carpeta del tema e instala lo
  que Colab no trae de serie: `gymnasium` en los nueve (nadie más lo
  arrastra), más `ale-py`, `tensorboardX` o `moviepy` en los que los usan.

Y ahora la parte honesta, que son dos:

- **Nada de esto se ha ejecutado en Colab.** Toda la verificación se ha
  hecho en local, en macOS. Los notebooks *deberían* funcionar en Colab,
  porque la celda de arranque hace allí lo mismo que hacía antes más las
  instalaciones que faltaban, pero no lo hemos comprobado y no vamos a decir
  que funciona algo que no hemos visto funcionar.
- **`tema5/a2c.ipynb` no se ha podido verificar ni siquiera en local**, por
  un motivo que no tiene nada que ver con esta actualización y que se explica
  en `CAMBIOS-2026.md`. Su gemelo `tema5/a2c.py` sí está verificado.

---


Bienvenido Curso completo de Inteligencia Artificial de cero a experto, donde aprenderás conceptos claves del mundo de la IAy el aprendizaje automatizado tanto desde el punto de visto teórico como implementaciones prácticas con Python, en particular cubriremos aspectos como

* Introducción a la inteligencia artificial, con todos los conocimientos y terminología del sector.
* Construir tu primera IA sin experiencia previa de programación usando Python utilizando la ecuación de Bellman.
* Cómo combinar la inteligencia artificial con videojuegos con OpenAI Gym para aprender de forma efectiva.
* Técnicas de optimización de IA para alcanzar soluciones com máximo potencial en contextos reales.
* Redes neuronales desde el perceptrón simple hasta las redes neuronales de convolución para hacer que nuestro agente aprenda a jugar a la Atari clásica mirando la pantalla como lo haría un ser humano
* Toda la teoría explicada con transparencias, incluido Q-Learning, ecuación de Bellman, Redes Neuronales Artificiales y de Convolución, Entropía Cruzada o la función Softmax entre otras.
* Papers de referencia de toda la teoría para que complementes la formación del curso con los mismos papers de donde sale toda la parte teórica (ideal para los que están trabajando en el campo de la IA o escribiendo su propia tesis doctoral, pues hay muchas referencias en más de 30 artículos web diferentes).
* Y mucho más que trae el curso para que aprendas no solo los aspectos sencillos si no también todos los entresijos más avanzados del mundo de la inteligencia artificial con Python.

Y todo ello acompañado de lo mejor que tienen los cursos de Juan Gabriel Gomila:

* Soporte personalizado para todas las dudas del curso incluyendo un foro del curso donde intentaré responderte antes de 48 horas a tus dudas y una comunidad de Discord con miles de estudiantes que aprenden online conmigo.
* Ejemplos amenos y enfocados a que los conceptos teóricos te queden super claros y sin dudas intermedias
* Aprende desde cero acerca del mundo de la IA – ya que aprenderás desde un lienzo en blanco e iremos construyendo las ecuaciones necesarias acerca del mundo del aprendizaje por refuerzo y las redes neuronales paso a paso.
* Todo el código fuente  en Python – Si te quedas atascado solo tienes que ir a Github y descargar el material que no te funcione.
* Algoritmos que se usan en el mundo real -  No haremos un único algoritmo, si no diversos ejemplos con soluciones diversas y de dificultad y estructura de información creciente para que no solo memorices una receta de cocina, si no que adquieras todo lo necesario para poner en producción tus propios algoritmos.
