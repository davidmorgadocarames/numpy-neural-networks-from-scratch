# Autograd desde cero

En el resto de este repositorio, cada capa de `capas.py`/`capas_cnn.py` deriva su propio
`backward()` a mano: alguien se sentó con papel, aplicó la regla de la cadena a esa operación
concreta, y escribió el resultado en NumPy. Es explícito y verificable, pero no escala -- cada
operación nueva exige derivar y escribir un backward nuevo.

Aquí, en cambio, [`autograd.py`](autograd.py) construye un **grafo de operaciones sobre la
marcha**: cada vez que se suma, multiplica o se aplica una función a un `Tensor`, el grafo se
amplía solo, guardando de qué operación salió y cómo repartir el gradiente hacia sus entradas.
`Tensor.backward()` recorre ese grafo en orden topológico inverso aplicando la regla de la
cadena de forma **genérica**, una sola vez -- no una por tipo de capa. Es, en miniatura, lo que
hacen PyTorch/TensorFlow por debajo.

No sustituye a `capas.py` en ningún proyecto del repo -- entrenar 07/08 con esto sería más
lento sin aportar ningún resultado nuevo (el gradiente sería idéntico al ya verificado a mano).
Es una pieza aparte, con su propia verificación.

## Qué implementa

`Tensor` soporta las operaciones necesarias para una red densa pequeña: `+`, `-`, `*`,
`@` (matmul), `**`, `.mean()`, `.sum()`, `.leaky_relu()`, `.sigmoid()`. Cada una registra su
propia regla de la cadena local (una función que recibe el gradiente de la salida y devuelve
el de cada entrada) -- ver `_padres` en cada método de `autograd.py`.

## Verificación — [`red_juguete.py`](red_juguete.py)

Una red 2 → 4 → 1 (LeakyReLU + Sigmoid), la misma forma que el proyecto 01 del repo (compuerta
XOR), entrenada con este motor en vez de con un `backward()` escrito a mano. Dos
comprobaciones, no una sola afirmación:

1. **Los gradientes son correctos**: se comparan contra diferencias finitas, el mismo criterio
   exacto que ya usa [`../tests/test_gradients.py`](../tests/test_gradients.py) para el
   backward escrito a mano. Los cuatro parámetros (`W1`, `b1`, `W2`, `b2`) coinciden hasta
   ~1e-12 -- ruido de precisión de punto flotante, no discrepancia real.
2. **La red aprende de verdad**: entrenada con el gradiente de este motor, resuelve XOR con
   4/4 aciertos (loss final 0.000525, partiendo de 0.2577).

```
=== Verificación de gradientes (autograd vs. diferencias finitas) ===
  W1: autograd vs. numérico -- OK (máxima diferencia: 2.56e-12)
  b1: autograd vs. numérico -- OK (máxima diferencia: 4.18e-12)
  W2: autograd vs. numérico -- OK (máxima diferencia: 2.30e-12)
  b2: autograd vs. numérico -- OK (máxima diferencia: 2.57e-12)

=== Entrenando XOR con autograd ===
Loss inicial: 0.2577  ->  Loss final: 0.000525
Aciertos: 4/4
```

### Un hallazgo honesto por el camino: el codo de LeakyReLU

La primera versión de esta verificación inicializaba **todos** los sesgos a cero (`np.zeros`),
práctica estándar. `b1` fallaba la comparación con diferencias finitas -- una diferencia
máxima de ~2e-2, pequeña pero no ruido de precisión. La causa: XOR incluye la fila de entrada
`[0, 0]`, y con `b1` exactamente en cero, la preactivación de *toda* la capa oculta para esa
fila es `X@W1 + b1 = 0 + 0 = 0` -- exactamente el punto no derivable de LeakyReLU (el "codo" en
Z=0, donde la pendiente salta de 0.01 a 1). Ahí la diferencia finita centrada cruza el codo (mide
la pendiente de un lado y del otro, y promedia), mientras que el backward analítico usa un
convenio fijo (`Z > 0 → pendiente 1`, si no → `0.01`) -- los dos son "correctos" a su manera,
pero no tienen por qué coincidir en ese único punto exacto.

No es un bug del motor: verificado inicializando `b1` con ruido pequeño en vez de cero, la
diferencia baja de ~2e-2 a ~4e-12 (ver `verificar_gradientes()` en `red_juguete.py`). La
verificación usa sesgos con ruido pequeño por este motivo; `entrenar_xor()` sí usa sesgos a
cero (práctica estándar, no afecta al entrenamiento real -- solo a la precisión exacta de
comparar contra diferencias finitas en ese punto concreto).

## Reproducir

```bash
pip install -r ../requirements.txt
python red_juguete.py
```

## Limitaciones

- No implementa softmax/entropía cruzada (la red de juguete usa sigmoid + error cuadrático
  medio para mantener el conjunto de operaciones mínimo) ni capas convolucionales.
- Sin optimización de grafo ni ningún tipo de paralelismo -- cada operación es un objeto
  Python con overhead propio; para una red mayor que esta sería notablemente más lento que el
  `backward()` a mano de `capas.py`, que es precisamente por lo que no se usa en 07/08.
