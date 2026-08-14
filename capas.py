"""
Mini-framework de red neuronal escrito desde cero en NumPy: capas densas y activaciones con
su propio forward/backward, sin TensorFlow/PyTorch/Keras de por medio. Los 7 proyectos de
capas densas de este repo importan este módulo en vez de reescribir las clases cada vez.

Cada capa expone dos métodos:
  - forward(X, entrenando=True):  calcula la salida y, solo si entrenando=True, GUARDA lo
    necesario para el backward (entrada, Z, etc.). Con entrenando=False no toca ese estado —
    así se puede evaluar test a mitad de una época de entrenamiento (ver predecir() más abajo)
    sin corromper lo que el backward() del batch de train de esa misma época todavía necesita.
  - backward(dZ_o_dA, learning_rate): usa lo guardado para calcular gradientes, actualizar sus
    propios parámetros (si los tiene) y devolver el gradiente que le corresponde a la capa
    anterior — así se encadenan sin que el bucle de entrenamiento necesite saber qué hay dentro
    de cada capa (backpropagation genérica: "for capa in reversed(red): grad = capa.backward(...)").
"""

import numpy as np


def _normal(rng, shape):
    """randn de np.random o standard_normal de un Generator, según cuál se use -- así el
    resto del código no necesita saber si recibió un rng explícito (reproducible con
    seed_modelo) o ninguno (comportamiento legacy, estado global de np.random)."""
    return rng.standard_normal(shape) if rng is not None else np.random.randn(*shape)


def _uniform(rng, low, high, shape):
    return rng.uniform(low, high, shape) if rng is not None else np.random.uniform(low, high, shape)


class CapaDensa:
    """Capa totalmente conectada: Z = X @ W + b."""

    def __init__(self, dim_entrada, dim_salida, semilla_he=True, rng=None):
        # Inicialización He (recomendada con LeakyReLU/ReLU): evita que las activaciones
        # exploten o se desvanezcan según crece el número de capas/neuronas.
        # rng=None (por defecto) mantiene el comportamiento de siempre (estado global de
        # np.random); pasar un np.random.Generator permite controlar la inicialización de
        # pesos con una semilla propia (seed_modelo), independiente de la de los datos.
        if semilla_he:
            self.W = _normal(rng, (dim_entrada, dim_salida)) * np.sqrt(2.0 / dim_entrada)
        else:
            self.W = _uniform(rng, -0.5, 0.5, (dim_entrada, dim_salida))
        self.b = np.zeros((1, dim_salida))
        self.X_entrada = None
        self.Z = None

    def forward(self, X, entrenando=True):
        Z = np.dot(X, self.W) + self.b
        if entrenando:
            self.X_entrada = X
            self.Z = Z
        return Z

    def backward(self, dZ, learning_rate):
        dW = np.dot(self.X_entrada.T, dZ)
        db = np.sum(dZ, axis=0, keepdims=True)
        dX_anterior = np.dot(dZ, self.W.T)

        self.W -= learning_rate * dW
        self.b -= learning_rate * db

        return dX_anterior


class ActivacionLeakyReLU:
    def __init__(self):
        self.Z = None

    def forward(self, Z, entrenando=True):
        if entrenando:
            self.Z = Z
        return np.where(Z > 0, Z, 0.01 * Z)

    def backward(self, dA, learning_rate):
        derivada = np.where(self.Z > 0, 1, 0.01)
        return dA * derivada


class ActivacionSigmoide:
    def __init__(self):
        self.A = None

    def forward(self, Z, entrenando=True):
        A = 1 / (1 + np.exp(-Z))
        if entrenando:
            self.A = A
        return A

    def backward(self, dA, learning_rate):
        derivada = self.A * (1 - self.A)
        return dA * derivada


class ActivacionSoftmax:
    """Sin backward propio: combinada con entropía cruzada, el gradiente de la capa de
    salida se simplifica a (prediccion - objetivo) / N y se calcula directamente en el
    bucle de entrenamiento, antes de propagar hacia las capas anteriores (por eso todos los
    bucles de este repo excluyen la última capa al recorrer la red al revés)."""

    def forward(self, Z, entrenando=True):
        exp_z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def predecir(red, X):
    """Forward pass para evaluar (validación/test) reutilizando el forward() real de cada capa
    con entrenando=False, en vez de reimplementar la fórmula de cada tipo de capa por separado
    — así no hay dos copias de la misma matemática que puedan divergir en silencio si se cambia
    una activación y se olvida actualizar la otra. Necesario (en vez de llamar a forward() sin
    más) para poder evaluar en test A MITAD del bucle de entrenamiento sin corromper lo que hace
    falta para el backward() del batch de train de esa misma época."""
    activacion = X
    for capa in red:
        activacion = capa.forward(activacion, entrenando=False)
    return activacion


def matriz_pesos(red, indices_capas_densas):
    """Extrae las matrices W de las CapaDensa indicadas (por índice en la lista `red`), para
    la visualización 'Cerebro de la IA' — inspección directa de los pesos que la red aprendió."""
    return [red[i].W for i in indices_capas_densas]
