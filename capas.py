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


class OptimizadorSGD:
    """Descenso de gradiente puro: W -= learning_rate * dW. Es el optimizador por defecto de
    CapaDensa -- sin estado propio, así que no cambia el comportamiento de ningún proyecto que
    no pida uno distinto explícitamente."""

    def actualizar(self, W, b, dW, db, learning_rate):
        return W - learning_rate * dW, b - learning_rate * db


class OptimizadorAdam:
    """Adam (Kingma & Ba, 2015): en vez de dar el mismo paso en la dirección del gradiente
    (SGD), lleva una media móvil del gradiente (momento de primer orden, m -- suaviza la
    dirección, como inercia) y una media móvil del gradiente al cuadrado (momento de segundo
    orden, v -- reduce el paso donde el gradiente es grande/ruidoso, lo amplía donde es
    pequeño/estable). `m` y `v` se inicializan a cero, lo que sesga sus primeras estimaciones
    hacia cero -- la corrección de sesgo (dividir por 1 - beta^t) compensa ese arranque en frío.

    Cada instancia lleva el estado (m, v, t) de UNA sola capa -- no se comparte entre capas,
    cada CapaDensa necesita su propio OptimizadorAdam."""

    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.mW = self.vW = self.mb = self.vb = None
        self.t = 0

    def actualizar(self, W, b, dW, db, learning_rate):
        if self.mW is None:
            self.mW, self.vW = np.zeros_like(W), np.zeros_like(W)
            self.mb, self.vb = np.zeros_like(b), np.zeros_like(b)

        self.t += 1
        self.mW = self.beta1 * self.mW + (1 - self.beta1) * dW
        self.vW = self.beta2 * self.vW + (1 - self.beta2) * dW ** 2
        self.mb = self.beta1 * self.mb + (1 - self.beta1) * db
        self.vb = self.beta2 * self.vb + (1 - self.beta2) * db ** 2

        correccion1 = 1 - self.beta1 ** self.t
        correccion2 = 1 - self.beta2 ** self.t
        mW_hat, vW_hat = self.mW / correccion1, self.vW / correccion2
        mb_hat, vb_hat = self.mb / correccion1, self.vb / correccion2

        W_nuevo = W - learning_rate * mW_hat / (np.sqrt(vW_hat) + self.epsilon)
        b_nuevo = b - learning_rate * mb_hat / (np.sqrt(vb_hat) + self.epsilon)
        return W_nuevo, b_nuevo


def _normal(rng, shape):
    """randn de np.random o standard_normal de un Generator, según cuál se use -- así el
    resto del código no necesita saber si recibió un rng explícito (reproducible con
    seed_modelo) o ninguno (comportamiento legacy, estado global de np.random)."""
    return rng.standard_normal(shape) if rng is not None else np.random.randn(*shape)


def _uniform(rng, low, high, shape):
    return rng.uniform(low, high, shape) if rng is not None else np.random.uniform(low, high, shape)


class CapaDensa:
    """Capa totalmente conectada: Z = X @ W + b."""

    def __init__(self, dim_entrada, dim_salida, semilla_he=True, rng=None, optimizador=None):
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
        # optimizador=None (por defecto) usa SGD puro -- comportamiento idéntico al de siempre,
        # así que ningún proyecto existente se ve afectado si no pide uno distinto a propósito.
        self.optimizador = optimizador if optimizador is not None else OptimizadorSGD()

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

        self.W, self.b = self.optimizador.actualizar(self.W, self.b, dW, db, learning_rate)

        return dX_anterior


class CapaBatchNorm:
    """Normaliza cada característica a media 0 / varianza 1 sobre el batch actual, y aplica
    una escala y un desplazamiento aprendidos (gamma, beta) -- así la red conserva la libertad
    de deshacer la normalización si le conviene, en vez de forzar siempre media 0 en cada capa.

    En evaluación (entrenando=False) usa la media/varianza acumuladas durante el entrenamiento
    (`running_mean`/`running_var`, medias móviles con momento) en vez de las del batch actual:
    en test no siempre hay un batch grande -- puede ser una sola muestra -- del que sacar
    estadísticas fiables, así que se reutilizan las que ya se vieron entrenando.

    `gamma`/`beta` se actualizan con el mismo `optimizador` intercambiable que `CapaDensa`
    (SGD por defecto, Adam si se pasa uno) -- son parámetros aprendidos como cualquier otro.

    Derivación del backward: la forma vectorizada estándar (ver "Batch Normalization backward
    pass" de CS231n, Stanford), no la versión ingenua paso a paso por el grafo -- son
    algebraicamente equivalentes, pero esta evita varios tensores intermedios."""

    def __init__(self, dim, momentum=0.9, epsilon=1e-5, optimizador=None):
        self.gamma = np.ones((1, dim))
        self.beta = np.zeros((1, dim))
        self.momentum = momentum
        self.epsilon = epsilon
        self.running_mean = np.zeros((1, dim))
        self.running_var = np.ones((1, dim))
        self.optimizador = optimizador if optimizador is not None else OptimizadorSGD()
        self._cache = None

    def forward(self, X, entrenando=True):
        if entrenando:
            mu = X.mean(axis=0, keepdims=True)
            var = X.var(axis=0, keepdims=True)
            std_inv = 1.0 / np.sqrt(var + self.epsilon)
            X_norm = (X - mu) * std_inv
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
            self._cache = (X, X_norm, mu, std_inv)
        else:
            X_norm = (X - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
        return self.gamma * X_norm + self.beta

    def backward(self, dZ, learning_rate):
        X, X_norm, mu, std_inv = self._cache
        N = X.shape[0]

        dgamma = np.sum(dZ * X_norm, axis=0, keepdims=True)
        dbeta = np.sum(dZ, axis=0, keepdims=True)

        dX_norm = dZ * self.gamma
        X_mu = X - mu
        dvar = np.sum(dX_norm * X_mu, axis=0, keepdims=True) * -0.5 * std_inv ** 3
        dmu = (np.sum(dX_norm * -std_inv, axis=0, keepdims=True)
               + dvar * np.mean(-2 * X_mu, axis=0, keepdims=True))
        dX_anterior = dX_norm * std_inv + dvar * 2 * X_mu / N + dmu / N

        self.gamma, self.beta = self.optimizador.actualizar(self.gamma, self.beta, dgamma, dbeta, learning_rate)

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
