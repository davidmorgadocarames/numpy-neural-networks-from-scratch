"""
Capas de convolución, para este proyecto -- ampliación de `../capas.py` (que sigue
importándose aquí para `CapaDensa`, `ActivacionLeakyReLU` y `ActivacionSoftmax`; no se
reescriben capas que ya funcionan). `CapaConv2D` y `CapaMaxPool2D` usan la técnica im2col:
en vez de recorrer la imagen con bucles Python (lentísimo en NumPy puro), la convolución se
convierte en una única multiplicación de matrices, que sí delega en las rutinas de álgebra
lineal optimizadas de NumPy. Es lo que hace viable entrenar una CNN entera en CPU, sin GPU y
sin ningún framework de deep learning, en un tiempo razonable.

Todas las capas siguen la misma interfaz que las de `capas.py`: `forward(X, entrenando=True)`
calcula la salida y, solo si entrenando=True, guarda lo necesario para el backward;
`backward(dZ, learning_rate)` devuelve el gradiente de la capa anterior y actualiza sus propios
parámetros si los tiene. `CapaDropout` usa ese mismo argumento para además cambiar de
comportamiento (en evaluación no apaga ninguna neurona), pero la interfaz es la misma en todas
las capas -- por eso `predecir_cnn()` puede tratarlas todas por igual sin casos especiales.
"""

import numpy as np


def im2col(X, kh, kw, stride=1):
    """Reorganiza cada ventana kh x kw de la imagen en una fila, para poder convolucionar con
    una sola multiplicación de matrices. Técnica estándar (usada internamente por
    TensorFlow/PyTorch en su día antes de tener kernels dedicados), aquí escrita a mano."""
    N, H, W, C = X.shape
    out_h = (H - kh) // stride + 1
    out_w = (W - kw) // stride + 1
    columnas = np.zeros((N, out_h, out_w, kh, kw, C), dtype=X.dtype)
    for y in range(kh):
        y_max = y + stride * out_h
        for x in range(kw):
            x_max = x + stride * out_w
            columnas[:, :, :, y, x, :] = X[:, y:y_max:stride, x:x_max:stride, :]
    return columnas.reshape(N * out_h * out_w, kh * kw * C), out_h, out_w


def col2im(dcolumnas, forma_X, kh, kw, stride, out_h, out_w):
    """Inverso de im2col para el backward: reparte el gradiente de cada ventana de vuelta a
    las posiciones de la imagen original de las que salió (sumando donde las ventanas se
    solapan, que es lo que le pasa a cualquier píxel que participó en más de una convolución)."""
    N, H, W, C = forma_X
    dX = np.zeros((N, H, W, C), dtype=dcolumnas.dtype)
    dcolumnas = dcolumnas.reshape(N, out_h, out_w, kh, kw, C)
    for y in range(kh):
        y_max = y + stride * out_h
        for x in range(kw):
            x_max = x + stride * out_w
            dX[:, y:y_max:stride, x:x_max:stride, :] += dcolumnas[:, :, :, y, x, :]
    return dX


class CapaConv2D:
    """Convolución 2D 'válida' (sin padding): Z = conv(X, W) + b, implementada como
    im2col + multiplicación de matrices. W tiene forma (kh, kw, canales_entrada,
    canales_salida)."""

    def __init__(self, kh, kw, canales_entrada, canales_salida, stride=1):
        self.kh, self.kw, self.stride = kh, kw, stride
        # Inicialización He, igual que CapaDensa en capas.py: evita que las activaciones
        # exploten o se desvanezcan según el número de entradas por neurona (aquí, kh*kw*C_in).
        self.W = np.random.randn(kh, kw, canales_entrada, canales_salida) * np.sqrt(
            2.0 / (kh * kw * canales_entrada)
        )
        self.b = np.zeros(canales_salida)
        self._cache = None

    def forward(self, X, entrenando=True):
        columnas, out_h, out_w = im2col(X, self.kh, self.kw, self.stride)
        W_col = self.W.reshape(-1, self.W.shape[-1])
        Z = (columnas @ W_col + self.b).reshape(X.shape[0], out_h, out_w, -1)
        if entrenando:
            self._cache = (X.shape, columnas, out_h, out_w)
        return Z

    def backward(self, dZ, learning_rate):
        forma_X, columnas, out_h, out_w = self._cache
        canales_salida = self.W.shape[-1]
        dZ_col = dZ.reshape(-1, canales_salida)

        dW = (columnas.T @ dZ_col).reshape(self.W.shape)
        db = dZ_col.sum(axis=0)

        W_col = self.W.reshape(-1, canales_salida)
        dcolumnas = dZ_col @ W_col.T
        dX = col2im(dcolumnas, forma_X, self.kh, self.kw, self.stride, out_h, out_w)

        self.W -= learning_rate * dW
        self.b -= learning_rate * db
        return dX


class CapaMaxPool2D:
    """Max pooling: reduce cada ventana size x size a su valor máximo. Implementado con la
    técnica de "reshape + max" (vectorizada, sin bucles por píxel): si las dimensiones no son
    múltiplo del tamaño de ventana, se recorta el sobrante (esas filas/columnas finales quedan
    fuera del pooling, igual que harían los frameworks de producción en modo 'valid')."""

    def __init__(self, size=2, stride=2):
        self.size, self.stride = size, stride
        self._cache = None

    def forward(self, X, entrenando=True):
        N, H, W, C = X.shape
        s = self.stride
        H_recortado, W_recortado = (H // s) * s, (W // s) * s
        X_recortado = X[:, :H_recortado, :W_recortado, :]
        out_h, out_w = H_recortado // s, W_recortado // s
        X_ventanas = X_recortado.reshape(N, out_h, s, out_w, s, C)
        salida = X_ventanas.max(axis=(2, 4))
        if entrenando:
            # Máscara de qué posición ganó el máximo en cada ventana -- es donde el backward
            # tiene que mandar el gradiente (las demás posiciones de la ventana no influyeron
            # en la salida, así que su gradiente es 0).
            mascara = X_ventanas == salida[:, :, None, :, None, :]
            self._cache = (X.shape, mascara, H_recortado, W_recortado)
        return salida

    def backward(self, dZ, learning_rate):
        forma_X, mascara, H_recortado, W_recortado = self._cache
        N, H, W, C = forma_X
        s = self.stride
        # Si dos posiciones empatan al máximo (raro, pero posible), se reparte el gradiente
        # entre ellas a partes iguales en vez de duplicarlo.
        n_ganadores = mascara.sum(axis=(2, 4), keepdims=True)
        dX_ventanas = mascara * (dZ[:, :, None, :, None, :] / n_ganadores)
        dX_recortado = dX_ventanas.reshape(N, H_recortado, W_recortado, C)
        dX = np.zeros(forma_X, dtype=dZ.dtype)
        dX[:, :H_recortado, :W_recortado, :] = dX_recortado
        return dX


class CapaFlatten:
    """Aplana (N, H, W, C) -> (N, H*W*C) para conectar la parte convolucional con las capas
    densas. Sin parámetros propios -- el backward solo deshace el reshape."""

    def __init__(self):
        self._forma_entrada = None

    def forward(self, X, entrenando=True):
        if entrenando:
            self._forma_entrada = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, dZ, learning_rate):
        return dZ.reshape(self._forma_entrada)


class CapaDropout:
    """Apaga aleatoriamente una fracción `tasa` de neuronas en cada pasada de entrenamiento
    (poniéndolas a 0) y reescala el resto por 1/(1-tasa) para que la magnitud total de la
    activación no cambie de media ('inverted dropout', el estándar en frameworks modernos).
    En evaluación (entrenando=False) no apaga nada -- por eso necesita ese argumento extra que
    el resto de capas no tiene."""

    def __init__(self, tasa=0.3):
        self.tasa = tasa
        self._mascara = None

    def forward(self, X, entrenando=True):
        if not entrenando:
            return X
        self._mascara = (np.random.rand(*X.shape) > self.tasa) / (1.0 - self.tasa)
        return X * self._mascara

    def backward(self, dZ, learning_rate):
        return dZ * self._mascara


def predecir_cnn(red, X):
    """Forward pass para evaluar test reutilizando el forward() real de cada capa con
    entrenando=False, en vez de reimplementar la fórmula de cada tipo de capa por separado
    (mismo motivo y misma solución que `predecir()` en ../capas.py: evitar dos copias de la
    misma matemática que puedan divergir en silencio, y no corromper a mitad de época la caché
    que el backward del batch de train todavía necesita). Como todas las capas -- incluida
    CapaDropout -- aceptan `entrenando` en su forward(), no hace falta ningún caso especial
    aquí: en evaluación, Dropout simplemente no apaga ninguna neurona."""
    activacion = X
    for capa in red:
        activacion = capa.forward(activacion, entrenando=False)
    return activacion


def augmentar_lote(X, rng, angulo_max=15.0, zoom_rango=(0.9, 1.1), desplazamiento_max=2.0,
                    prob_flip=0.5):
    """Data augmentation manual en NumPy puro -- el equivalente hecho a mano de capas como
    `RandomFlip`, `RandomRotation`, `RandomZoom` y `RandomTranslation` de Keras. Cada imagen
    del lote recibe un flip horizontal aleatorio y una rotación + zoom + desplazamiento
    aleatorios e independientes (una transformación distinta por imagen y por época, no una
    única transformación fija para todo el dataset).

    La rotación/zoom/desplazamiento se aplican como una única transformación afín por imagen,
    con mapeo inverso (para cada píxel de salida, de dónde viene en la imagen original) y
    muestreo por vecino más cercano -- más simple que la interpolación bilineal que usaría
    Keras por debajo, con un coste en calidad de imagen despreciable para el propósito aquí
    (que la red no pueda memorizar la posición/orientación exacta de cada ejemplo)."""
    N, H, W, C = X.shape
    Xf = X.copy()

    mascara_flip = rng.random(N) < prob_flip
    Xf[mascara_flip] = Xf[mascara_flip][:, :, ::-1, :]

    theta = np.deg2rad(rng.uniform(-angulo_max, angulo_max, N))
    escala = rng.uniform(zoom_rango[0], zoom_rango[1], N)
    dx = rng.uniform(-desplazamiento_max, desplazamiento_max, N)
    dy = rng.uniform(-desplazamiento_max, desplazamiento_max, N)

    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    yy = (yy - cy)[None, :, :]
    xx = (xx - cx)[None, :, :]

    # Mapeo inverso: para cada píxel (y, x) de la imagen transformada, qué coordenada (ys, xs)
    # de la imagen ORIGINAL hay que leer -- por eso se usa la rotación/escala inversas.
    cos_t = np.cos(-theta)[:, None, None]
    sin_t = np.sin(-theta)[:, None, None]
    escala_inv = (1.0 / escala)[:, None, None]

    xs = escala_inv * (cos_t * xx - sin_t * yy) - dx[:, None, None] + cx
    ys = escala_inv * (sin_t * xx + cos_t * yy) - dy[:, None, None] + cy
    xs = np.clip(np.round(xs).astype(int), 0, W - 1)
    ys = np.clip(np.round(ys).astype(int), 0, H - 1)

    n_idx = np.arange(N)[:, None, None]
    return Xf[n_idx, ys, xs, :]
