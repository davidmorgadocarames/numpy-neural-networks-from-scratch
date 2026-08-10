"""
Verificación numérica (por diferencias finitas) de los backward() escritos a mano en
capas.py y capas_cnn.py. Es la comprobación estándar para una capa de red neuronal: si el
gradiente analítico (el que calcula backward()) coincide con la pendiente medida
perturbando cada entrada/parámetro en +-eps y viendo cuánto cambia la salida, el backward
está bien derivado. Reemplaza la referencia informal a "verificado antes de entrenar" del
README del proyecto 08 por algo que cualquiera puede ejecutar y comprobar por sí mismo:

    pytest RRNN/tests/test_gradients.py -v
"""

import numpy as np

from capas import ActivacionLeakyReLU, ActivacionSigmoide, CapaDensa
from capas_cnn import CapaConv2D, CapaMaxPool2D

EPS = 1e-5
ATOL = 1e-4


def gradiente_numerico_entrada(capa, X, dZ_fija, eps=EPS):
    """dL/dX por diferencias finitas centradas, con L(X) = sum(forward(X) * dZ_fija) --
    dZ_fija hace de "gradiente que llega desde la capa siguiente" en un backward real."""
    grad = np.zeros_like(X, dtype=float)
    for idx in np.ndindex(X.shape):
        original = X[idx]
        X[idx] = original + eps
        mas = np.sum(capa.forward(X, entrenando=False) * dZ_fija)
        X[idx] = original - eps
        menos = np.sum(capa.forward(X, entrenando=False) * dZ_fija)
        X[idx] = original
        grad[idx] = (mas - menos) / (2 * eps)
    return grad


def gradiente_numerico_parametro(capa, nombre_atributo, X, dZ_fija, eps=EPS):
    """Igual que arriba pero perturbando un parámetro de la capa (W o b) en vez de X."""
    parametro = getattr(capa, nombre_atributo)
    grad = np.zeros_like(parametro, dtype=float)
    for idx in np.ndindex(parametro.shape):
        original = parametro[idx]
        parametro[idx] = original + eps
        mas = np.sum(capa.forward(X, entrenando=False) * dZ_fija)
        parametro[idx] = original - eps
        menos = np.sum(capa.forward(X, entrenando=False) * dZ_fija)
        parametro[idx] = original
        grad[idx] = (mas - menos) / (2 * eps)
    return grad


def gradiente_analitico_parametro(capa, nombre_atributo, X, dZ_fija):
    """El backward() real no expone dW/db directamente (los usa para actualizar el
    parámetro in-place), así que se extraen del propio efecto de la actualización: con
    learning_rate=1, backward hace `parametro -= 1 * grad`, luego grad = antes - después."""
    antes = getattr(capa, nombre_atributo).copy()
    capa.forward(X, entrenando=True)
    capa.backward(dZ_fija, learning_rate=1.0)
    despues = getattr(capa, nombre_atributo)
    grad = antes - despues
    setattr(capa, nombre_atributo, antes)  # deja la capa como estaba para el resto del test
    return grad


def test_capa_densa_gradientes():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4, 5))
    capa = CapaDensa(5, 3)
    dZ_fija = rng.normal(size=(4, 3))

    dX_num = gradiente_numerico_entrada(capa, X.copy(), dZ_fija)
    dW_num = gradiente_numerico_parametro(capa, "W", X, dZ_fija)
    db_num = gradiente_numerico_parametro(capa, "b", X, dZ_fija)

    capa.forward(X, entrenando=True)
    dX_analitico = capa.backward(dZ_fija, learning_rate=0.0)  # lr=0: no muta W/b
    dW_analitico = gradiente_analitico_parametro(capa, "W", X, dZ_fija)
    db_analitico = gradiente_analitico_parametro(capa, "b", X, dZ_fija)

    assert np.allclose(dX_num, dX_analitico, atol=ATOL)
    assert np.allclose(dW_num, dW_analitico, atol=ATOL)
    assert np.allclose(db_num, db_analitico, atol=ATOL)


def test_activacion_leaky_relu_gradiente():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(4, 6))
    capa = ActivacionLeakyReLU()
    dZ_fija = rng.normal(size=(4, 6))

    dX_num = gradiente_numerico_entrada(capa, X.copy(), dZ_fija)
    capa.forward(X, entrenando=True)
    dX_analitico = capa.backward(dZ_fija, learning_rate=0.0)

    assert np.allclose(dX_num, dX_analitico, atol=ATOL)


def test_activacion_sigmoide_gradiente():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(4, 6))
    capa = ActivacionSigmoide()
    dZ_fija = rng.normal(size=(4, 6))

    dX_num = gradiente_numerico_entrada(capa, X.copy(), dZ_fija)
    capa.forward(X, entrenando=True)
    dX_analitico = capa.backward(dZ_fija, learning_rate=0.0)

    assert np.allclose(dX_num, dX_analitico, atol=ATOL)


def test_capa_conv2d_gradientes():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(1, 5, 5, 2))
    capa = CapaConv2D(kh=2, kw=2, canales_entrada=2, canales_salida=2)
    salida_forma = capa.forward(X, entrenando=False).shape
    dZ_fija = rng.normal(size=salida_forma)

    dX_num = gradiente_numerico_entrada(capa, X.copy(), dZ_fija)
    dW_num = gradiente_numerico_parametro(capa, "W", X, dZ_fija)
    db_num = gradiente_numerico_parametro(capa, "b", X, dZ_fija)

    capa.forward(X, entrenando=True)
    dX_analitico = capa.backward(dZ_fija, learning_rate=0.0)
    dW_analitico = gradiente_analitico_parametro(capa, "W", X, dZ_fija)
    db_analitico = gradiente_analitico_parametro(capa, "b", X, dZ_fija)

    assert np.allclose(dX_num, dX_analitico, atol=ATOL)
    assert np.allclose(dW_num, dW_analitico, atol=ATOL)
    assert np.allclose(db_num, db_analitico, atol=ATOL)


def test_capa_maxpool2d_gradiente():
    rng = np.random.default_rng(4)
    # Valores separados a propósito (no una gaussiana estrecha) para que ningún empate de
    # máximo caiga dentro del rango del eps de la diferencia finita.
    X = rng.normal(scale=10.0, size=(1, 4, 4, 2))
    capa = CapaMaxPool2D(size=2, stride=2)
    salida_forma = capa.forward(X, entrenando=False).shape
    dZ_fija = rng.normal(size=salida_forma)

    dX_num = gradiente_numerico_entrada(capa, X.copy(), dZ_fija)
    capa.forward(X, entrenando=True)
    dX_analitico = capa.backward(dZ_fija, learning_rate=0.0)

    assert np.allclose(dX_num, dX_analitico, atol=ATOL)
