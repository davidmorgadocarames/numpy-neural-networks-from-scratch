"""
Una red pequeña (2 → 4 → 1, LeakyReLU + Sigmoid, la misma forma que el proyecto 01 del repo,
compuerta XOR) entrenada con `autograd.py` en vez de con un `backward()` escrito a mano.

Dos verificaciones, no una sola afirmación:
1. Los gradientes que calcula `Tensor.backward()` coinciden con el gradiente medido por
   diferencias finitas -- el mismo criterio exacto que ya usa
   `../tests/test_gradients.py` para el backward escrito a mano de `capas.py`.
2. La red entrena de verdad y resuelve XOR -- no basta con que el gradiente sea correcto en un
   punto suelto, tiene que servir para aprender algo.

Uso: python red_juguete.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from autograd import Tensor

EPS = 1e-5
ATOL = 1e-4

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
Y = np.array([[0], [1], [1], [0]], dtype=float)  # XOR


def forward(X_t, W1, b1, W2, b2):
    H = (X_t @ W1 + b1).leaky_relu()
    pred = (H @ W2 + b2).sigmoid()
    return pred


def perdida(X_t, Y_t, W1, b1, W2, b2):
    pred = forward(X_t, W1, b1, W2, b2)
    return ((pred - Y_t) ** 2).mean()


def gradiente_numerico(f, tensor_objetivo, eps=EPS):
    """Diferencias finitas centradas sobre CADA elemento de tensor_objetivo.valor -- mismo
    criterio que gradiente_numerico_parametro() en tests/test_gradients.py."""
    grad = np.zeros_like(tensor_objetivo.valor)
    for idx in np.ndindex(tensor_objetivo.valor.shape):
        original = tensor_objetivo.valor[idx]
        tensor_objetivo.valor[idx] = original + eps
        mas = f().valor
        tensor_objetivo.valor[idx] = original - eps
        menos = f().valor
        tensor_objetivo.valor[idx] = original
        grad[idx] = (mas - menos) / (2 * eps)
    return grad


def verificar_gradientes(seed=0):
    """Sesgos inicializados con ruido pequeño, NO a cero -- a propósito. Con b1 exactamente en
    cero, la fila de entrada [0,0] de XOR da una preactivación EXACTAMENTE 0 en toda la capa
    oculta (X@W1 + b1 = 0 + 0), justo el punto no derivable de LeakyReLU (el "codo" en Z=0).
    Ahí la diferencia finita centrada cruza el codo (mide la pendiente ~1 de un lado y ~0.01
    del otro y promedia), mientras que el analítico usa un convenio fijo (Z>0 → pendiente 1,
    si no → 0.01) -- los dos son "correctos" a su manera, pero no coinciden exactamente en ese
    único punto. No es un bug del motor: si b1 no es exactamente cero, el error baja de ~2e-2 a
    ~1e-12 (verificado). `entrenar_xor()` sí usa sesgos a cero (práctica estándar, no afecta al
    entrenamiento real) -- este matiz solo importa para la verificación por diferencias finitas."""
    rng = np.random.default_rng(seed)
    X_t = Tensor(X)
    Y_t = Tensor(Y)
    W1 = Tensor(rng.normal(size=(2, 4)) * 0.5)
    b1 = Tensor(rng.normal(size=(1, 4)) * 0.1)
    W2 = Tensor(rng.normal(size=(4, 1)) * 0.5)
    b2 = Tensor(rng.normal(size=(1, 1)) * 0.1)

    loss = perdida(X_t, Y_t, W1, b1, W2, b2)
    loss.backward()

    ok = True
    for nombre, tensor in [("W1", W1), ("b1", b1), ("W2", W2), ("b2", b2)]:
        grad_num = gradiente_numerico(lambda: perdida(X_t, Y_t, W1, b1, W2, b2), tensor)
        coincide = np.allclose(tensor.grad, grad_num, atol=ATOL)
        ok = ok and coincide
        print(f"  {nombre}: autograd vs. numérico -- {'OK' if coincide else 'FALLO'} "
              f"(máxima diferencia: {np.max(np.abs(tensor.grad - grad_num)):.2e})")
    return ok


def entrenar_xor(epochs=3000, learning_rate=0.5, seed=0):
    rng = np.random.default_rng(seed)
    X_t = Tensor(X)
    Y_t = Tensor(Y)
    W1 = Tensor(rng.normal(size=(2, 4)) * 0.5)
    b1 = Tensor(np.zeros((1, 4)))
    W2 = Tensor(rng.normal(size=(4, 1)) * 0.5)
    b2 = Tensor(np.zeros((1, 1)))
    parametros = [W1, b1, W2, b2]

    historial_loss = []
    for epoch in range(epochs):
        for p in parametros:
            p.grad = np.zeros_like(p.valor)  # backward() ACUMULA -- hay que vaciar antes de cada paso

        loss = perdida(X_t, Y_t, W1, b1, W2, b2)
        loss.backward()
        historial_loss.append(float(loss.valor))

        for p in parametros:
            p.valor = p.valor - learning_rate * p.grad

    pred_final = forward(X_t, W1, b1, W2, b2).valor
    return historial_loss, pred_final


def main():
    print("=== Verificación de gradientes (autograd vs. diferencias finitas) ===")
    ok = verificar_gradientes()
    print("Todos los gradientes coinciden." if ok else "Hay gradientes que NO coinciden -- revisar autograd.py.")

    print("\n=== Entrenando XOR con autograd ===")
    historial_loss, pred_final = entrenar_xor()
    print(f"Loss inicial: {historial_loss[0]:.4f}  ->  Loss final: {historial_loss[-1]:.6f}")
    print("Predicciones finales (objetivo: 0, 1, 1, 0):")
    for entrada, pred, objetivo in zip(X, pred_final, Y):
        print(f"  {entrada.astype(int)} -> {pred[0]:.4f} (objetivo {int(objetivo[0])})")

    aciertos = int(np.sum((pred_final > 0.5).astype(int) == Y))
    print(f"Aciertos: {aciertos}/4")

    assert ok, "Los gradientes de autograd no coinciden con los numéricos"
    assert aciertos == 4, "La red no resolvió XOR"
    print("\nVerificado: gradientes correctos Y la red aprende XOR de verdad.")


if __name__ == "__main__":
    main()
