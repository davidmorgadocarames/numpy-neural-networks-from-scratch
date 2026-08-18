"""
Un motor de diferenciación automática en modo inverso, minúsculo, desde cero -- la pieza que
falta en el resto del repo. En `../capas.py`, cada capa deriva su propio `backward()` a mano
(la regla de la cadena, aplicada por una persona sobre el papel, una vez por tipo de capa).
Aquí, en cambio, se construye un grafo de operaciones sobre la marcha (cada vez que se suma,
multiplica o aplica una función a un `Tensor`, el grafo se amplía solo) y `backward()` recorre
ese grafo hacia atrás aplicando la regla de la cadena de forma GENÉRICA -- una sola vez, no una
por capa. Es literalmente lo que hacen PyTorch/TensorFlow por debajo (a una escala
industrial); esto es la versión mínima que cabe entender de una sentada.

Cada `Tensor` guarda:
  - `valor`: el array de NumPy real.
  - `grad`: el gradiente acumulado (se rellena al llamar a `backward()`).
  - `_padres`: de qué Tensors salió esta operación, junto con la función que sabe repartir el
    gradiente hacia cada uno (la regla de la cadena de ESA operación concreta).

No sustituye a `capas.py` en ningún proyecto del repo -- ver `red_juguete.py` para una red
pequeña entrenada con este motor y verificada contra diferencias finitas, el mismo criterio
que ya usa `../tests/test_gradients.py` para el backward escrito a mano.
"""

import numpy as np


def _sumar_gradiente_con_broadcast(grad, forma_objetivo):
    """Si una operación hizo broadcast (p. ej. sumar un bias (1, D) a una matriz (N, D)), el
    gradiente que le corresponde al operando pequeño es la suma del gradiente grande sobre los
    ejes que se expandieron -- sin este ajuste, `grad` tendría una forma que ya no encaja con
    el operando original."""
    while grad.ndim > len(forma_objetivo):
        grad = grad.sum(axis=0)
    for eje, tam in enumerate(forma_objetivo):
        if tam == 1 and grad.shape[eje] != 1:
            grad = grad.sum(axis=eje, keepdims=True)
    return grad


class Tensor:
    def __init__(self, valor, _padres=(), _nombre="const"):
        self.valor = np.asarray(valor, dtype=float)
        self.grad = np.zeros_like(self.valor)
        self._padres = _padres  # tupla de (Tensor_padre, funcion_backward_local)
        self._nombre = _nombre

    def __repr__(self):
        return f"Tensor({self.valor}, nombre={self._nombre!r})"

    # === Operaciones -- cada una registra su propia regla de la cadena local ===

    def __add__(self, otro):
        otro = otro if isinstance(otro, Tensor) else Tensor(otro)
        salida = Tensor(self.valor + otro.valor, _nombre="add")
        salida._padres = (
            (self, lambda dS: _sumar_gradiente_con_broadcast(dS, self.valor.shape)),
            (otro, lambda dS: _sumar_gradiente_con_broadcast(dS, otro.valor.shape)),
        )
        return salida

    def __sub__(self, otro):
        otro = otro if isinstance(otro, Tensor) else Tensor(otro)
        salida = Tensor(self.valor - otro.valor, _nombre="sub")
        salida._padres = (
            (self, lambda dS: _sumar_gradiente_con_broadcast(dS, self.valor.shape)),
            (otro, lambda dS: _sumar_gradiente_con_broadcast(-dS, otro.valor.shape)),
        )
        return salida

    def __mul__(self, otro):
        otro = otro if isinstance(otro, Tensor) else Tensor(otro)
        salida = Tensor(self.valor * otro.valor, _nombre="mul")
        salida._padres = (
            (self, lambda dS: _sumar_gradiente_con_broadcast(dS * otro.valor, self.valor.shape)),
            (otro, lambda dS: _sumar_gradiente_con_broadcast(dS * self.valor, otro.valor.shape)),
        )
        return salida

    def __matmul__(self, otro):
        salida = Tensor(self.valor @ otro.valor, _nombre="matmul")
        salida._padres = (
            (self, lambda dS: dS @ otro.valor.T),
            (otro, lambda dS: self.valor.T @ dS),
        )
        return salida

    def __pow__(self, potencia):
        salida = Tensor(self.valor ** potencia, _nombre=f"pow{potencia}")
        salida._padres = ((self, lambda dS: dS * potencia * self.valor ** (potencia - 1)),)
        return salida

    def mean(self):
        n = self.valor.size
        salida = Tensor(self.valor.mean(), _nombre="mean")
        salida._padres = ((self, lambda dS: dS * np.ones_like(self.valor) / n),)
        return salida

    def sum(self, axis=None, keepdims=False):
        salida = Tensor(self.valor.sum(axis=axis, keepdims=keepdims), _nombre="sum")

        def _hacia_atras(dS):
            if axis is not None and not keepdims:
                dS = np.expand_dims(dS, axis=axis)
            return np.ones_like(self.valor) * dS

        salida._padres = ((self, _hacia_atras),)
        return salida

    def leaky_relu(self, pendiente=0.01):
        salida = Tensor(np.where(self.valor > 0, self.valor, pendiente * self.valor), _nombre="leaky_relu")
        salida._padres = ((self, lambda dS: dS * np.where(self.valor > 0, 1.0, pendiente)),)
        return salida

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-self.valor))
        salida = Tensor(s, _nombre="sigmoid")
        salida._padres = ((self, lambda dS: dS * s * (1 - s)),)
        return salida

    # === El motor: orden topológico + regla de la cadena, una sola vez ===

    def backward(self):
        """Rellena `.grad` de este Tensor y de todos sus antecesores en el grafo. Solo tiene
        sentido llamarlo sobre un escalar (una pérdida) -- por eso arranca con grad=1: dL/dL=1
        por definición, todo lo demás sale de propagar eso hacia atrás por la regla de la
        cadena en cada operación registrada."""
        orden = []
        visitados = set()

        def _topologico(nodo):
            if id(nodo) in visitados:
                return
            visitados.add(id(nodo))
            for padre, _ in nodo._padres:
                _topologico(padre)
            orden.append(nodo)

        _topologico(self)

        self.grad = np.ones_like(self.valor)
        for nodo in reversed(orden):
            for padre, funcion_backward_local in nodo._padres:
                padre.grad = padre.grad + funcion_backward_local(nodo.grad)
