"""
Compuerta XOR con una red neuronal (NumPy puro, sin frameworks) usando el mini-framework
modular de capas.py. XOR es el ejemplo clásico de un problema NO separable linealmente: una
sola neurona (regresión logística) no puede resolverlo, hace falta al menos una capa oculta.
Es el "hola mundo" de las redes neuronales y el punto de partida de este repo.

Uso: python xor_gate.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from capas import ActivacionLeakyReLU, ActivacionSigmoide, CapaDensa, predecir

SEED = 42
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def main() -> None:
    np.random.seed(SEED)

    # XOR solo tiene 4 combinaciones posibles de entrada -- son universo completo del
    # problema, no una muestra. No hay train/test split: el objetivo aquí no es medir
    # generalización a datos nuevos, sino demostrar que la red aprende una frontera de
    # decisión no lineal que una sola neurona no podría representar.
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([[0], [1], [1], [0]])

    red = [
        CapaDensa(dim_entrada=2, dim_salida=16, semilla_he=False),
        ActivacionLeakyReLU(),
        CapaDensa(dim_entrada=16, dim_salida=1, semilla_he=False),
        ActivacionSigmoide(),
    ]

    learning_rate = 0.3
    epochs = 1000
    historial_loss = []

    for epoch in range(epochs):
        activacion = X
        for capa in red:
            activacion = capa.forward(activacion)
        A_final = activacion

        loss = np.mean((A_final - Y) ** 2)
        historial_loss.append(loss)

        gradiente = 2 * (A_final - Y) / Y.shape[0]
        for capa in reversed(red):
            gradiente = capa.backward(gradiente, learning_rate)

    # === Evaluación: las 4 combinaciones son el único conjunto de evaluación posible ===
    predicciones = predecir(red, X)
    predicciones_binarias = (predicciones > 0.5).astype(int)
    aciertos = int(np.sum(predicciones_binarias == Y))

    # Matriz de confusión 2x2 manual (sin sklearn, para mantener el proyecto 100% NumPy)
    matriz_confusion = np.zeros((2, 2), dtype=int)
    for real, pred in zip(Y.flatten(), predicciones_binarias.flatten()):
        matriz_confusion[real, pred] += 1

    metrics = {
        "epochs": epochs,
        "loss_final": float(historial_loss[-1]),
        "aciertos": f"{aciertos}/4",
        "predicciones": [
            {"entrada": X[i].tolist(), "objetivo": int(Y[i][0]), "prediccion": float(predicciones[i][0])}
            for i in range(4)
        ],
        "matriz_confusion": matriz_confusion.tolist(),
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Loss final: {historial_loss[-1]:.6f} | Aciertos: {aciertos}/4")
    for p in metrics["predicciones"]:
        print(f"  {p['entrada']} -> objetivo {p['objetivo']}, predicho {p['prediccion']:.4f}")

    # === Gráfico 1: curva de aprendizaje ===
    plt.figure(figsize=(6, 4))
    plt.plot(historial_loss, color="darkgreen", linewidth=2)
    plt.title("Curva de aprendizaje (XOR)", fontweight="bold")
    plt.xlabel("Épocas")
    plt.ylabel("Error cuadrático medio")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "learning_curve.png", dpi=150)
    plt.close()

    # === Gráfico 2: visualización de datos = frontera de decisión aprendida ===
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    rejilla = np.c_[xx.ravel(), yy.ravel()]
    Z_rejilla = predecir(red, rejilla).reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    fondo = plt.contourf(xx, yy, Z_rejilla, alpha=0.8, cmap="RdYlBu_r")
    plt.colorbar(fondo, label="Salida de la red (0 a 1)")
    for i in range(len(X)):
        color = "blue" if Y[i][0] == 0 else "red"
        plt.scatter(X[i, 0], X[i, 1], color=color, edgecolors="k", s=120, zorder=3)
    plt.title("Frontera de decisión aprendida (XOR)", fontweight="bold")
    plt.xlabel("Entrada X1")
    plt.ylabel("Entrada X2")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "data_visualization.png", dpi=150)
    plt.close()

    # === Gráfico 3: matriz de confusión ===
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(matriz_confusion, cmap="Blues")
    ax.set_title(f"Matriz de confusión (aciertos: {aciertos}/4)", fontweight="bold")
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0", "1"])
    ax.set_yticklabels(["0", "1"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matriz_confusion[i, j]), ha="center", va="center",
                     color="white" if matriz_confusion[i, j] > 2 else "black", fontsize=14)
    fig.colorbar(im, fraction=0.046)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    print(f"Resultados guardados en {RESULTS_DIR}")


if __name__ == "__main__":
    main()
