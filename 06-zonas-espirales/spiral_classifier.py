"""
Clasificador de 3 brazos de una espiral (NumPy puro) -- el benchmark clásico para demostrar
que una red con capas ocultas puede aprender fronteras de decisión curvas que ningún modelo
lineal podría representar. Red modular profunda (Densa -> LeakyReLU -> Densa -> LeakyReLU ->
Densa -> Softmax), 2 capas ocultas de 64 neuronas.

Se separa un 20% como test para medir generalización real con una matriz de confusión sobre
puntos nunca vistos en el entrenamiento.

Uso: python spiral_classifier.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from capas import ActivacionLeakyReLU, ActivacionSoftmax, CapaDensa, predecir

SEED = 0
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
N_POR_BRAZO = 150
K_CLASES = 3


def main() -> None:
    np.random.seed(SEED)

    X = np.zeros((N_POR_BRAZO * K_CLASES, 2))
    Y_num = np.zeros(N_POR_BRAZO * K_CLASES, dtype="uint8")
    for i in range(K_CLASES):
        r = np.linspace(0.0, 1, N_POR_BRAZO)
        t = np.linspace(i * 4, (i + 1) * 4, N_POR_BRAZO) + np.random.randn(N_POR_BRAZO) * 0.2
        X[i * N_POR_BRAZO : (i + 1) * N_POR_BRAZO] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y_num[i * N_POR_BRAZO : (i + 1) * N_POR_BRAZO] = i

    Y = np.zeros((N_POR_BRAZO * K_CLASES, K_CLASES))
    Y[np.arange(N_POR_BRAZO * K_CLASES), Y_num] = 1

    # Split train/test estratificado por brazo (80% train, 20% test)
    rng = np.random.default_rng(SEED)
    indices_train, indices_test = [], []
    for clase in range(K_CLASES):
        idx_clase = np.where(Y_num == clase)[0]
        rng.shuffle(idx_clase)
        corte = int(0.8 * len(idx_clase))
        indices_train.extend(idx_clase[:corte])
        indices_test.extend(idx_clase[corte:])
    indices_train, indices_test = np.array(indices_train), np.array(indices_test)

    X_train, X_test = X[indices_train], X[indices_test]
    Y_train, Y_test = Y[indices_train], Y[indices_test]
    Y_num_test = Y_num[indices_test]

    red = [
        CapaDensa(dim_entrada=2, dim_salida=64),
        ActivacionLeakyReLU(),
        CapaDensa(dim_entrada=64, dim_salida=64),
        ActivacionLeakyReLU(),
        CapaDensa(dim_entrada=64, dim_salida=3),
        ActivacionSoftmax(),
    ]

    learning_rate = 0.2
    epochs = 5000
    historial_loss_train, historial_loss_test = [], []

    for epoch in range(epochs):
        activacion = X_train
        for capa in red:
            activacion = capa.forward(activacion)
        A_train = activacion

        loss_train = -np.mean(np.sum(Y_train * np.log(A_train + 1e-15), axis=1))
        historial_loss_train.append(loss_train)

        A_test = predecir(red, X_test)
        loss_test = -np.mean(np.sum(Y_test * np.log(A_test + 1e-15), axis=1))
        historial_loss_test.append(loss_test)

        gradiente = (A_train - Y_train) / Y_train.shape[0]
        for capa in reversed(red[:-1]):
            gradiente = capa.backward(gradiente, learning_rate)

    A_test = predecir(red, X_test)
    pred_test = np.argmax(A_test, axis=1)
    accuracy_test = float(np.mean(pred_test == Y_num_test))

    matriz_confusion = np.zeros((3, 3), dtype=int)
    for real, pred in zip(Y_num_test, pred_test):
        matriz_confusion[real, pred] += 1

    metrics = {
        "epochs": epochs,
        "loss_train_final": float(historial_loss_train[-1]),
        "loss_test_final": float(historial_loss_test[-1]),
        "accuracy_test": accuracy_test,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "matriz_confusion": matriz_confusion.tolist(),
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Accuracy test: {accuracy_test:.4f} ({len(X_test)} puntos de test)")

    # === Gráfico 1: curva de aprendizaje ===
    plt.figure(figsize=(6, 4))
    plt.plot(historial_loss_train, color="purple", label="Train")
    plt.plot(historial_loss_test, color="orange", linestyle="--", label="Test")
    plt.title("Curva de aprendizaje (reto espiral)", fontweight="bold")
    plt.xlabel("Épocas")
    plt.ylabel("Pérdida")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "learning_curve.png", dpi=150)
    plt.close()

    # === Gráfico 2: visualización de datos = zonas de espiral aprendidas ===
    step = 0.02
    xx, yy = np.meshgrid(np.arange(-1.2, 1.2, step), np.arange(-1.2, 1.2, step))
    rejilla = np.c_[xx.ravel(), yy.ravel()]
    frontera = np.argmax(predecir(red, rejilla), axis=1).reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, frontera, alpha=0.5, cmap="jet")
    plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_num[indices_train], s=30, cmap="jet",
                edgecolors="k", label="Train")
    plt.scatter(X_test[:, 0], X_test[:, 1], c=Y_num_test, s=80, cmap="jet", marker="*",
                edgecolors="white", linewidths=1.2, label="Test")
    plt.title("Zonas de espirales aprendidas por la red", fontweight="bold")
    plt.xlabel("Eje X")
    plt.ylabel("Eje Y")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "data_visualization.png", dpi=150)
    plt.close()

    # === Gráfico 3: matriz de confusión (test) ===
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(matriz_confusion, cmap="Blues")
    ax.set_title(f"Matriz de confusión test (accuracy={accuracy_test:.2%})", fontweight="bold")
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(["Brazo 0", "Brazo 1", "Brazo 2"])
    ax.set_yticklabels(["Brazo 0", "Brazo 1", "Brazo 2"])
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(matriz_confusion[i, j]), ha="center", va="center",
                     color="white" if matriz_confusion[i, j] > 5 else "black", fontsize=12)
    fig.colorbar(im, fraction=0.046)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    print(f"Resultados guardados en {RESULTS_DIR}")


if __name__ == "__main__":
    main()
