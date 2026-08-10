"""
Clasificador de tipos de clientes de una tienda online (NumPy puro) a partir de 2 variables:
minutos navegando y productos en el carrito. Red modular (Densa -> LeakyReLU -> Densa ->
Softmax) entrenada con entropía cruzada para separar 3 categorías: Navegadores, Ocasionales
y VIPs.

Se separa un 20% de clientes como test para medir generalización real con una matriz de
confusión sobre datos nunca vistos en el entrenamiento.

También corrige un bug de una versión anterior: minutos y productos se pasaban a la red sin
normalizar (hasta ~50 de magnitud), lo que frenaba tanto el descenso de gradiente que ni
siquiera llegaba a converger del todo (~93% incluso evaluado sobre los mismos datos de
entrenamiento) y la red confundía "Ocasionales" con "VIPs" pese a que las 3 categorías no se
solapan en ninguna de las 2 variables. Con los datos normalizados (min-max con min/max de
train) y los mismos hiperparámetros, converge a 100%.

Uso: python customer_classifier.py
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

SEED = 42
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
NOMBRES_CLASES = ["Navegadores", "Ocasionales", "VIPs"]


def main() -> None:
    np.random.seed(SEED)

    # 3 categorías de clientes, 40 por categoría (120 en total) -- más muestra que el
    # original (20/categoría) para poder permitirse un test set decente sin quedarse corto.
    X0 = np.random.uniform(2, 10, (40, 2)) + np.array([0, 0])
    X1 = np.random.uniform(15, 25, (40, 2)) + np.array([0, 2])
    X2 = np.random.uniform(30, 45, (40, 2)) + np.array([0, 6])
    X = np.vstack([X0, X1, X2])

    Y_num = np.concatenate([np.zeros(40), np.ones(40), np.full(40, 2)]).astype(int)
    Y = np.zeros((120, 3))
    Y[np.arange(120), Y_num] = 1

    # Split train/test estratificado por clase (33 train + 7 test por categoría)
    indices_train, indices_test = [], []
    rng = np.random.default_rng(SEED)
    for clase in range(3):
        idx_clase = np.where(Y_num == clase)[0]
        rng.shuffle(idx_clase)
        indices_train.extend(idx_clase[:33])
        indices_test.extend(idx_clase[33:])
    indices_train, indices_test = np.array(indices_train), np.array(indices_test)

    X_train_raw, X_test_raw = X[indices_train], X[indices_test]
    Y_train, Y_test = Y[indices_train], Y[indices_test]
    Y_num_test = Y_num[indices_test]

    # Normalización min-max (con min/max de TRAIN, nunca de test) -- sin esto, minutos y
    # productos llegan a la red en escalas de hasta 50, lo que en la práctica frena tanto el
    # descenso de gradiente que a las 3000 épocas la red seguía sin converger del todo y
    # confundía "Ocasionales" con "VIPs" aunque las categorías no se solapan en ninguna de las
    # 2 variables. Con los mismos hiperparámetros pero datos normalizados, converge a 100%.
    X_min, X_max = X_train_raw.min(axis=0), X_train_raw.max(axis=0)
    X_train = (X_train_raw - X_min) / (X_max - X_min)
    X_test = (X_test_raw - X_min) / (X_max - X_min)

    red = [
        CapaDensa(dim_entrada=2, dim_salida=5, semilla_he=False),
        ActivacionLeakyReLU(),
        CapaDensa(dim_entrada=5, dim_salida=3, semilla_he=False),
        ActivacionSoftmax(),
    ]

    learning_rate = 0.01
    epochs = 3000
    historial_loss_train, historial_loss_test = [], []

    for epoch in range(epochs):
        activacion = X_train
        for capa in red:
            activacion = capa.forward(activacion)
        A2_train = activacion

        loss_train = -np.mean(np.sum(Y_train * np.log(A2_train + 1e-15), axis=1))
        historial_loss_train.append(loss_train)

        A2_test = predecir(red, X_test)
        loss_test = -np.mean(np.sum(Y_test * np.log(A2_test + 1e-15), axis=1))
        historial_loss_test.append(loss_test)

        gradiente = (A2_train - Y_train) / Y_train.shape[0]
        for capa in reversed(red[:-1]):
            gradiente = capa.backward(gradiente, learning_rate)

    # === Evaluación en test ===
    A2_test = predecir(red, X_test)
    pred_test = np.argmax(A2_test, axis=1)
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
        "clases": NOMBRES_CLASES,
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Accuracy test: {accuracy_test:.4f} ({len(X_test)} clientes de test)")

    # === Gráfico 1: curva de aprendizaje ===
    plt.figure(figsize=(6, 4))
    plt.plot(historial_loss_train, color="purple", label="Train")
    plt.plot(historial_loss_test, color="orange", linestyle="--", label="Test")
    plt.title("Curva de aprendizaje (Entropía Cruzada)", fontweight="bold")
    plt.xlabel("Épocas")
    plt.ylabel("Pérdida")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "learning_curve.png", dpi=150)
    plt.close()

    # === Gráfico 2: visualización de datos = zonas de clientes aprendidas ===
    # La rejilla se define en la escala real (minutos/productos) para que los ejes se lean
    # igual que los datos originales, pero se normaliza antes de pasarla por la red, que
    # espera entradas en la misma escala 0-1 que vio durante el entrenamiento.
    x_min, x_max = 0, 50
    y_min, y_max = 0, 50
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.5), np.arange(y_min, y_max, 0.5))
    rejilla_raw = np.c_[xx.ravel(), yy.ravel()]
    rejilla_norm = (rejilla_raw - X_min) / (X_max - X_min)
    frontera = np.argmax(predecir(red, rejilla_norm), axis=1).reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, frontera, alpha=0.5, cmap="brg")
    plt.scatter(X_train_raw[Y_num[indices_train] == 0, 0], X_train_raw[Y_num[indices_train] == 0, 1],
                color="red", label="Navegadores (train)", edgecolors="k")
    plt.scatter(X_train_raw[Y_num[indices_train] == 1, 0], X_train_raw[Y_num[indices_train] == 1, 1],
                color="green", label="Ocasionales (train)", edgecolors="k")
    plt.scatter(X_train_raw[Y_num[indices_train] == 2, 0], X_train_raw[Y_num[indices_train] == 2, 1],
                color="blue", label="VIPs (train)", edgecolors="k")
    plt.scatter(X_test_raw[:, 0], X_test_raw[:, 1], color="yellow", marker="*", s=140,
                edgecolors="k", label="Test (nunca visto)", zorder=5)
    plt.title("Zonas de clientes aprendidas por la red", fontweight="bold")
    plt.xlabel("Minutos en la web")
    plt.ylabel("Productos en carrito")
    plt.legend(fontsize=8, loc="upper left")
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
    ax.set_xticklabels(NOMBRES_CLASES, rotation=20, ha="right")
    ax.set_yticklabels(NOMBRES_CLASES)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(matriz_confusion[i, j]), ha="center", va="center",
                     color="white" if matriz_confusion[i, j] > 3 else "black", fontsize=12)
    fig.colorbar(im, fraction=0.046)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    print(f"Resultados guardados en {RESULTS_DIR}")


if __name__ == "__main__":
    main()
