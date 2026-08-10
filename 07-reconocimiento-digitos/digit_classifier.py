"""
Reconocimiento de dígitos manuscritos (MNIST) con una red neuronal escrita 100% en NumPy
(sin TensorFlow/Keras) -- el proyecto más avanzado del conjunto de clasificadores densos. Red
modular 784 -> 128 (LeakyReLU) -> 10 (Softmax), entrenada con entropía cruzada.

Se separa un conjunto de test para poder reportar una accuracy y una matriz de confusión
honestas sobre dígitos que la red nunca vio en el entrenamiento. Los pesos entrenados se
guardan en results/red_pesos.npz para que demo_gradio.py pueda cargarlos sin tener que
reentrenar.

Uso: python digit_classifier.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from capas import ActivacionLeakyReLU, ActivacionSoftmax, CapaDensa, predecir

SEED = 42
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
N_TRAIN = 1200
N_TEST = 300


def main() -> None:
    np.random.seed(SEED)

    print("Descargando MNIST (sklearn fetch_openml)...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
    X_puro, Y_puro = mnist.data, mnist.target.astype(int)

    # Muestra estratificada: mismo número de train/test por dígito para que la matriz de
    # confusión no esté sesgada por clases con más ejemplos que otras.
    rng = np.random.default_rng(SEED)
    indices_train, indices_test = [], []
    for digito in range(10):
        idx_digito = np.where(Y_puro == digito)[0]
        rng.shuffle(idx_digito)
        indices_train.extend(idx_digito[: N_TRAIN // 10])
        indices_test.extend(idx_digito[N_TRAIN // 10 : N_TRAIN // 10 + N_TEST // 10])
    indices_train, indices_test = np.array(indices_train), np.array(indices_test)
    rng.shuffle(indices_train)
    rng.shuffle(indices_test)

    X_train = X_puro[indices_train] / 255.0
    Y_train_num = Y_puro[indices_train]
    X_test = X_puro[indices_test] / 255.0
    Y_test_num = Y_puro[indices_test]

    Y_train = np.zeros((len(X_train), 10))
    Y_train[np.arange(len(X_train)), Y_train_num] = 1

    red = [
        CapaDensa(dim_entrada=784, dim_salida=128),
        ActivacionLeakyReLU(),
        CapaDensa(dim_entrada=128, dim_salida=10),
        ActivacionSoftmax(),
    ]

    learning_rate = 0.1
    epochs = 800
    historial_loss_train, historial_accuracy_test = [], []

    print(f"Entrenando con {len(X_train)} imágenes, evaluando sobre {len(X_test)} de test...")
    for epoch in range(epochs):
        activacion = X_train
        for capa in red:
            activacion = capa.forward(activacion)
        A_train = activacion

        loss_train = -np.mean(np.sum(Y_train * np.log(A_train + 1e-15), axis=1))
        historial_loss_train.append(loss_train)

        gradiente = (A_train - Y_train) / Y_train.shape[0]
        for capa in reversed(red[:-1]):
            gradiente = capa.backward(gradiente, learning_rate)

        if epoch % 20 == 0 or epoch == epochs - 1:
            pred_test = np.argmax(predecir(red, X_test), axis=1)
            acc_test = float(np.mean(pred_test == Y_test_num))
            historial_accuracy_test.append(acc_test)
            if epoch % 100 == 0:
                print(f"Época {epoch}/{epochs} - loss train: {loss_train:.4f} - accuracy test: {acc_test:.4f}")

    # === Evaluación final ===
    A_test = predecir(red, X_test)
    pred_test = np.argmax(A_test, axis=1)
    accuracy_test = float(np.mean(pred_test == Y_test_num))

    matriz_confusion = np.zeros((10, 10), dtype=int)
    for real, pred in zip(Y_test_num, pred_test):
        matriz_confusion[real, pred] += 1

    metrics = {
        "epochs": epochs,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "loss_train_final": float(historial_loss_train[-1]),
        "accuracy_test": accuracy_test,
        "matriz_confusion": matriz_confusion.tolist(),
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Guardamos los pesos entrenados para que demo_gradio.py no tenga que reentrenar
    np.savez(
        RESULTS_DIR / "red_pesos.npz",
        W1=red[0].W, b1=red[0].b, W2=red[2].W, b2=red[2].b,
    )

    print(f"Accuracy en test: {accuracy_test:.4f} ({len(X_test)} imágenes)")

    # === Gráfico 1: curva de aprendizaje ===
    plt.figure(figsize=(6, 4))
    plt.plot(historial_loss_train, color="blue")
    plt.title("Curva de aprendizaje: reconocimiento de dígitos", fontweight="bold")
    plt.xlabel("Épocas")
    plt.ylabel("Pérdida (entropía cruzada, train)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "learning_curve.png", dpi=150)
    plt.close()

    # === Gráfico 2: visualización de datos = muestra de dígitos de entrenamiento ===
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.5))
    for digito in range(10):
        idx = np.where(Y_train_num == digito)[0][0]
        ax = axes[digito // 5, digito % 5]
        ax.imshow(X_train[idx].reshape(28, 28), cmap="gray")
        ax.set_title(f"Dígito {digito}", fontsize=10)
        ax.axis("off")
    plt.suptitle("Muestra del dataset de entrenamiento (MNIST)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "data_visualization.png", dpi=150)
    plt.close()

    # === Gráfico 3: matriz de confusión (test) ===
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matriz_confusion, cmap="Blues")
    ax.set_title(f"Matriz de confusión test (accuracy={accuracy_test:.2%})", fontweight="bold")
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    for i in range(10):
        for j in range(10):
            valor = matriz_confusion[i, j]
            if valor > 0:
                ax.text(j, i, str(valor), ha="center", va="center",
                         color="white" if valor > matriz_confusion.max() / 2 else "black", fontsize=8)
    fig.colorbar(im, fraction=0.046)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    print(f"Resultados y pesos guardados en {RESULTS_DIR}")


if __name__ == "__main__":
    main()
