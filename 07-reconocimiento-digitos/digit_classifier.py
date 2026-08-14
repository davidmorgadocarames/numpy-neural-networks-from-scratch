"""
Reconocimiento de dígitos manuscritos (MNIST) con una red neuronal escrita 100% en NumPy
(sin TensorFlow/Keras) -- el proyecto más avanzado del conjunto de clasificadores densos. Red
modular 784 -> 128 (LeakyReLU) -> 10 (Softmax), entrenada con entropía cruzada.

Split en TRES partes -- train / validación / test, estratificado por dígito -- con early
stopping mirando el error de VALIDACIÓN en cada época, igual que en 04-prediccion-temperatura-
dia-noche, 05-precio-casas, 03-tipos-clientes y 06-zonas-espirales. Antes solo había
train/test y la accuracy de test se muestreaba cada 20 épocas durante el propio entrenamiento
sin que esa señal decidiera nada -- inofensivo mientras nadie mira esa curva para decidir
cuándo parar, pero metodológicamente sucio: el test debe tocarse una única vez, con la red ya
congelada. Los pesos entrenados se guardan en results/red_pesos.npz para que demo_gradio.py
pueda cargarlos sin tener que reentrenar.

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

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
N_TRAIN = 1200
N_VAL = 300
N_TEST = 300

# Early stopping: para el entrenamiento en cuanto el error de VALIDACIÓN deja de mejorar de
# verdad (ver README de 04/05 para la explicación completa).
PACIENCIA_EARLY_STOP = 200
MEJORA_MINIMA_RELATIVA = 0.005


def main(seed_split=42, seed_modelo=42, quiet=False, guardar_graficas=True) -> dict:
    """seed_split y seed_modelo son independientes entre sí -- no hay SEED_DATOS porque MNIST
    es un dataset real y fijo, no generado sintéticamente: lo único aleatorio es qué imágenes
    se muestrean (seed_split) y cómo se inicializa la red (seed_modelo). Ver README para el
    análisis de robustez sobre múltiples semillas."""
    if not quiet:
        print("Descargando MNIST (sklearn fetch_openml)...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
    X_puro, Y_puro = mnist.data, mnist.target.astype(int)

    # Muestra estratificada: mismo número de train/val/test por dígito para que la matriz de
    # confusión no esté sesgada por clases con más ejemplos que otras.
    rng = np.random.default_rng(seed_split)
    indices_train, indices_val, indices_test = [], [], []
    for digito in range(10):
        idx_digito = np.where(Y_puro == digito)[0]
        rng.shuffle(idx_digito)
        corte_train = N_TRAIN // 10
        corte_val = corte_train + N_VAL // 10
        corte_test = corte_val + N_TEST // 10
        indices_train.extend(idx_digito[:corte_train])
        indices_val.extend(idx_digito[corte_train:corte_val])
        indices_test.extend(idx_digito[corte_val:corte_test])
    indices_train = np.array(indices_train)
    indices_val = np.array(indices_val)
    indices_test = np.array(indices_test)
    rng.shuffle(indices_train)
    rng.shuffle(indices_val)
    rng.shuffle(indices_test)

    X_train = X_puro[indices_train] / 255.0
    Y_train_num = Y_puro[indices_train]
    X_val = X_puro[indices_val] / 255.0
    Y_val_num = Y_puro[indices_val]
    X_test = X_puro[indices_test] / 255.0
    Y_test_num = Y_puro[indices_test]

    Y_train = np.zeros((len(X_train), 10))
    Y_train[np.arange(len(X_train)), Y_train_num] = 1
    Y_val = np.zeros((len(X_val), 10))
    Y_val[np.arange(len(X_val)), Y_val_num] = 1

    rng_modelo = np.random.default_rng(seed_modelo)
    red = [
        CapaDensa(dim_entrada=784, dim_salida=128, rng=rng_modelo),
        ActivacionLeakyReLU(),
        CapaDensa(dim_entrada=128, dim_salida=10, rng=rng_modelo),
        ActivacionSoftmax(),
    ]

    learning_rate = 0.1
    epochs = 800
    historial_loss_train, historial_loss_val = [], []

    # Checkpoint del mejor punto de validación: el early stopping corta ~PACIENCIA_EARLY_STOP
    # épocas DESPUÉS del mínimo real de loss_val, así que quedarse con los pesos de la época de
    # corte sería quedarse con pesos peores que los del mínimo. Se guarda una copia de los pesos
    # de cada CapaDensa cada vez que loss_val marca un nuevo mínimo, y se restauran al salir del
    # bucle. Importante: .copy(), no una referencia -- si no, "restaurar" acabaría dejando los
    # pesos finales (los arrays se siguen modificando in-place en cada paso de gradiente).
    mejor_loss_val = np.inf
    mejor_epoca = None
    mejores_pesos = None

    if not quiet:
        print(f"Entrenando con {len(X_train)} imágenes, validando sobre {len(X_val)}, "
              f"test reservado con {len(X_test)}...")
    for epoch in range(epochs):
        activacion = X_train
        for capa in red:
            activacion = capa.forward(activacion)
        A_train = activacion

        loss_train = -np.mean(np.sum(Y_train * np.log(A_train + 1e-15), axis=1))
        historial_loss_train.append(loss_train)

        A_val = predecir(red, X_val)
        loss_val = -np.mean(np.sum(Y_val * np.log(A_val + 1e-15), axis=1))
        historial_loss_val.append(loss_val)

        if loss_val < mejor_loss_val:
            mejor_loss_val = loss_val
            mejor_epoca = epoch
            mejores_pesos = [(c.W.copy(), c.b.copy()) for c in red if isinstance(c, CapaDensa)]

        gradiente = (A_train - Y_train) / Y_train.shape[0]
        for capa in reversed(red[:-1]):
            gradiente = capa.backward(gradiente, learning_rate)

        if not quiet and (epoch % 100 == 0 or epoch == epochs - 1):
            print(f"Época {epoch}/{epochs} - loss train: {loss_train:.4f} - loss val: {loss_val:.4f}")

        if epoch >= PACIENCIA_EARLY_STOP:
            loss_val_referencia = historial_loss_val[epoch - PACIENCIA_EARLY_STOP]
            mejora_relativa = (loss_val_referencia - loss_val) / loss_val_referencia
            if mejora_relativa < MEJORA_MINIMA_RELATIVA:
                if not quiet:
                    print(f"Early stopping en la época {epoch + 1}: el error de validación lleva "
                          f"{PACIENCIA_EARLY_STOP} épocas sin mejorar un {MEJORA_MINIMA_RELATIVA:.1%}")
                break
    else:
        if not quiet:
            print(f"Entrenamiento completado sin activar el early stopping (llegó a la época {epochs})")

    epocas_entrenadas = len(historial_loss_train)

    for capa, (W, b) in zip([c for c in red if isinstance(c, CapaDensa)], mejores_pesos):
        capa.W, capa.b = W, b
    if not quiet:
        print(f"Pesos restaurados al mínimo de validación: época {mejor_epoca + 1} "
              f"(loss_val={mejor_loss_val:.6f}), frente a la época de corte {epocas_entrenadas}")

    # === Única vez que se toca el conjunto de test, ya con la red entrenada (pesos del mínimo
    # de validación, no los de la última época entrenada) ===
    A_test = predecir(red, X_test)
    pred_test = np.argmax(A_test, axis=1)
    accuracy_test = float(np.mean(pred_test == Y_test_num))

    matriz_confusion = np.zeros((10, 10), dtype=int)
    for real, pred in zip(Y_test_num, pred_test):
        matriz_confusion[real, pred] += 1

    metrics = {
        "seed_split": seed_split,
        "seed_modelo": seed_modelo,
        "epochs_configuradas": epochs,
        "epochs_entrenadas": epocas_entrenadas,
        "epoca_mejor_val": mejor_epoca + 1,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "loss_train_final": float(historial_loss_train[mejor_epoca]),
        "loss_val_final": float(mejor_loss_val),
        "accuracy_test": accuracy_test,
        "matriz_confusion": matriz_confusion.tolist(),
    }

    if not guardar_graficas:
        return {**metrics, "historial_loss_val": historial_loss_val}

    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Guardamos los pesos entrenados para que demo_gradio.py no tenga que reentrenar
    np.savez(
        RESULTS_DIR / "red_pesos.npz",
        W1=red[0].W, b1=red[0].b, W2=red[2].W, b2=red[2].b,
    )

    if not quiet:
        print(f"Accuracy en test: {accuracy_test:.4f} ({len(X_test)} imágenes)")

    # === Gráfico 1: curva de aprendizaje (train + validación, que es lo que decide cuándo
    # parar) ===
    plt.figure(figsize=(6, 4))
    plt.plot(historial_loss_train, color="blue", label="Train")
    plt.plot(historial_loss_val, color="orange", linestyle="--", label="Validación")
    plt.title("Curva de aprendizaje: reconocimiento de dígitos", fontweight="bold")
    plt.xlabel("Épocas")
    plt.ylabel("Pérdida (entropía cruzada)")
    plt.legend()
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

    if not quiet:
        print(f"Resultados y pesos guardados en {RESULTS_DIR}")

    return {**metrics, "historial_loss_val": historial_loss_val}


if __name__ == "__main__":
    main()
