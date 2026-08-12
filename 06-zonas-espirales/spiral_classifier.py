"""
Clasificador de 3 brazos de una espiral (NumPy puro) -- el benchmark clásico para demostrar
que una red con capas ocultas puede aprender fronteras de decisión curvas que ningún modelo
lineal podría representar. Red modular profunda (Densa -> LeakyReLU -> Densa -> LeakyReLU ->
Densa -> Softmax), 2 capas ocultas de 64 neuronas.

Split en TRES partes -- 60% train / 20% validación / 20% test, estratificado por brazo -- con
early stopping mirando el error de VALIDACIÓN. Es un cambio necesario, no cosmético: con solo
train/test y una red de ~9k parámetros, este problema sobreajusta a partir de la época ~1047
(el error de test empieza a subir mientras el de train sigue bajando) y sin un conjunto de
validación separado no hay forma legítima de detectar ese punto de corte sin espiar el propio
test -- parar mirando el test y luego reportar accuracy sobre ese mismo test sería trampa. Con
validación, el early stopping decide cuándo parar sin haber tocado el test, que se evalúa una
única vez al final ya con la red congelada.

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

# Early stopping: para el entrenamiento en cuanto el error de VALIDACIÓN deja de mejorar de
# verdad (ver README de 04/05 para la explicación completa).
PACIENCIA_EARLY_STOP = 200
MEJORA_MINIMA_RELATIVA = 0.005


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

    # Split train/val/test estratificado por brazo (60% train / 20% validación / 20% test)
    rng = np.random.default_rng(SEED)
    indices_train, indices_val, indices_test = [], [], []
    for clase in range(K_CLASES):
        idx_clase = np.where(Y_num == clase)[0]
        rng.shuffle(idx_clase)
        corte_train = int(0.6 * len(idx_clase))
        corte_val = int(0.8 * len(idx_clase))
        indices_train.extend(idx_clase[:corte_train])
        indices_val.extend(idx_clase[corte_train:corte_val])
        indices_test.extend(idx_clase[corte_val:])
    indices_train = np.array(indices_train)
    indices_val = np.array(indices_val)
    indices_test = np.array(indices_test)

    X_train, X_val, X_test = X[indices_train], X[indices_val], X[indices_test]
    Y_train, Y_val, Y_test = Y[indices_train], Y[indices_val], Y[indices_test]
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

        if epoch >= PACIENCIA_EARLY_STOP:
            loss_val_referencia = historial_loss_val[epoch - PACIENCIA_EARLY_STOP]
            mejora_relativa = (loss_val_referencia - loss_val) / loss_val_referencia
            if mejora_relativa < MEJORA_MINIMA_RELATIVA:
                print(f"Early stopping en la época {epoch + 1}: el error de validación lleva "
                      f"{PACIENCIA_EARLY_STOP} épocas sin mejorar un {MEJORA_MINIMA_RELATIVA:.1%}")
                break
    else:
        print(f"Entrenamiento completado sin activar el early stopping (llegó a la época {epochs})")

    epocas_entrenadas = len(historial_loss_train)

    for capa, (W, b) in zip([c for c in red if isinstance(c, CapaDensa)], mejores_pesos):
        capa.W, capa.b = W, b
    print(f"Pesos restaurados al mínimo de validación: época {mejor_epoca + 1} "
          f"(loss_val={mejor_loss_val:.6f}), frente a la época de corte {epocas_entrenadas}")

    # === Única vez que se toca el conjunto de test, ya con la red entrenada (pesos del mínimo
    # de validación, no los de la última época entrenada) ===
    A_test = predecir(red, X_test)
    pred_test = np.argmax(A_test, axis=1)
    accuracy_test = float(np.mean(pred_test == Y_num_test))
    loss_test = float(-np.mean(np.sum(Y_test * np.log(A_test + 1e-15), axis=1)))

    matriz_confusion = np.zeros((3, 3), dtype=int)
    for real, pred in zip(Y_num_test, pred_test):
        matriz_confusion[real, pred] += 1

    metrics = {
        "epochs_configuradas": epochs,
        "epochs_entrenadas": epocas_entrenadas,
        "epoca_mejor_val": mejor_epoca + 1,
        "loss_train_final": float(historial_loss_train[mejor_epoca]),
        "loss_val_final": float(mejor_loss_val),
        "loss_test_final": loss_test,
        "accuracy_test": accuracy_test,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "matriz_confusion": matriz_confusion.tolist(),
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Accuracy test: {accuracy_test:.4f} ({len(X_test)} puntos de test)")

    # === Gráfico 1: curva de aprendizaje (train + validación, que es lo que decide cuándo
    # parar) ===
    plt.figure(figsize=(6, 4))
    plt.plot(historial_loss_train, color="purple", label="Train")
    plt.plot(historial_loss_val, color="orange", linestyle="--", label="Validación")
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
    plt.scatter(X_val[:, 0], X_val[:, 1], c=Y_num[indices_val], s=50, cmap="jet", marker="^",
                edgecolors="k", label="Validación")
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
