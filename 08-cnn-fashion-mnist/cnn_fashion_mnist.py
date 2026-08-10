"""
CNN (Conv2D + MaxPool + Dropout) sobre Fashion-MNIST, escrita 100% en NumPy -- sin
TensorFlow, sin Keras, sin PyTorch, ni siquiera para la parte convolucional (ver
`capas_cnn.py`, que implementa la convolución a mano con la técnica im2col). Es el proyecto
más avanzado de este repo: los 7 anteriores usan solo capas densas, este añade convoluciones,
pooling y dropout desde cero.

La red se entrena **dos veces con la misma arquitectura y los mismos pesos iniciales**: una
vez tal cual (*baseline*) y otra con data augmentation (flip horizontal + rotación + zoom +
desplazamiento aleatorios, reimplementados a mano en `capas_cnn.augmentar_lote` -- ver ese
módulo para el porqué de cada elección), para medir el efecto real de la técnica en vez de
solo mencionarla.

Split en tres partes -- train / validación / test -- estratificado por clase. El early
stopping de AMBOS entrenamientos (baseline y augmented) decide cuándo parar mirando el loss de
VALIDACIÓN, nunca el de test: el test se evalúa una única vez por versión, después de que el
entrenamiento ya ha terminado, así que la comparación final baseline vs augmented es limpia
por construcción -- no hace falta igualar manualmente ningún criterio de parada entre las dos
versiones, cada una para cuando su propia validación deja de mejorar.

Se usa Fashion-MNIST en lugar de MNIST porque es lo bastante difícil como para justificar
convoluciones (los dígitos de MNIST se resuelven casi igual de bien con una red densa) y no
requiere descarga manual de datos (se descarga vía sklearn.fetch_openml, igual que MNIST en
`07-reconocimiento-digitos`).

Uso: python cnn_fashion_mnist.py
"""

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from capas import ActivacionLeakyReLU, ActivacionSoftmax, CapaDensa
from capas_cnn import CapaConv2D, CapaDropout, CapaFlatten, CapaMaxPool2D, augmentar_lote, predecir_cnn

SEED = 42
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# 240 train (60%) / 60 validación (20%, decide el early stopping) / 100 test (20%, se evalúa
# una sola vez, después de entrenar) por cada una de las 10 clases.
N_POR_CLASE_TRAIN = 240
N_POR_CLASE_VAL = 60
N_POR_CLASE_TEST = 100
NOMBRES_CLASES = [
    "Camiseta", "Pantalón", "Jersey", "Vestido", "Abrigo",
    "Sandalia", "Camisa", "Zapatilla", "Bolso", "Botín",
]

LEARNING_RATE = 0.12
EPOCHS_MAX = 400
# Ventana de mejora relativa sobre el loss de VALIDACIÓN (ver README de 02-celsius-fahrenheit /
# 05-precio-casas para la explicación general del criterio). Al decidir sobre validación en
# vez de test, baseline y augmented pueden compararse sin ningún ajuste manual de por medio:
# cada entrenamiento para cuando su propia validación dice que ya no mejora.
PACIENCIA_EARLY_STOP = 150
MEJORA_MINIMA_RELATIVA = 0.005


def cargar_datos():
    print("Descargando Fashion-MNIST (sklearn fetch_openml)...")
    datos = fetch_openml("Fashion-MNIST", version=1, as_frame=False, parser="liac-arff")
    X_puro, Y_puro = datos.data, datos.target.astype(int)

    # Muestra estratificada (mismo patrón que 07-reconocimiento-digitos): mismo número de
    # train/validación/test por clase para que la matriz de confusión no esté sesgada.
    rng = np.random.default_rng(SEED)
    idx_train, idx_val, idx_test = [], [], []
    for clase in range(10):
        idx_clase = np.where(Y_puro == clase)[0]
        rng.shuffle(idx_clase)
        fin_train = N_POR_CLASE_TRAIN
        fin_val = N_POR_CLASE_TRAIN + N_POR_CLASE_VAL
        fin_test = fin_val + N_POR_CLASE_TEST
        idx_train.extend(idx_clase[:fin_train])
        idx_val.extend(idx_clase[fin_train:fin_val])
        idx_test.extend(idx_clase[fin_val:fin_test])
    idx_train, idx_val, idx_test = np.array(idx_train), np.array(idx_val), np.array(idx_test)
    rng.shuffle(idx_train)
    rng.shuffle(idx_val)
    rng.shuffle(idx_test)

    X_train = (X_puro[idx_train] / 255.0).reshape(-1, 28, 28, 1)
    Y_train_num = Y_puro[idx_train]
    X_val = (X_puro[idx_val] / 255.0).reshape(-1, 28, 28, 1)
    Y_val_num = Y_puro[idx_val]
    X_test = (X_puro[idx_test] / 255.0).reshape(-1, 28, 28, 1)
    Y_test_num = Y_puro[idx_test]
    Y_train = np.eye(10)[Y_train_num]

    return X_train, Y_train, Y_train_num, X_val, Y_val_num, X_test, Y_test_num


def crear_red():
    """Misma arquitectura y mismos pesos iniciales para baseline y augmented -- se llama con
    la semilla global reseteada justo antes, así que cualquier diferencia final entre ambos
    entrenamientos se debe a los datos (con o sin augmentation), no a la inicialización."""
    return [
        CapaConv2D(3, 3, 1, 8), ActivacionLeakyReLU(), CapaMaxPool2D(2, 2),
        CapaConv2D(3, 3, 8, 16), ActivacionLeakyReLU(), CapaMaxPool2D(2, 2),
        CapaFlatten(),
        CapaDensa(400, 64, semilla_he=True), ActivacionLeakyReLU(),
        CapaDropout(0.3),
        CapaDensa(64, 10, semilla_he=True), ActivacionSoftmax(),
    ]


def entrenar(nombre, red, X_train, Y_train, X_val, Y_val_num, usar_augmentation, rng_aug):
    """Bucle de entrenamiento full-batch (mismo estilo que el resto del repo), con early
    stopping por ventana de mejora relativa sobre el loss de VALIDACIÓN (ver README de
    02-celsius-fahrenheit / 05-precio-casas para la explicación general del criterio). El
    conjunto de test no entra en esta función -- no se toca hasta después de entrenar."""
    Y_val_onehot = np.eye(10)[Y_val_num]
    historial_loss_train, historial_loss_val, historial_acc_val = [], [], []
    t0 = time.time()

    for epoch in range(EPOCHS_MAX):
        X_epoca = augmentar_lote(X_train, rng_aug) if usar_augmentation else X_train

        activacion = X_epoca
        for capa in red:
            activacion = capa.forward(activacion, entrenando=True)
        probs_train = activacion
        loss_train = -np.mean(np.sum(Y_train * np.log(probs_train + 1e-15), axis=1))
        historial_loss_train.append(loss_train)

        # Validación: SIEMPRE sobre imágenes originales, sin augmentation -- la augmentation es
        # una técnica de entrenamiento, no algo que la red vaya a ver en producción.
        probs_val = predecir_cnn(red, X_val)
        loss_val = -np.mean(np.sum(Y_val_onehot * np.log(probs_val + 1e-15), axis=1))
        historial_loss_val.append(loss_val)
        acc_val = float(np.mean(np.argmax(probs_val, axis=1) == Y_val_num))
        historial_acc_val.append(acc_val)

        gradiente = (probs_train - Y_train) / Y_train.shape[0]
        for capa in reversed(red[:-1]):
            gradiente = capa.backward(gradiente, LEARNING_RATE)

        if epoch % 50 == 0:
            print(f"  [{nombre}] época {epoch}: loss_train={loss_train:.4f} "
                  f"loss_val={loss_val:.4f} acc_val={acc_val:.4f}")

        if epoch >= PACIENCIA_EARLY_STOP:
            loss_val_referencia = historial_loss_val[epoch - PACIENCIA_EARLY_STOP]
            mejora_relativa = (loss_val_referencia - loss_val) / loss_val_referencia
            if mejora_relativa < MEJORA_MINIMA_RELATIVA:
                print(f"  [{nombre}] early stopping en la época {epoch + 1}: el loss de "
                      f"validación lleva {PACIENCIA_EARLY_STOP} épocas sin mejorar un "
                      f"{MEJORA_MINIMA_RELATIVA:.1%} ({time.time() - t0:.0f}s)")
                break
    else:
        print(f"  [{nombre}] completó las {EPOCHS_MAX} épocas sin activar el early stopping "
              f"({time.time() - t0:.0f}s)")

    return historial_loss_train, historial_loss_val, historial_acc_val


def evaluar(red, X_test, Y_test_num):
    """Única evaluación sobre test, después de que entrenar() ya ha terminado y la red está
    congelada."""
    probs_test = predecir_cnn(red, X_test)
    pred_test = np.argmax(probs_test, axis=1)
    accuracy = float(np.mean(pred_test == Y_test_num))
    matriz_confusion = np.zeros((10, 10), dtype=int)
    for real, pred in zip(Y_test_num, pred_test):
        matriz_confusion[real, pred] += 1
    return accuracy, matriz_confusion


def graficar_confusion(matriz, accuracy, titulo, ruta):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matriz, cmap="Blues")
    ax.set_title(f"{titulo} (accuracy={accuracy:.2%})", fontweight="bold")
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(NOMBRES_CLASES, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(NOMBRES_CLASES, fontsize=8)
    for i in range(10):
        for j in range(10):
            valor = matriz[i, j]
            if valor > 0:
                ax.text(j, i, str(valor), ha="center", va="center", fontsize=7,
                         color="white" if valor > matriz.max() / 2 else "black")
    fig.colorbar(im, fraction=0.046)
    plt.tight_layout()
    plt.savefig(ruta, dpi=150)
    plt.close()


def main() -> None:
    X_train, Y_train, Y_train_num, X_val, Y_val_num, X_test, Y_test_num = cargar_datos()
    print(f"Entrenando con {len(X_train)} imágenes, validando con {len(X_val)}, "
          f"evaluando sobre {len(X_test)} de test...")

    # === Gráfico de datos: muestra del dataset + ejemplo de lo que hace la augmentation ===
    rng_demo = np.random.default_rng(SEED)
    fig, axes = plt.subplots(3, 10, figsize=(16, 5.5))
    for clase in range(10):
        idx = np.where(Y_train_num == clase)[0][0]
        axes[0, clase].imshow(X_train[idx, :, :, 0], cmap="gray")
        axes[0, clase].set_title(NOMBRES_CLASES[clase], fontsize=8)
    ejemplo = X_train[np.where(Y_train_num == 0)[0][0]:np.where(Y_train_num == 0)[0][0] + 1]
    for fila in (1, 2):
        aug = augmentar_lote(np.repeat(ejemplo, 10, axis=0), rng_demo)
        for col in range(10):
            axes[fila, col].imshow(aug[col, :, :, 0], cmap="gray")
    for fila, etiqueta in [(0, "Original"), (1, "Augmentation\n(ej. 1)"), (2, "Augmentation\n(ej. 2)")]:
        for col in range(10):
            axes[fila, col].set_xticks([])
            axes[fila, col].set_yticks([])
        axes[fila, 0].set_ylabel(etiqueta, fontsize=9)
    plt.suptitle("Fila 1: una imagen por clase — Filas 2-3: la MISMA imagen (clase 0, "
                 "Camiseta) tras 10 augmentations aleatorias distintas", fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "data_visualization.png", dpi=150)
    plt.close()

    resultados = {}
    curvas = {}
    for nombre, usar_aug in [("baseline", False), ("augmented", True)]:
        print(f"\n=== Entrenando '{nombre}' (augmentation={usar_aug}) ===")
        np.random.seed(SEED)  # mismos pesos iniciales para los dos entrenamientos
        red = crear_red()
        rng_aug = np.random.default_rng(SEED)
        hist_train, hist_val, hist_acc_val = entrenar(
            nombre, red, X_train, Y_train, X_val, Y_val_num, usar_aug, rng_aug
        )
        accuracy, matriz_confusion = evaluar(red, X_test, Y_test_num)
        print(f"  [{nombre}] accuracy final en test: {accuracy:.4f}")

        graficar_confusion(
            matriz_confusion, accuracy,
            f"Matriz de confusión test — {nombre}",
            RESULTS_DIR / f"confusion_matrix_{nombre}.png",
        )

        resultados[nombre] = {
            "epochs_entrenadas": len(hist_train),
            "accuracy_test": accuracy,
            "loss_train_final": float(hist_train[-1]),
            "loss_val_final": float(hist_val[-1]),
            "matriz_confusion": matriz_confusion.tolist(),
            "historial_accuracy_val": hist_acc_val,
        }
        curvas[nombre] = hist_acc_val

    metrics = {
        "epochs_max_configuradas": EPOCHS_MAX,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "clases": NOMBRES_CLASES,
        "resultados": resultados,
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # === Curva de aprendizaje comparativa: accuracy de VALIDACIÓN, baseline vs augmented (es
    # lo que se registra época a época; el test es un único número final por versión) ===
    plt.figure(figsize=(8, 4.5))
    plt.plot(curvas["baseline"], color="blue", label="Baseline (sin augmentation)")
    plt.plot(curvas["augmented"], color="green", linestyle="--", label="Con data augmentation")
    plt.title("Accuracy en validación: baseline vs data augmentation", fontweight="bold")
    plt.xlabel("Épocas")
    plt.ylabel("Accuracy en validación")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "learning_curve.png", dpi=150)
    plt.close()

    print("\n=== Resumen ===")
    print(f"Baseline:   {resultados['baseline']['accuracy_test']:.4f} "
          f"({resultados['baseline']['epochs_entrenadas']} épocas)")
    print(f"Augmented:  {resultados['augmented']['accuracy_test']:.4f} "
          f"({resultados['augmented']['epochs_entrenadas']} épocas)")
    print(f"Resultados guardados en {RESULTS_DIR}")


if __name__ == "__main__":
    main()
