"""
CNN (Conv2D + MaxPool + Dropout) sobre Fashion-MNIST, escrita 100% en NumPy -- sin
TensorFlow, sin Keras, sin PyTorch, ni siquiera para la parte convolucional (ver
`capas_cnn.py`, que implementa la convolución a mano con la técnica im2col). Es el proyecto
más avanzado de este repo: los 7 anteriores usan solo capas densas, este añade convoluciones,
pooling y dropout desde cero.

La red se entrena **tres veces con la misma arquitectura y los mismos pesos iniciales**: tal
cual (*baseline*), con data augmentation completa (flip horizontal + rotación + zoom +
desplazamiento aleatorios, reimplementados a mano en `capas_cnn.augmentar_lote` -- ver ese
módulo para el porqué de cada elección) y con la misma augmentation pero sin flip
(*augmented_sin_flip*, prob_flip=0.0), para aislar si el flip específicamente ayuda o
perjudica en vez de asumirlo -- ver README para el estudio completo.

Split en tres partes -- train / validación / test -- estratificado por clase. El early
stopping de las TRES variantes decide cuándo parar mirando el loss de VALIDACIÓN, nunca el de
test: el test se evalúa una única vez por versión, después de que el entrenamiento ya ha
terminado, así que la comparación final entre variantes es limpia por construcción -- no hace
falta igualar manualmente ningún criterio de parada, cada una para cuando su propia validación
deja de mejorar.

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


def cargar_datos(seed_split, quiet=False):
    if not quiet:
        print("Descargando Fashion-MNIST (sklearn fetch_openml)...")
    datos = fetch_openml("Fashion-MNIST", version=1, as_frame=False, parser="liac-arff")
    X_puro, Y_puro = datos.data, datos.target.astype(int)

    # Muestra estratificada (mismo patrón que 07-reconocimiento-digitos): mismo número de
    # train/validación/test por clase para que la matriz de confusión no esté sesgada.
    rng = np.random.default_rng(seed_split)
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


def crear_red(rng_modelo):
    """Misma arquitectura y mismos pesos iniciales para las tres variantes -- se llama con un
    rng_modelo fresco (recién creado a partir de seed_modelo) cada vez, así que cualquier
    diferencia final entre entrenamientos se debe a los datos (con o sin augmentation), no a la
    inicialización. El mismo rng también gobierna las máscaras de CapaDropout durante el
    entrenamiento -- su secuencia de sorteos es idéntica en las tres variantes hasta el punto
    en que cada una para (early stopping), por la misma razón."""
    return [
        CapaConv2D(3, 3, 1, 8, rng=rng_modelo), ActivacionLeakyReLU(), CapaMaxPool2D(2, 2),
        CapaConv2D(3, 3, 8, 16, rng=rng_modelo), ActivacionLeakyReLU(), CapaMaxPool2D(2, 2),
        CapaFlatten(),
        CapaDensa(400, 64, semilla_he=True, rng=rng_modelo), ActivacionLeakyReLU(),
        CapaDropout(0.3, rng=rng_modelo),
        CapaDensa(64, 10, semilla_he=True, rng=rng_modelo), ActivacionSoftmax(),
    ]


def entrenar(nombre, red, X_train, Y_train, X_val, Y_val_num, usar_augmentation, rng_aug, prob_flip=0.5, quiet=False):
    """Bucle de entrenamiento full-batch (mismo estilo que el resto del repo), con early
    stopping por ventana de mejora relativa sobre el loss de VALIDACIÓN (ver README de
    02-celsius-fahrenheit / 05-precio-casas para la explicación general del criterio). El
    conjunto de test no entra en esta función -- no se toca hasta después de entrenar.

    `prob_flip` se pasa tal cual a `augmentar_lote()` -- permite aislar el efecto del flip
    horizontal del resto de la augmentation (rotación/zoom/desplazamiento) usando la misma
    `rng_aug` con la misma semilla: como `augmentar_lote` siempre consume el mismo número de
    valores aleatorios en el mismo orden (primero decide el flip, comparando contra el umbral,
    luego rotación/zoom/desplazamiento), dos llamadas con distinto `prob_flip` pero la misma
    semilla producen exactamente las mismas rotaciones/zooms/desplazamientos -- la única
    diferencia real entre ambas es si se aplica el flip o no.

    Checkpoint del mejor punto de validación: el early stopping corta ~PACIENCIA_EARLY_STOP
    épocas DESPUÉS del mínimo real de loss_val, así que quedarse con los pesos de la época de
    corte sería quedarse con pesos peores que los del mínimo. Se guarda una copia de los pesos
    de cada capa con parámetros propios (CapaDensa y CapaConv2D, ambas con W/b) cada vez que
    loss_val marca un nuevo mínimo, y se restauran sobre `red` (in-place) al salir del bucle.
    Importante: .copy(), no una referencia -- si no, "restaurar" acabaría dejando los pesos
    finales (los arrays se siguen modificando in-place en cada paso de gradiente)."""
    Y_val_onehot = np.eye(10)[Y_val_num]
    historial_loss_train, historial_loss_val, historial_acc_val = [], [], []
    capas_con_pesos = [c for c in red if isinstance(c, (CapaDensa, CapaConv2D))]
    mejor_loss_val = np.inf
    mejor_epoca = None
    mejores_pesos = None
    t0 = time.time()

    for epoch in range(EPOCHS_MAX):
        X_epoca = augmentar_lote(X_train, rng_aug, prob_flip=prob_flip) if usar_augmentation else X_train

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

        if loss_val < mejor_loss_val:
            mejor_loss_val = loss_val
            mejor_epoca = epoch
            mejores_pesos = [(c.W.copy(), c.b.copy()) for c in capas_con_pesos]

        gradiente = (probs_train - Y_train) / Y_train.shape[0]
        for capa in reversed(red[:-1]):
            gradiente = capa.backward(gradiente, LEARNING_RATE)

        if not quiet and epoch % 50 == 0:
            print(f"  [{nombre}] época {epoch}: loss_train={loss_train:.4f} "
                  f"loss_val={loss_val:.4f} acc_val={acc_val:.4f}")

        if epoch >= PACIENCIA_EARLY_STOP:
            loss_val_referencia = historial_loss_val[epoch - PACIENCIA_EARLY_STOP]
            mejora_relativa = (loss_val_referencia - loss_val) / loss_val_referencia
            if mejora_relativa < MEJORA_MINIMA_RELATIVA:
                if not quiet:
                    print(f"  [{nombre}] early stopping en la época {epoch + 1}: el loss de "
                          f"validación lleva {PACIENCIA_EARLY_STOP} épocas sin mejorar un "
                          f"{MEJORA_MINIMA_RELATIVA:.1%} ({time.time() - t0:.0f}s)")
                break
    else:
        if not quiet:
            print(f"  [{nombre}] completó las {EPOCHS_MAX} épocas sin activar el early stopping "
                  f"({time.time() - t0:.0f}s)")

    for capa, (W, b) in zip(capas_con_pesos, mejores_pesos):
        capa.W, capa.b = W, b
    if not quiet:
        print(f"  [{nombre}] pesos restaurados al mínimo de validación: época {mejor_epoca + 1} "
              f"(loss_val={mejor_loss_val:.6f}), frente a la época de corte {len(historial_loss_train)}")

    return historial_loss_train, historial_loss_val, historial_acc_val, mejor_epoca, mejor_loss_val


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


def main(seed_split=42, seed_modelo=42, quiet=False, guardar_graficas=True) -> dict:
    """seed_split y seed_modelo son independientes -- no hay SEED_DATOS porque Fashion-MNIST es
    un dataset real y fijo. seed_modelo se resetea a un rng fresco antes de cada una de las 3
    variantes (ver crear_red) para que sigan partiendo de los mismos pesos iniciales -- la
    comparación baseline/augmented/augmented_sin_flip sigue siendo justa dentro de cada semilla
    de la barrida. Ver README para el análisis de robustez sobre múltiples semillas."""
    X_train, Y_train, Y_train_num, X_val, Y_val_num, X_test, Y_test_num = cargar_datos(seed_split, quiet=quiet)
    if not quiet:
        print(f"Entrenando con {len(X_train)} imágenes, validando con {len(X_val)}, "
              f"evaluando sobre {len(X_test)} de test...")

    if guardar_graficas:
        # === Gráfico de datos: muestra del dataset + ejemplo de lo que hace la augmentation ===
        rng_demo = np.random.default_rng(seed_modelo)
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

    # Tercera variante -- "augmented_sin_flip": misma augmentation que "augmented" (rotación,
    # zoom, desplazamiento) pero con prob_flip=0.0, para aislar si el flip horizontal ayuda o
    # perjudica en Fashion-MNIST. Tiene sentido dudarlo: a diferencia de dígitos manuscritos,
    # varias prendas de Fashion-MNIST no son simétricas en la práctica (zapatillas, sandalias,
    # botines tienen una orientación de puntera; algunas camisetas llevan estampados no
    # simétricos) -- reflejarlas podría estar enseñando a la red una variación que no se
    # corresponde con cómo aparecen las prendas reales.
    resultados = {}
    curvas = {}
    variantes = [
        ("baseline", False, 0.5),
        ("augmented", True, 0.5),
        ("augmented_sin_flip", True, 0.0),
    ]
    for nombre, usar_aug, prob_flip in variantes:
        if not quiet:
            print(f"\n=== Entrenando '{nombre}' (augmentation={usar_aug}, prob_flip={prob_flip}) ===")
        rng_modelo = np.random.default_rng(seed_modelo)  # mismos pesos iniciales en las tres variantes
        red = crear_red(rng_modelo)
        rng_aug = np.random.default_rng(seed_modelo)
        hist_train, hist_val, hist_acc_val, mejor_epoca, mejor_loss_val = entrenar(
            nombre, red, X_train, Y_train, X_val, Y_val_num, usar_aug, rng_aug, prob_flip=prob_flip, quiet=quiet
        )
        # evaluar() se llama con los pesos ya restaurados al mínimo de validación dentro de
        # entrenar() -- test sigue tocándose una única vez, con la red congelada.
        accuracy, matriz_confusion = evaluar(red, X_test, Y_test_num)
        if not quiet:
            print(f"  [{nombre}] accuracy final en test: {accuracy:.4f}")

        if guardar_graficas:
            graficar_confusion(
                matriz_confusion, accuracy,
                f"Matriz de confusión test — {nombre}",
                RESULTS_DIR / f"confusion_matrix_{nombre}.png",
            )

        resultados[nombre] = {
            "epochs_entrenadas": len(hist_train),
            "epoca_mejor_val": mejor_epoca + 1,
            "accuracy_test": accuracy,
            "loss_train_final": float(hist_train[mejor_epoca]),
            "loss_val_final": float(mejor_loss_val),
            "matriz_confusion": matriz_confusion.tolist(),
            "historial_accuracy_val": hist_acc_val,
            "historial_loss_val": hist_val,
        }
        curvas[nombre] = hist_acc_val

    metrics = {
        "seed_split": seed_split,
        "seed_modelo": seed_modelo,
        "epochs_max_configuradas": EPOCHS_MAX,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "clases": NOMBRES_CLASES,
        "resultados": resultados,
    }

    if not guardar_graficas:
        return metrics

    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # === Curva de aprendizaje comparativa: accuracy de VALIDACIÓN en las tres variantes (es lo
    # que se registra época a época; el test es un único número final por versión) ===
    plt.figure(figsize=(8, 4.5))
    plt.plot(curvas["baseline"], color="blue", label="Baseline (sin augmentation)")
    plt.plot(curvas["augmented"], color="green", linestyle="--", label="Augmented (flip=0.5)")
    plt.plot(curvas["augmented_sin_flip"], color="red", linestyle=":", label="Augmented sin flip (flip=0.0)")
    plt.title("Accuracy en validación: baseline vs augmentation con/sin flip", fontweight="bold")
    plt.xlabel("Épocas")
    plt.ylabel("Accuracy en validación")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "learning_curve.png", dpi=150)
    plt.close()

    if not quiet:
        print("\n=== Resumen ===")
        for nombre, _, _ in variantes:
            print(f"{nombre:20s} {resultados[nombre]['accuracy_test']:.4f} "
                  f"({resultados[nombre]['epochs_entrenadas']} épocas)")
        print(f"Resultados guardados en {RESULTS_DIR}")

    return metrics


if __name__ == "__main__":
    main()
