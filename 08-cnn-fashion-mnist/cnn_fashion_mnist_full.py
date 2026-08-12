"""
Variante de `cnn_fashion_mnist.py` sobre Fashion-MNIST COMPLETO (70.000 imágenes, no la
muestra reducida de 2.400/600/1.000), para poder comparar en el propio repositorio qué gana la
CNN al entrenar con mucha más muestra. Se guarda como script aparte -- no sustituye al
original -- para que ambos resultados (y ambas arquitecturas de entrenamiento) queden
documentados uno junto al otro. Ver `../07-reconocimiento-digitos/digit_classifier_full.py`
para la misma idea aplicada a la red densa de MNIST.

Con 2.400 imágenes, un batch "full" (toda la muestra de golpe, 1 actualización de pesos por
época) era manejable. Con ~42.000 imágenes de train, seguir haciendo full-batch significaría 1
sola actualización de pesos por época sobre una matriz enorme -- converge muy despacio en
número de actualizaciones. La solución estándar es entrenar por MINI-BATCHES: en cada época, la
muestra se recorre en trozos de `BATCH_SIZE` imágenes, con una actualización de pesos por trozo
(~1.313 actualizaciones por época con batch_size=32) en vez de 1 sola.

Split estratificado por clase, pero usando el 60/20/20 de CADA clase sobre las 7.000 imágenes
que tiene cada una de las 10 clases de Fashion-MNIST (a diferencia de MNIST, aquí sí está
perfectamente equilibrado: exactamente 7.000 imágenes por clase, 70.000 en total).

Error clásico a evitar al pasar de full-batch a mini-batch: la normalización del gradiente de
la capa de salida debe dividir por el tamaño del BATCH actual (`Yb.shape[0]`), no por el
tamaño de todo el conjunto de train (`Y_train.shape[0]`). Si se deja el denominador del
full-batch original, el gradiente queda a una escala equivocada en cada batch y el
entrenamiento se vuelve absurdamente lento o inestable sin que salte ningún error explícito.

La augmentation (ver `capas_cnn.augmentar_lote`) se sigue aplicando sobre el conjunto de train
COMPLETO una vez por época, igual que en la versión original -- son operaciones vectorizadas de
NumPy sobre el array entero, no un bucle por imagen, así que no hace falta aplicarla batch a
batch; el resultado de esa época ya augmentada es lo que luego se recorre en mini-batches.

Uso: python cnn_fashion_mnist_full.py
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
RESULTS_DIR = Path(__file__).parent / "results_full"
RESULTS_DIR.mkdir(exist_ok=True)

FRAC_TRAIN, FRAC_VAL = 0.6, 0.2  # el resto (0.2) es test
BATCH_SIZE = 32
NOMBRES_CLASES = [
    "Camiseta", "Pantalón", "Jersey", "Vestido", "Abrigo",
    "Sandalia", "Camisa", "Zapatilla", "Bolso", "Botín",
]

LEARNING_RATE = 0.12
# Con ~1.313 actualizaciones de pesos por época (en vez de 1 en la versión full-batch), la red
# converge en muchas menos épocas -- por eso el techo y la paciencia son mucho más bajos que en
# cnn_fashion_mnist.py (EPOCHS_MAX=400, paciencia=150). Ver README para la comparación medida.
EPOCHS_MAX = 25
PACIENCIA_EARLY_STOP = 5
MEJORA_MINIMA_RELATIVA = 0.005


def cargar_datos():
    print("Descargando Fashion-MNIST (sklearn fetch_openml)...")
    datos = fetch_openml("Fashion-MNIST", version=1, as_frame=False, parser="liac-arff")
    X_puro, Y_puro = datos.data, datos.target.astype(int)

    # Split estratificado 60/20/20 usando las 7.000 imágenes completas de cada clase (Fashion-
    # MNIST está perfectamente equilibrado, a diferencia de MNIST).
    rng = np.random.default_rng(SEED)
    idx_train, idx_val, idx_test = [], [], []
    for clase in range(10):
        idx_clase = np.where(Y_puro == clase)[0]
        rng.shuffle(idx_clase)
        n_clase = len(idx_clase)
        corte_train = int(n_clase * FRAC_TRAIN)
        corte_val = corte_train + int(n_clase * FRAC_VAL)
        idx_train.extend(idx_clase[:corte_train])
        idx_val.extend(idx_clase[corte_train:corte_val])
        idx_test.extend(idx_clase[corte_val:])
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


def generar_batches(X, Y, batch_size, rng):
    """Recorre (X, Y) en mini-batches de tamaño `batch_size`, en un orden aleatorio distinto
    cada vez que se llama. El último batch de la época puede ser más pequeño que `batch_size`
    -- por eso el gradiente se normaliza por el tamaño real de CADA batch."""
    n = X.shape[0]
    indices = rng.permutation(n)
    for inicio in range(0, n, batch_size):
        idx_batch = indices[inicio : inicio + batch_size]
        yield X[idx_batch], Y[idx_batch]


def entrenar(nombre, red, X_train, Y_train, X_val, Y_val_num, usar_augmentation, rng_aug, rng_batches, prob_flip=0.5):
    """Bucle de entrenamiento por MINI-BATCHES, con early stopping por ventana de mejora
    relativa sobre el loss de VALIDACIÓN (ver README de 02-celsius-fahrenheit / 05-precio-casas
    para la explicación general del criterio, y el docstring del módulo para el porqué del
    mini-batch aquí). El conjunto de test no entra en esta función.

    `prob_flip` se pasa tal cual a `augmentar_lote()` -- permite aislar el efecto del flip
    horizontal del resto de la augmentation (rotación/zoom/desplazamiento) usando la misma
    `rng_aug` con la misma semilla: `augmentar_lote` siempre consume el mismo número de valores
    aleatorios en el mismo orden con independencia de `prob_flip`, así que dos llamadas con la
    misma semilla producen exactamente las mismas rotaciones/zooms/desplazamientos -- la única
    diferencia real es si se aplica el flip o no.

    Checkpoint del mejor punto de validación: igual que en el resto del repo, se guarda una
    copia (.copy(), no una referencia) de los pesos de cada capa con parámetros propios
    (CapaDensa y CapaConv2D) cada vez que loss_val marca un nuevo mínimo, y se restauran sobre
    `red` (in-place) al salir del bucle."""
    Y_val_onehot = np.eye(10)[Y_val_num]
    historial_loss_train, historial_loss_val, historial_acc_val = [], [], []
    capas_con_pesos = [c for c in red if isinstance(c, (CapaDensa, CapaConv2D))]
    mejor_loss_val = np.inf
    mejor_epoca = None
    mejores_pesos = None
    t0 = time.time()

    for epoch in range(EPOCHS_MAX):
        X_epoca = augmentar_lote(X_train, rng_aug, prob_flip=prob_flip) if usar_augmentation else X_train

        loss_train_acumulada = 0.0
        n_vistas = 0
        for Xb, Yb in generar_batches(X_epoca, Y_train, BATCH_SIZE, rng_batches):
            activacion = Xb
            for capa in red:
                activacion = capa.forward(activacion, entrenando=True)
            probs_train = activacion

            loss_batch = -np.mean(np.sum(Yb * np.log(probs_train + 1e-15), axis=1))
            loss_train_acumulada += loss_batch * Xb.shape[0]
            n_vistas += Xb.shape[0]

            # Normalizado por el tamaño del BATCH actual (Yb), no por el de todo train -- ver
            # docstring del módulo para el error clásico que esto evita.
            gradiente = (probs_train - Yb) / Yb.shape[0]
            for capa in reversed(red[:-1]):
                gradiente = capa.backward(gradiente, LEARNING_RATE)

        loss_train = loss_train_acumulada / n_vistas
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

        print(f"  [{nombre}] época {epoch}: loss_train={loss_train:.4f} "
              f"loss_val={loss_val:.4f} acc_val={acc_val:.4f} ({time.time() - t0:.0f}s)")

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

    for capa, (W, b) in zip(capas_con_pesos, mejores_pesos):
        capa.W, capa.b = W, b
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


def main() -> None:
    X_train, Y_train, Y_train_num, X_val, Y_val_num, X_test, Y_test_num = cargar_datos()
    n_batches_por_epoca = int(np.ceil(len(X_train) / BATCH_SIZE))
    print(f"Entrenando con {len(X_train)} imágenes ({n_batches_por_epoca} batches/época de "
          f"{BATCH_SIZE}), validando con {len(X_val)}, evaluando sobre {len(X_test)} de test...")

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
                 "Camiseta) tras 10 augmentations aleatorias distintas (dataset completo)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "data_visualization.png", dpi=150)
    plt.close()

    # Tercera variante -- "augmented_sin_flip": misma augmentation que "augmented" (rotación,
    # zoom, desplazamiento) pero con prob_flip=0.0, para aislar si el flip horizontal ayuda o
    # perjudica en Fashion-MNIST. Tiene sentido dudarlo: a diferencia de dígitos manuscritos,
    # varias prendas no son simétricas en la práctica (zapatillas, sandalias, botines tienen
    # una orientación de puntera) -- reflejarlas podría enseñar a la red una variación que no
    # se corresponde con cómo aparecen las prendas reales.
    resultados = {}
    curvas = {}
    variantes = [
        ("baseline", False, 0.5),
        ("augmented", True, 0.5),
        ("augmented_sin_flip", True, 0.0),
    ]
    for nombre, usar_aug, prob_flip in variantes:
        print(f"\n=== Entrenando '{nombre}' (augmentation={usar_aug}, prob_flip={prob_flip}) ===")
        np.random.seed(SEED)  # mismos pesos iniciales en las tres variantes
        red = crear_red()
        rng_aug = np.random.default_rng(SEED)
        rng_batches = np.random.default_rng(SEED)
        hist_train, hist_val, hist_acc_val, mejor_epoca, mejor_loss_val = entrenar(
            nombre, red, X_train, Y_train, X_val, Y_val_num, usar_aug, rng_aug, rng_batches, prob_flip=prob_flip
        )
        # evaluar() se llama con los pesos ya restaurados al mínimo de validación dentro de
        # entrenar() -- test sigue tocándose una única vez, con la red congelada.
        accuracy, matriz_confusion = evaluar(red, X_test, Y_test_num)
        print(f"  [{nombre}] accuracy final en test: {accuracy:.4f}")

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
        }
        curvas[nombre] = hist_acc_val

    metrics = {
        "batch_size": BATCH_SIZE,
        "n_batches_por_epoca": n_batches_por_epoca,
        "epochs_max_configuradas": EPOCHS_MAX,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "clases": NOMBRES_CLASES,
        "resultados": resultados,
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # === Curva de aprendizaje comparativa: accuracy de VALIDACIÓN en las tres variantes ===
    plt.figure(figsize=(8, 4.5))
    plt.plot(curvas["baseline"], color="blue", label="Baseline (sin augmentation)")
    plt.plot(curvas["augmented"], color="green", linestyle="--", label="Augmented (flip=0.5)")
    plt.plot(curvas["augmented_sin_flip"], color="red", linestyle=":", label="Augmented sin flip (flip=0.0)")
    plt.title("Accuracy en validación: baseline vs augmentation con/sin flip (dataset completo)", fontweight="bold")
    plt.xlabel("Épocas")
    plt.ylabel("Accuracy en validación")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "learning_curve.png", dpi=150)
    plt.close()

    print("\n=== Resumen ===")
    for nombre, _, _ in variantes:
        print(f"{nombre:20s} {resultados[nombre]['accuracy_test']:.4f} "
              f"({resultados[nombre]['epochs_entrenadas']} épocas)")
    print(f"Resultados guardados en {RESULTS_DIR}")


if __name__ == "__main__":
    main()
