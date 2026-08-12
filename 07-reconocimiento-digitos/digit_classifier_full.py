"""
Variante de `digit_classifier.py` sobre el dataset MNIST COMPLETO (70.000 imágenes, no la
muestra reducida de 1.200/300/300), para poder comparar en el propio repositorio qué gana la
red al entrenar con mucha más muestra. Se guarda como script aparte -- no sustituye al
original -- para que ambos resultados queden documentados uno junto al otro.

Con 1.200 imágenes, un batch "full" (toda la muestra de golpe, 1 actualización de pesos por
época) era barato de calcular. Con ~42.000 imágenes de train, seguir haciendo full-batch
significaría 1 sola actualización de pesos por época sobre una matriz enorme -- converge muy
despacio en número de actualizaciones, aunque cada una sea más precisa. La solución estándar es
entrenar por MINI-BATCHES: en cada época, la muestra se recorre en trozos de `BATCH_SIZE`
imágenes, con una actualización de pesos por trozo (~1.312 actualizaciones por época con
batch_size=32 sobre 42.000 imágenes) en vez de 1 sola.

Split estratificado por dígito, pero usando el 60/20/20 de CADA clase (no un número fijo por
clase) para aprovechar las 70.000 imágenes completas -- MNIST no tiene exactamente el mismo
número de ejemplos por dígito (entre 6.313 y 7.877), así que el split resultante no es
perfectamente equilibrado entre clases, a diferencia de la muestra reducida del script
original.

Error clásico a evitar al pasar de full-batch a mini-batch: la normalización del gradiente de
la capa de salida debe dividir por el tamaño del BATCH actual (`Yb.shape[0]`), no por el
tamaño de todo el conjunto de train (`Y_train.shape[0]`). Si se deja el denominador del
full-batch original, el gradiente queda ~1.312 veces más pequeño de lo que debería en cada
batch (o al revés si se invierte el error), y el entrenamiento se vuelve absurdamente lento o
inestable sin que salte ningún error explícito -- un bug silencioso.

Uso: python digit_classifier_full.py
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
RESULTS_DIR = Path(__file__).parent / "results_full"
RESULTS_DIR.mkdir(exist_ok=True)
FRAC_TRAIN, FRAC_VAL = 0.6, 0.2  # el resto (0.2) es test
BATCH_SIZE = 32

# Con ~1.312 actualizaciones de pesos por época (en vez de 1 en la versión full-batch), la red
# converge en muchas menos épocas -- por eso el techo y la paciencia son mucho más bajos que en
# digit_classifier.py (epochs=800, paciencia=200). Ver README para la comparación medida.
EPOCHS_MAX = 30
PACIENCIA_EARLY_STOP = 5
MEJORA_MINIMA_RELATIVA = 0.005


def generar_batches(X, Y, batch_size, rng):
    """Recorre (X, Y) en mini-batches de tamaño `batch_size`, en un orden aleatorio distinto
    cada vez que se llama (una época = una llamada). El último batch de la época puede ser más
    pequeño que `batch_size` si el total no es múltiplo exacto -- por eso el gradiente se
    normaliza por el tamaño real de CADA batch, nunca por `batch_size` a secas."""
    n = X.shape[0]
    indices = rng.permutation(n)
    for inicio in range(0, n, batch_size):
        idx_batch = indices[inicio : inicio + batch_size]
        yield X[idx_batch], Y[idx_batch]


def main() -> None:
    np.random.seed(SEED)

    print("Descargando MNIST (sklearn fetch_openml)...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
    X_puro, Y_puro = mnist.data, mnist.target.astype(int)

    # Split estratificado 60/20/20 usando TODOS los ejemplos de cada dígito (no un número fijo
    # igual por clase, para no descartar imágenes de las clases con más muestra).
    rng = np.random.default_rng(SEED)
    indices_train, indices_val, indices_test = [], [], []
    for digito in range(10):
        idx_digito = np.where(Y_puro == digito)[0]
        rng.shuffle(idx_digito)
        n_digito = len(idx_digito)
        corte_train = int(n_digito * FRAC_TRAIN)
        corte_val = corte_train + int(n_digito * FRAC_VAL)
        indices_train.extend(idx_digito[:corte_train])
        indices_val.extend(idx_digito[corte_train:corte_val])
        indices_test.extend(idx_digito[corte_val:])
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

    red = [
        CapaDensa(dim_entrada=784, dim_salida=128),
        ActivacionLeakyReLU(),
        CapaDensa(dim_entrada=128, dim_salida=10),
        ActivacionSoftmax(),
    ]

    learning_rate = 0.1
    historial_loss_train, historial_loss_val = [], []

    # Checkpoint del mejor punto de validación (ver README raíz, "Checkpoint del mejor punto de
    # validación") -- igual que en el resto del repo, con .copy() para no guardar referencias.
    mejor_loss_val = np.inf
    mejor_epoca = None
    mejores_pesos = None

    rng_batches = np.random.default_rng(SEED)
    n_batches_por_epoca = int(np.ceil(len(X_train) / BATCH_SIZE))

    print(f"Entrenando con {len(X_train)} imágenes ({n_batches_por_epoca} batches/época de "
          f"{BATCH_SIZE}), validando sobre {len(X_val)}, test reservado con {len(X_test)}...")
    for epoch in range(EPOCHS_MAX):
        loss_train_acumulada = 0.0
        n_vistas = 0
        for Xb, Yb in generar_batches(X_train, Y_train, BATCH_SIZE, rng_batches):
            activacion = Xb
            for capa in red:
                activacion = capa.forward(activacion)
            Ab = activacion

            loss_batch = -np.mean(np.sum(Yb * np.log(Ab + 1e-15), axis=1))
            loss_train_acumulada += loss_batch * Xb.shape[0]
            n_vistas += Xb.shape[0]

            # Normalizado por el tamaño del BATCH actual (Yb), no por el de todo train -- ver
            # docstring del módulo para el error clásico que esto evita.
            gradiente = (Ab - Yb) / Yb.shape[0]
            for capa in reversed(red[:-1]):
                gradiente = capa.backward(gradiente, learning_rate)

        loss_train = loss_train_acumulada / n_vistas
        historial_loss_train.append(loss_train)

        A_val = predecir(red, X_val)
        loss_val = -np.mean(np.sum(Y_val * np.log(A_val + 1e-15), axis=1))
        historial_loss_val.append(loss_val)

        if loss_val < mejor_loss_val:
            mejor_loss_val = loss_val
            mejor_epoca = epoch
            mejores_pesos = [(c.W.copy(), c.b.copy()) for c in red if isinstance(c, CapaDensa)]

        print(f"Época {epoch}/{EPOCHS_MAX} - loss train: {loss_train:.4f} - loss val: {loss_val:.4f}")

        if epoch >= PACIENCIA_EARLY_STOP:
            loss_val_referencia = historial_loss_val[epoch - PACIENCIA_EARLY_STOP]
            mejora_relativa = (loss_val_referencia - loss_val) / loss_val_referencia
            if mejora_relativa < MEJORA_MINIMA_RELATIVA:
                print(f"Early stopping en la época {epoch + 1}: el error de validación lleva "
                      f"{PACIENCIA_EARLY_STOP} épocas sin mejorar un {MEJORA_MINIMA_RELATIVA:.1%}")
                break
    else:
        print(f"Entrenamiento completado sin activar el early stopping (llegó a la época {EPOCHS_MAX})")

    epocas_entrenadas = len(historial_loss_train)

    for capa, (W, b) in zip([c for c in red if isinstance(c, CapaDensa)], mejores_pesos):
        capa.W, capa.b = W, b
    print(f"Pesos restaurados al mínimo de validación: época {mejor_epoca + 1} "
          f"(loss_val={mejor_loss_val:.6f}), frente a la época de corte {epocas_entrenadas}")

    # === Única vez que se toca el conjunto de test, ya con la red entrenada ===
    A_test = predecir(red, X_test)
    pred_test = np.argmax(A_test, axis=1)
    accuracy_test = float(np.mean(pred_test == Y_test_num))

    matriz_confusion = np.zeros((10, 10), dtype=int)
    for real, pred in zip(Y_test_num, pred_test):
        matriz_confusion[real, pred] += 1

    metrics = {
        "batch_size": BATCH_SIZE,
        "n_batches_por_epoca": n_batches_por_epoca,
        "epochs_configuradas": EPOCHS_MAX,
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
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    np.savez(
        RESULTS_DIR / "red_pesos.npz",
        W1=red[0].W, b1=red[0].b, W2=red[2].W, b2=red[2].b,
    )

    print(f"Accuracy en test: {accuracy_test:.4f} ({len(X_test)} imágenes)")

    # === Gráfico 1: curva de aprendizaje ===
    plt.figure(figsize=(6, 4))
    plt.plot(historial_loss_train, color="blue", label="Train")
    plt.plot(historial_loss_val, color="orange", linestyle="--", label="Validación")
    plt.title("Curva de aprendizaje: reconocimiento de dígitos (dataset completo, mini-batch)", fontweight="bold")
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
    plt.suptitle("Muestra del dataset de entrenamiento (MNIST completo)", fontweight="bold")
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
