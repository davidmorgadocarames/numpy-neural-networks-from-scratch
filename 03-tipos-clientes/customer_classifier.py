"""
Clasificador de tipos de clientes de una tienda online (NumPy puro) a partir de 2 variables:
minutos navegando y productos en el carrito. Red modular (Densa -> LeakyReLU -> Densa ->
Softmax) entrenada con entropía cruzada para separar 3 categorías: Navegadores, Ocasionales
y VIPs.

Split en TRES partes -- 60% train / 20% validación / 20% test, estratificado por clase -- para
poder decidir cuándo parar mirando el error de VALIDACIÓN (early stopping) sin tocar el
conjunto de test hasta la evaluación final, igual que en 04-prediccion-temperatura-dia-noche y
05-precio-casas. Antes solo había train/test: la curva de "test" se dibujaba en cada época sin
que nada dependiera de ella, pero no había ninguna señal honesta para decidir cuándo parar de
entrenar sin espiar el propio test.

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

SEED_DATOS = 42  # gobierna solo la generación de datos -- fijo siempre, nunca es argumento
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
NOMBRES_CLASES = ["Navegadores", "Ocasionales", "VIPs"]
COLORES_CLASES = ["red", "green", "blue"]

# Early stopping: para el entrenamiento en cuanto el error de VALIDACIÓN deja de mejorar de
# verdad (ver README de 04/05 para la explicación completa).
PACIENCIA_EARLY_STOP = 200
MEJORA_MINIMA_RELATIVA = 0.005


def main(seed_split=42, seed_modelo=42, quiet=False, guardar_graficas=True) -> dict:
    """seed_split y seed_modelo son independientes entre sí y de SEED_DATOS -- ver README para
    el análisis de robustez sobre múltiples semillas."""
    # Datos: SIEMPRE con SEED_DATOS, vía la API legacy de np.random -- así los 120 clientes no
    # cambian nunca, sea cual sea seed_split/seed_modelo.
    np.random.seed(SEED_DATOS)

    # 3 categorías de clientes, 40 por categoría (120 en total) -- más muestra que el
    # original (20/categoría) para poder permitirse un split de 3 partes decente.
    X0 = np.random.uniform(2, 10, (40, 2)) + np.array([0, 0])
    X1 = np.random.uniform(15, 25, (40, 2)) + np.array([0, 2])
    X2 = np.random.uniform(30, 45, (40, 2)) + np.array([0, 6])
    X = np.vstack([X0, X1, X2])

    Y_num = np.concatenate([np.zeros(40), np.ones(40), np.full(40, 2)]).astype(int)
    Y = np.zeros((120, 3))
    Y[np.arange(120), Y_num] = 1

    # Split train/val/test estratificado por clase: 24 train (60%) + 8 val (20%) + 8 test (20%)
    # por categoría.
    indices_train, indices_val, indices_test = [], [], []
    rng = np.random.default_rng(seed_split)
    N_TRAIN_CLASE, N_VAL_CLASE = 24, 8
    for clase in range(3):
        idx_clase = np.where(Y_num == clase)[0]
        rng.shuffle(idx_clase)
        indices_train.extend(idx_clase[:N_TRAIN_CLASE])
        indices_val.extend(idx_clase[N_TRAIN_CLASE : N_TRAIN_CLASE + N_VAL_CLASE])
        indices_test.extend(idx_clase[N_TRAIN_CLASE + N_VAL_CLASE :])
    indices_train = np.array(indices_train)
    indices_val = np.array(indices_val)
    indices_test = np.array(indices_test)

    X_train_raw, X_val_raw, X_test_raw = X[indices_train], X[indices_val], X[indices_test]
    Y_train, Y_val, Y_test = Y[indices_train], Y[indices_val], Y[indices_test]
    Y_num_test = Y_num[indices_test]

    # Normalización min-max (con min/max de TRAIN, nunca de val/test) -- sin esto, minutos y
    # productos llegan a la red en escalas de hasta 50, lo que en la práctica frena tanto el
    # descenso de gradiente que a las 3000 épocas la red seguía sin converger del todo y
    # confundía "Ocasionales" con "VIPs" aunque las categorías no se solapan en ninguna de las
    # 2 variables. Con los mismos hiperparámetros pero datos normalizados, converge a 100%.
    X_min, X_max = X_train_raw.min(axis=0), X_train_raw.max(axis=0)
    X_train = (X_train_raw - X_min) / (X_max - X_min)
    X_val = (X_val_raw - X_min) / (X_max - X_min)
    X_test = (X_test_raw - X_min) / (X_max - X_min)

    rng_modelo = np.random.default_rng(seed_modelo)
    red = [
        CapaDensa(dim_entrada=2, dim_salida=5, semilla_he=False, rng=rng_modelo),
        ActivacionLeakyReLU(),
        CapaDensa(dim_entrada=5, dim_salida=3, semilla_he=False, rng=rng_modelo),
        ActivacionSoftmax(),
    ]

    learning_rate = 0.01
    epochs = 3000
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
    accuracy_test = float(np.mean(pred_test == Y_num_test))
    loss_test = float(-np.mean(np.sum(Y_test * np.log(A_test + 1e-15), axis=1)))

    matriz_confusion = np.zeros((3, 3), dtype=int)
    for real, pred in zip(Y_num_test, pred_test):
        matriz_confusion[real, pred] += 1

    metrics = {
        "seed_datos": SEED_DATOS,
        "seed_split": seed_split,
        "seed_modelo": seed_modelo,
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
        "clases": NOMBRES_CLASES,
    }

    if not guardar_graficas:
        return {**metrics, "historial_loss_val": historial_loss_val}

    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if not quiet:
        print(f"Accuracy test: {accuracy_test:.4f} ({len(X_test)} clientes de test)")

    # === Gráfico 1: curva de aprendizaje (train + validación, que es lo que decide cuándo
    # parar) ===
    plt.figure(figsize=(6, 4))
    plt.plot(historial_loss_train, color="purple", label="Train")
    plt.plot(historial_loss_val, color="orange", linestyle="--", label="Validación")
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
    for clase, nombre in enumerate(NOMBRES_CLASES):
        mask_train = Y_num[indices_train] == clase
        plt.scatter(X_train_raw[mask_train, 0], X_train_raw[mask_train, 1],
                    color=COLORES_CLASES[clase], label=f"{nombre} (train)", edgecolors="k")
    plt.scatter(X_val_raw[:, 0], X_val_raw[:, 1],
                c=[COLORES_CLASES[c] for c in Y_num[indices_val]],
                marker="^", s=70, edgecolors="k", label="Validación")
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

    if not quiet:
        print(f"Resultados guardados en {RESULTS_DIR}")

    return {**metrics, "historial_loss_val": historial_loss_val}


if __name__ == "__main__":
    main()
