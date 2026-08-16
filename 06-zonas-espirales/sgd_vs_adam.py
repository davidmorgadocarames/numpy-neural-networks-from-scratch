"""
SGD vs Adam sobre el mismo problema de espirales -- ¿cambiaría mucho el pipeline usar un
optimizador con memoria en vez de descenso de gradiente puro? Mismos datos, misma arquitectura
(2 -> 64 -> 64 -> 3), mismo split, mismos pesos iniciales en ambas variantes -- la única
diferencia es el optimizador de cada CapaDensa (ver `OptimizadorSGD`/`OptimizadorAdam` en
capas.py) y su learning_rate propio (Adam necesita uno mucho más pequeño que SGD: sus pasos ya
vienen normalizados por la varianza del gradiente, así que el mismo 0.2 que usa SGD aquí
haría explotar el entrenamiento).

No toca ni sustituye a spiral_classifier.py -- ese script documenta un hallazgo metodológico
propio (early stopping que no detecta el sobreajuste con solo 30 puntos de validación) y sus
resultados (accuracy 97.78%, resultados/) se quedan como están. Este script reutiliza su
generación de datos y su split (generar_datos, split_estratificado) para no duplicar el
generador de espirales atribuido a CS231n, pero guarda sus propios resultados en
results_sgd_vs_adam/ para no pisar los del proyecto canónico.

Uso: python sgd_vs_adam.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from capas import ActivacionLeakyReLU, ActivacionSoftmax, CapaDensa, OptimizadorAdam, OptimizadorSGD, predecir

from spiral_classifier import (
    MEJORA_MINIMA_RELATIVA,
    PACIENCIA_EARLY_STOP,
    generar_datos,
    split_estratificado,
)

RESULTS_DIR = Path(__file__).parent / "results_sgd_vs_adam"
RESULTS_DIR.mkdir(exist_ok=True)

EPOCHS = 5000  # mismo techo de seguridad que el proyecto canónico
LEARNING_RATE_SGD = 0.2  # igual que spiral_classifier.py, para una comparación justa
LEARNING_RATE_ADAM = 0.01  # calibrado a mano: 0.001 converge de sobra, 0.05 empieza a oscilar

VARIANTES = [
    ("sgd", OptimizadorSGD, LEARNING_RATE_SGD),
    ("adam", OptimizadorAdam, LEARNING_RATE_ADAM),
]
COLORES_VARIANTES = {"sgd": "#4C72B0", "adam": "#55A868"}


def crear_red(rng_modelo, clase_optimizador):
    """Misma arquitectura y mismos pesos iniciales en las dos variantes -- se llama con un
    rng_modelo fresco (recién creado a partir de seed_modelo) cada vez, así que cualquier
    diferencia final entre sgd y adam se debe al optimizador, no a la inicialización. Cada
    CapaDensa lleva su PROPIA instancia del optimizador (Adam necesita estado -- m, v -- por
    capa, no puede compartirse entre las tres)."""
    return [
        CapaDensa(2, 64, rng=rng_modelo, optimizador=clase_optimizador()),
        ActivacionLeakyReLU(),
        CapaDensa(64, 64, rng=rng_modelo, optimizador=clase_optimizador()),
        ActivacionLeakyReLU(),
        CapaDensa(64, 3, rng=rng_modelo, optimizador=clase_optimizador()),
        ActivacionSoftmax(),
    ]


def entrenar(nombre, red, X_train, Y_train, X_val, Y_val, learning_rate, quiet=False):
    """Mismo bucle full-batch con early stopping y checkpoint del mínimo de validación que
    spiral_classifier.py (ver su README para la explicación completa) -- aquí parametrizado en
    learning_rate para poder usar uno distinto por variante."""
    historial_loss_train, historial_loss_val = [], []
    mejor_loss_val = np.inf
    mejor_epoca = None
    mejores_pesos = None

    for epoch in range(EPOCHS):
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
                    print(f"  [{nombre}] early stopping en la época {epoch + 1}")
                break
    else:
        if not quiet:
            print(f"  [{nombre}] completó las {EPOCHS} épocas sin activar el early stopping")

    for capa, (W, b) in zip([c for c in red if isinstance(c, CapaDensa)], mejores_pesos):
        capa.W, capa.b = W, b
    if not quiet:
        print(f"  [{nombre}] pesos restaurados al mínimo de validación: época {mejor_epoca + 1} "
              f"(loss_val={mejor_loss_val:.6f}), frente a la época de corte {len(historial_loss_train)}")

    return historial_loss_train, historial_loss_val, mejor_epoca, mejor_loss_val


def graficar_frontera(red, X_train, Y_num, indices_train, X_val, indices_val, X_test, Y_num_test,
                       titulo, ruta_salida):
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
    plt.title(titulo, fontweight="bold")
    plt.xlabel("Eje X")
    plt.ylabel("Eje Y")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150)
    plt.close()


def graficar_confusion(matriz, accuracy, titulo, ruta):
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(matriz, cmap="Blues")
    ax.set_title(f"{titulo} (accuracy={accuracy:.2%})", fontweight="bold")
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(["Brazo 0", "Brazo 1", "Brazo 2"])
    ax.set_yticklabels(["Brazo 0", "Brazo 1", "Brazo 2"])
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(matriz[i, j]), ha="center", va="center",
                     color="white" if matriz[i, j] > 5 else "black", fontsize=12)
    fig.colorbar(im, fraction=0.046)
    plt.tight_layout()
    plt.savefig(ruta, dpi=150)
    plt.close()


def graficar_curva_comparativa(curvas_val, ruta_salida):
    """Pérdida de VALIDACIÓN por época, sgd vs adam superpuestas -- la pregunta concreta de
    "cambiaría mucho el pipeline" es sobre todo esta: ¿converge Adam en muchas menos épocas?"""
    plt.figure(figsize=(8, 4.5))
    for nombre, historial in curvas_val.items():
        plt.plot(historial, color=COLORES_VARIANTES[nombre], label=nombre.upper())
    plt.yscale("log")
    plt.title("SGD vs Adam: pérdida de validación por época", fontweight="bold")
    plt.xlabel("Época")
    plt.ylabel("Pérdida de validación (escala log)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150)
    plt.close()


def main(seed_split=0, seed_modelo=0, quiet=False, guardar_graficas=True) -> dict:
    """seed_split y seed_modelo son independientes entre sí y de SEED_DATOS, igual que en
    spiral_classifier.py -- ver su README para el análisis de robustez sobre múltiples
    semillas y la explicación de por qué SEED_DATOS se queda fija."""
    X, Y, Y_num = generar_datos()
    indices_train, indices_val, indices_test = split_estratificado(Y_num, seed_split)

    X_train, X_val, X_test = X[indices_train], X[indices_val], X[indices_test]
    Y_train, Y_val, Y_test = Y[indices_train], Y[indices_val], Y[indices_test]
    Y_num_test = Y_num[indices_test]

    resultados = {}
    curvas_val = {}
    for nombre, clase_optimizador, learning_rate in VARIANTES:
        if not quiet:
            print(f"\n=== Entrenando '{nombre}' (learning_rate={learning_rate}) ===")
        rng_modelo = np.random.default_rng(seed_modelo)  # mismos pesos iniciales en ambas variantes
        red = crear_red(rng_modelo, clase_optimizador)
        hist_train, hist_val, mejor_epoca, mejor_loss_val = entrenar(
            nombre, red, X_train, Y_train, X_val, Y_val, learning_rate, quiet=quiet
        )

        A_test = predecir(red, X_test)
        pred_test = np.argmax(A_test, axis=1)
        accuracy_test = float(np.mean(pred_test == Y_num_test))
        matriz_confusion = np.zeros((3, 3), dtype=int)
        for real, pred in zip(Y_num_test, pred_test):
            matriz_confusion[real, pred] += 1
        if not quiet:
            print(f"  [{nombre}] accuracy en test: {accuracy_test:.4f}")

        if guardar_graficas:
            graficar_confusion(matriz_confusion, accuracy_test, f"Matriz de confusión test — {nombre.upper()}",
                                RESULTS_DIR / f"confusion_matrix_{nombre}.png")
            graficar_frontera(red, X_train, Y_num, indices_train, X_val, indices_val, X_test, Y_num_test,
                               f"Zonas de espirales — {nombre.upper()} (accuracy={accuracy_test:.2%})",
                               RESULTS_DIR / f"zonas_{nombre}.png")

        resultados[nombre] = {
            "learning_rate": learning_rate,
            "epochs_entrenadas": len(hist_train),
            "epoca_mejor_val": mejor_epoca + 1,
            "loss_train_final": float(hist_train[mejor_epoca]),
            "loss_val_final": float(mejor_loss_val),
            "accuracy_test": accuracy_test,
            "matriz_confusion": matriz_confusion.tolist(),
            "historial_loss_val": hist_val,
        }
        curvas_val[nombre] = hist_val

    metrics = {
        "seed_datos_ver": "spiral_classifier.SEED_DATOS",
        "seed_split": seed_split,
        "seed_modelo": seed_modelo,
        "epochs_max_configuradas": EPOCHS,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "resultados": resultados,
    }

    if not guardar_graficas:
        return metrics

    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    graficar_curva_comparativa(curvas_val, RESULTS_DIR / "learning_curve_comparativa.png")

    if not quiet:
        print("\n--- Resumen ---")
        for nombre, datos in resultados.items():
            print(f"{nombre:6s} accuracy={datos['accuracy_test']:.4f}  "
                  f"epocas={datos['epochs_entrenadas']}  epoca_mejor_val={datos['epoca_mejor_val']}")
        print(f"Resultados guardados en {RESULTS_DIR}")

    return metrics


if __name__ == "__main__":
    main()
