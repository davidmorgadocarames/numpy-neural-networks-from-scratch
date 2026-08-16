"""
SGD vs Adam sobre MNIST completo, mini-batch (misma escala que digit_classifier_full.py:
~42.000 train, batch_size=32). No toca digit_classifier_full.py -- reutiliza su cargar_datos()
y guarda sus propios resultados en results_full_sgd_vs_adam/. Mismo patrón que
06-zonas-espirales/sgd_vs_adam.py y 07-reconocimiento-digitos/sgd_vs_adam.py.

Uso: python sgd_vs_adam_full.py
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

from digit_classifier_full import (
    BATCH_SIZE,
    EPOCHS_MAX,
    MEJORA_MINIMA_RELATIVA,
    PACIENCIA_EARLY_STOP,
    cargar_datos,
    generar_batches,
)

RESULTS_DIR = Path(__file__).parent / "results_full_sgd_vs_adam"
RESULTS_DIR.mkdir(exist_ok=True)

LEARNING_RATE_SGD = 0.1  # igual que digit_classifier_full.py
LEARNING_RATE_ADAM = 0.001  # calibrado a mano, igual que la variante full-batch de 07

VARIANTES = [
    ("sgd", OptimizadorSGD, LEARNING_RATE_SGD),
    ("adam", OptimizadorAdam, LEARNING_RATE_ADAM),
]
COLORES_VARIANTES = {"sgd": "#4C72B0", "adam": "#55A868"}


def crear_red(rng_modelo, clase_optimizador):
    return [
        CapaDensa(dim_entrada=784, dim_salida=128, rng=rng_modelo, optimizador=clase_optimizador()),
        ActivacionLeakyReLU(),
        CapaDensa(dim_entrada=128, dim_salida=10, rng=rng_modelo, optimizador=clase_optimizador()),
        ActivacionSoftmax(),
    ]


def entrenar(nombre, red, X_train, Y_train, X_val, Y_val, seed_modelo, learning_rate, quiet=False):
    historial_loss_train, historial_loss_val = [], []
    mejor_loss_val = np.inf
    mejor_epoca = None
    mejores_pesos = None
    rng_batches = np.random.default_rng(seed_modelo)

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

        if epoch >= PACIENCIA_EARLY_STOP:
            loss_val_referencia = historial_loss_val[epoch - PACIENCIA_EARLY_STOP]
            mejora_relativa = (loss_val_referencia - loss_val) / loss_val_referencia
            if mejora_relativa < MEJORA_MINIMA_RELATIVA:
                if not quiet:
                    print(f"  [{nombre}] early stopping en la época {epoch + 1}")
                break
    else:
        if not quiet:
            print(f"  [{nombre}] completó las {EPOCHS_MAX} épocas sin activar el early stopping")

    for capa, (W, b) in zip([c for c in red if isinstance(c, CapaDensa)], mejores_pesos):
        capa.W, capa.b = W, b
    if not quiet:
        print(f"  [{nombre}] pesos restaurados al mínimo de validación: época {mejor_epoca + 1} "
              f"(loss_val={mejor_loss_val:.6f}), frente a la época de corte {len(historial_loss_train)}")

    return historial_loss_train, historial_loss_val, mejor_epoca, mejor_loss_val


def graficar_confusion(matriz, accuracy, titulo, ruta):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matriz, cmap="Blues")
    ax.set_title(f"{titulo} (accuracy={accuracy:.2%})", fontweight="bold")
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    for i in range(10):
        for j in range(10):
            valor = matriz[i, j]
            if valor > 0:
                ax.text(j, i, str(valor), ha="center", va="center",
                         color="white" if valor > matriz.max() / 2 else "black", fontsize=8)
    fig.colorbar(im, fraction=0.046)
    plt.tight_layout()
    plt.savefig(ruta, dpi=150)
    plt.close()


def graficar_curva_comparativa(curvas_val, ruta_salida):
    plt.figure(figsize=(8, 4.5))
    for nombre, historial in curvas_val.items():
        plt.plot(historial, color=COLORES_VARIANTES[nombre], label=nombre.upper())
    plt.yscale("log")
    plt.title("SGD vs Adam: pérdida de validación por época (MNIST completo, mini-batch)", fontweight="bold")
    plt.xlabel("Época")
    plt.ylabel("Pérdida de validación (escala log)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150)
    plt.close()


def main(seed_split=42, seed_modelo=42, quiet=False, guardar_graficas=True) -> dict:
    X_train, Y_train, Y_train_num, X_val, Y_val, Y_val_num, X_test, Y_test_num = cargar_datos(seed_split, quiet=quiet)

    resultados = {}
    curvas_val = {}
    for nombre, clase_optimizador, learning_rate in VARIANTES:
        if not quiet:
            print(f"\n=== Entrenando '{nombre}' (learning_rate={learning_rate}) ===")
        rng_modelo = np.random.default_rng(seed_modelo)
        red = crear_red(rng_modelo, clase_optimizador)
        hist_train, hist_val, mejor_epoca, mejor_loss_val = entrenar(
            nombre, red, X_train, Y_train, X_val, Y_val, seed_modelo, learning_rate, quiet=quiet
        )

        A_test = predecir(red, X_test)
        pred_test = np.argmax(A_test, axis=1)
        accuracy_test = float(np.mean(pred_test == Y_test_num))
        matriz_confusion = np.zeros((10, 10), dtype=int)
        for real, pred in zip(Y_test_num, pred_test):
            matriz_confusion[real, pred] += 1
        if not quiet:
            print(f"  [{nombre}] accuracy en test: {accuracy_test:.4f}")

        if guardar_graficas:
            graficar_confusion(matriz_confusion, accuracy_test, f"Matriz de confusión test — {nombre.upper()}",
                                RESULTS_DIR / f"confusion_matrix_{nombre}.png")

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
        "seed_split": seed_split,
        "seed_modelo": seed_modelo,
        "batch_size": BATCH_SIZE,
        "epochs_max_configuradas": EPOCHS_MAX,
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
