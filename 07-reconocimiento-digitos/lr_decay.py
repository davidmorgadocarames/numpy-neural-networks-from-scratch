"""
Learning rate decay sobre la configuración de referencia (Adam, mini-batch, MNIST completo) --
ver README raíz para por qué esta es la configuración de referencia para todo lo nuevo del
repo, en vez de repetir la comparación SGD-vs-Adam otra vez.

Compara Adam con learning_rate CONSTANTE (el mismo 0.001 ya documentado en
sgd_vs_adam_full.py) frente a Adam con decaimiento exponencial del learning_rate época a
época: `lr(epoch) = LEARNING_RATE_ADAM * DECAY_RATE ** epoch`. Reutiliza `crear_red()` de
sgd_vs_adam_full.py (misma arquitectura, mismo Adam) sin tocar ese script -- solo cambia qué
learning_rate se le pasa a `capa.backward()` en cada época, ahora vía una función en vez de un
escalar fijo.

No toca ni reentrena digit_classifier_full.py ni sgd_vs_adam_full.py. Una sola ejecución
verificada (no barrido de N=20 semillas -- esto es una demostración de la técnica, no una
comparación metodológica).

Uso: python lr_decay.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from capas import CapaDensa, predecir

from digit_classifier_full import (
    BATCH_SIZE,
    EPOCHS_MAX,
    MEJORA_MINIMA_RELATIVA,
    PACIENCIA_EARLY_STOP,
    cargar_datos,
    generar_batches,
)
from sgd_vs_adam_full import LEARNING_RATE_ADAM, OptimizadorAdam, crear_red

RESULTS_DIR = Path(__file__).parent / "results_lr_decay"
RESULTS_DIR.mkdir(exist_ok=True)

DECAY_RATE = 0.9  # calibrado a mano: tras 10 épocas, lr *= 0.9**10 ~= 0.35 del valor inicial

VARIANTES = [
    ("adam_sin_decay", lambda epoch: LEARNING_RATE_ADAM),
    ("adam_con_decay", lambda epoch: LEARNING_RATE_ADAM * DECAY_RATE ** epoch),
]
COLORES_VARIANTES = {"adam_sin_decay": "#4C72B0", "adam_con_decay": "#55A868"}


def entrenar(nombre, red, X_train, Y_train, X_val, Y_val, seed_modelo, lr_schedule, quiet=False):
    historial_loss_train, historial_loss_val, historial_lr = [], [], []
    mejor_loss_val = np.inf
    mejor_epoca = None
    mejores_pesos = None
    rng_batches = np.random.default_rng(seed_modelo)

    for epoch in range(EPOCHS_MAX):
        learning_rate = lr_schedule(epoch)
        historial_lr.append(learning_rate)

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
                    print(f"  [{nombre}] early stopping en la época {epoch + 1} (lr={learning_rate:.6f})")
                break
    else:
        if not quiet:
            print(f"  [{nombre}] completó las {EPOCHS_MAX} épocas sin activar el early stopping")

    for capa, (W, b) in zip([c for c in red if isinstance(c, CapaDensa)], mejores_pesos):
        capa.W, capa.b = W, b
    if not quiet:
        print(f"  [{nombre}] pesos restaurados al mínimo de validación: época {mejor_epoca + 1} "
              f"(loss_val={mejor_loss_val:.6f}), frente a la época de corte {len(historial_loss_train)}")

    return historial_loss_train, historial_loss_val, historial_lr, mejor_epoca, mejor_loss_val


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


def graficar_curvas(curvas_val, curvas_lr, ruta_salida):
    """Dos paneles: pérdida de validación (arriba) y learning_rate efectivo (abajo), mismo eje
    X (épocas), para poder leer directamente qué le pasa a la pérdida cuando el lr decae."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True)
    for nombre, historial in curvas_val.items():
        ax1.plot(historial, color=COLORES_VARIANTES[nombre], label=nombre)
    ax1.set_yscale("log")
    ax1.set_ylabel("Pérdida de validación (escala log)")
    ax1.set_title("LR decay vs LR constante (Adam, mini-batch)", fontweight="bold")
    ax1.legend()
    ax1.grid(True, which="both", linestyle="--", alpha=0.4)

    for nombre, historial in curvas_lr.items():
        ax2.plot(historial, color=COLORES_VARIANTES[nombre], label=nombre)
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Learning rate efectivo")
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150)
    plt.close()


def main(seed_split=42, seed_modelo=42, quiet=False, guardar_graficas=True) -> dict:
    X_train, Y_train, Y_train_num, X_val, Y_val, Y_val_num, X_test, Y_test_num = cargar_datos(seed_split, quiet=quiet)

    resultados = {}
    curvas_val = {}
    curvas_lr = {}
    for nombre, lr_schedule in VARIANTES:
        if not quiet:
            print(f"\n=== Entrenando '{nombre}' ===")
        rng_modelo = np.random.default_rng(seed_modelo)
        red = crear_red(rng_modelo, OptimizadorAdam)
        hist_train, hist_val, hist_lr, mejor_epoca, mejor_loss_val = entrenar(
            nombre, red, X_train, Y_train, X_val, Y_val, seed_modelo, lr_schedule, quiet=quiet
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
            graficar_confusion(matriz_confusion, accuracy_test, f"Matriz de confusión test — {nombre}",
                                RESULTS_DIR / f"confusion_matrix_{nombre}.png")

        resultados[nombre] = {
            "epochs_entrenadas": len(hist_train),
            "epoca_mejor_val": mejor_epoca + 1,
            "loss_train_final": float(hist_train[mejor_epoca]),
            "loss_val_final": float(mejor_loss_val),
            "accuracy_test": accuracy_test,
            "matriz_confusion": matriz_confusion.tolist(),
            "historial_loss_val": hist_val,
            "historial_lr": hist_lr,
        }
        curvas_val[nombre] = hist_val
        curvas_lr[nombre] = hist_lr

    metrics = {
        "seed_split": seed_split,
        "seed_modelo": seed_modelo,
        "decay_rate": DECAY_RATE,
        "learning_rate_inicial": LEARNING_RATE_ADAM,
        "resultados": resultados,
    }

    if not guardar_graficas:
        return metrics

    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    graficar_curvas(curvas_val, curvas_lr, RESULTS_DIR / "lr_decay_comparativa.png")

    if not quiet:
        print("\n--- Resumen ---")
        for nombre, datos in resultados.items():
            print(f"{nombre:16s} accuracy={datos['accuracy_test']:.4f}  "
                  f"epocas={datos['epochs_entrenadas']}  epoca_mejor_val={datos['epoca_mejor_val']}")
        print(f"Resultados guardados en {RESULTS_DIR}")

    return metrics


if __name__ == "__main__":
    main()
