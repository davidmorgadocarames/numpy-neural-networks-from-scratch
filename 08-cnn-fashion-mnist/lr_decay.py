"""
Learning rate decay sobre la configuración de referencia (Adam, mini-batch, CNN Fashion-MNIST
completo) -- ver README raíz para por qué esta es la configuración de referencia para todo lo
nuevo del repo. Usa solo la variante `baseline` (sin augmentation) para no volver a expandir en
las 3 variantes -- esto es una demostración de la técnica, no una repetición de la comparación
de augmentation ya documentada.

Compara Adam con learning_rate CONSTANTE (el mismo 0.001 ya documentado en
sgd_vs_adam_full.py) frente a Adam con decaimiento exponencial:
`lr(epoch) = LEARNING_RATE_ADAM * DECAY_RATE ** epoch`. Reutiliza `crear_red()` de
sgd_vs_adam_full.py sin tocar ese script.

Uso: python lr_decay.py
"""

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from capas import CapaDensa, OptimizadorAdam
from capas_cnn import CapaConv2D, predecir_cnn

from cnn_fashion_mnist_full import (
    BATCH_SIZE,
    EPOCHS_MAX,
    MEJORA_MINIMA_RELATIVA,
    PACIENCIA_EARLY_STOP,
    cargar_datos,
    evaluar,
    generar_batches,
    graficar_confusion,
)
from sgd_vs_adam_full import LEARNING_RATE_ADAM, crear_red

RESULTS_DIR = Path(__file__).parent / "results_lr_decay"
RESULTS_DIR.mkdir(exist_ok=True)

DECAY_RATE = 0.9  # mismo valor que 07/lr_decay.py, para que las dos demostraciones sean comparables

VARIANTES = [
    ("adam_sin_decay", lambda epoch: LEARNING_RATE_ADAM),
    ("adam_con_decay", lambda epoch: LEARNING_RATE_ADAM * DECAY_RATE ** epoch),
]
COLORES_VARIANTES = {"adam_sin_decay": "#4C72B0", "adam_con_decay": "#55A868"}


def entrenar(nombre, red, X_train, Y_train, X_val, Y_val_num, rng_batches, lr_schedule, quiet=False):
    """Variante baseline únicamente: sin augmentar_lote(), igual que en sgd_vs_adam_full.py con
    usar_augmentation=False."""
    Y_val_onehot = np.eye(10)[Y_val_num]
    historial_loss_train, historial_loss_val, historial_lr = [], [], []
    capas_con_pesos = [c for c in red if isinstance(c, (CapaDensa, CapaConv2D))]
    mejor_loss_val = np.inf
    mejor_epoca = None
    mejores_pesos = None
    t0 = time.time()

    for epoch in range(EPOCHS_MAX):
        learning_rate = lr_schedule(epoch)
        historial_lr.append(learning_rate)

        loss_train_acumulada = 0.0
        n_vistas = 0
        for Xb, Yb in generar_batches(X_train, Y_train, BATCH_SIZE, rng_batches):
            activacion = Xb
            for capa in red:
                activacion = capa.forward(activacion, entrenando=True)
            probs_train = activacion

            loss_batch = -np.mean(np.sum(Yb * np.log(probs_train + 1e-15), axis=1))
            loss_train_acumulada += loss_batch * Xb.shape[0]
            n_vistas += Xb.shape[0]

            gradiente = (probs_train - Yb) / Yb.shape[0]
            for capa in reversed(red[:-1]):
                gradiente = capa.backward(gradiente, learning_rate)

        loss_train = loss_train_acumulada / n_vistas
        historial_loss_train.append(loss_train)

        probs_val = predecir_cnn(red, X_val)
        loss_val = -np.mean(np.sum(Y_val_onehot * np.log(probs_val + 1e-15), axis=1))
        historial_loss_val.append(loss_val)

        if loss_val < mejor_loss_val:
            mejor_loss_val = loss_val
            mejor_epoca = epoch
            mejores_pesos = [(c.W.copy(), c.b.copy()) for c in capas_con_pesos]

        if not quiet:
            print(f"  [{nombre}] época {epoch}: loss_val={loss_val:.4f} lr={learning_rate:.6f} "
                  f"({time.time() - t0:.0f}s)")

        if epoch >= PACIENCIA_EARLY_STOP:
            loss_val_referencia = historial_loss_val[epoch - PACIENCIA_EARLY_STOP]
            mejora_relativa = (loss_val_referencia - loss_val) / loss_val_referencia
            if mejora_relativa < MEJORA_MINIMA_RELATIVA:
                if not quiet:
                    print(f"  [{nombre}] early stopping en la época {epoch + 1} ({time.time() - t0:.0f}s)")
                break
    else:
        if not quiet:
            print(f"  [{nombre}] completó las {EPOCHS_MAX} épocas sin activar el early stopping")

    for capa, (W, b) in zip(capas_con_pesos, mejores_pesos):
        capa.W, capa.b = W, b
    if not quiet:
        print(f"  [{nombre}] pesos restaurados al mínimo de validación: época {mejor_epoca + 1} "
              f"(loss_val={mejor_loss_val:.6f}), frente a la época de corte {len(historial_loss_train)}")

    return historial_loss_train, historial_loss_val, historial_lr, mejor_epoca, mejor_loss_val


def graficar_curvas(curvas_val, curvas_lr, ruta_salida):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True)
    for nombre, historial in curvas_val.items():
        ax1.plot(historial, color=COLORES_VARIANTES[nombre], label=nombre)
    ax1.set_yscale("log")
    ax1.set_ylabel("Pérdida de validación (escala log)")
    ax1.set_title("LR decay vs LR constante (Adam, mini-batch, CNN baseline)", fontweight="bold")
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
    X_train, Y_train, Y_train_num, X_val, Y_val_num, X_test, Y_test_num = cargar_datos(seed_split, quiet=quiet)

    resultados = {}
    curvas_val = {}
    curvas_lr = {}
    for nombre, lr_schedule in VARIANTES:
        if not quiet:
            print(f"\n=== Entrenando '{nombre}' ===")
        rng_modelo = np.random.default_rng(seed_modelo)
        red = crear_red(rng_modelo, OptimizadorAdam)
        rng_batches = np.random.default_rng(seed_modelo)
        hist_train, hist_val, hist_lr, mejor_epoca, mejor_loss_val = entrenar(
            nombre, red, X_train, Y_train, X_val, Y_val_num, rng_batches, lr_schedule, quiet=quiet
        )

        accuracy, matriz_confusion = evaluar(red, X_test, Y_test_num)
        if not quiet:
            print(f"  [{nombre}] accuracy en test: {accuracy:.4f}")

        if guardar_graficas:
            graficar_confusion(matriz_confusion, accuracy, f"Matriz de confusión test — {nombre}",
                                RESULTS_DIR / f"confusion_matrix_{nombre}.png")

        resultados[nombre] = {
            "epochs_entrenadas": len(hist_train),
            "epoca_mejor_val": mejor_epoca + 1,
            "loss_val_final": float(mejor_loss_val),
            "accuracy_test": accuracy,
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
        "variante_augmentation": "baseline",
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
