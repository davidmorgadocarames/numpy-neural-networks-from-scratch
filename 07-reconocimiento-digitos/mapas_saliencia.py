"""
Mapas de saliencia: mismo gradiente que ya calcula
[`gradiente_respecto_a_entrada()`](fgsm.py) para FGSM, pero usado para VISUALIZAR en vez de
para atacar -- en lugar de mover los píxeles en esa dirección, se dibuja directamente el
tamaño del gradiente en cada píxel: cuánto cambiaría la pérdida si ese píxel concreto
cambiase. Los píxeles con gradiente grande son los que la red está "mirando" para decidir la
clase -- el equivalente para una red densa (sin estructura espacial interna) de lo que
Grad-CAM hace para una CNN (ver `../08-cnn-fashion-mnist/grad_cam.py`), que necesita un mapa
de activaciones convolucional que aquí no existe.

Uso: python mapas_saliencia.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from capas import OptimizadorAdam, predecir

from digit_classifier_full import cargar_datos
from fgsm import gradiente_respecto_a_entrada
from sgd_vs_adam_full import LEARNING_RATE_ADAM, crear_red, entrenar

RESULTS_DIR = Path(__file__).parent / "results_saliencia"
RESULTS_DIR.mkdir(exist_ok=True)


def mapa_saliencia(red, X, Y_onehot):
    """|dL/dX|, normalizado a [0,1] por imagen para que el brillo sea comparable entre
    ejemplos distintos -- el signo del gradiente ya no importa aquí (a diferencia de FGSM):
    solo interesa CUÁNTO influye cada píxel, no en qué dirección."""
    dX = gradiente_respecto_a_entrada(red, X, Y_onehot)
    saliencia = np.abs(dX)
    maximos = saliencia.max(axis=1, keepdims=True)
    maximos[maximos == 0] = 1.0  # evita dividir por cero si algún ejemplo tiene gradiente nulo
    return saliencia / maximos


def graficar_saliencia(red, X_test, Y_test_num, Y_test_onehot, ruta_salida, n=6, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_test), size=n, replace=False)
    X_muestra = X_test[idx]
    Y_muestra_onehot = Y_test_onehot[idx]
    Y_muestra_num = Y_test_num[idx]

    saliencia = mapa_saliencia(red, X_muestra.copy(), Y_muestra_onehot)
    pred = np.argmax(predecir(red, X_muestra), axis=1)

    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 5))
    for col in range(n):
        axes[0, col].imshow(X_muestra[col].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(f"real={Y_muestra_num[col]}, pred={pred[col]}", fontsize=9)
        axes[0, col].axis("off")

        axes[1, col].imshow(saliencia[col].reshape(28, 28), cmap="hot")
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("Dígito", fontsize=10)
    axes[1, 0].set_ylabel("Saliencia |dL/dX|", fontsize=10)
    plt.suptitle("Qué píxeles importan para la decisión de la red (más brillante = más peso)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150)
    plt.close()


def main(seed_split=42, seed_modelo=42, quiet=False, guardar_graficas=True) -> dict:
    X_train, Y_train, Y_train_num, X_val, Y_val, Y_val_num, X_test, Y_test_num = cargar_datos(seed_split, quiet=quiet)
    Y_test_onehot = np.eye(10)[Y_test_num]

    if not quiet:
        print("=== Entrenando la red de referencia (Adam, mini-batch) ===")
    rng_modelo = np.random.default_rng(seed_modelo)
    red = crear_red(rng_modelo, OptimizadorAdam)
    entrenar("saliencia_base", red, X_train, Y_train, X_val, Y_val, seed_modelo, LEARNING_RATE_ADAM, quiet=quiet)

    if not guardar_graficas:
        return {"seed_split": seed_split, "seed_modelo": seed_modelo}

    graficar_saliencia(red, X_test, Y_test_num, Y_test_onehot, RESULTS_DIR / "mapas_saliencia.png")

    if not quiet:
        print(f"Resultados guardados en {RESULTS_DIR}")

    return {"seed_split": seed_split, "seed_modelo": seed_modelo}


if __name__ == "__main__":
    main()
