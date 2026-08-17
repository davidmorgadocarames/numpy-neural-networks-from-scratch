"""
FGSM (Fast Gradient Sign Method, Goodfellow et al. 2015) sobre la configuración de referencia
(Adam, mini-batch, split libre -- ver README raíz). Reutiliza `crear_red()` y `entrenar()` de
`sgd_vs_adam_full.py` sin tocar ese script -- entrena una red idéntica a la ya documentada y
la ataca después de entrenada, no durante.

La idea del ataque: `CapaDensa.backward()` ya calcula, en cada capa, el gradiente de la
pérdida respecto a SU ENTRADA (`dX_anterior`) -- necesario para propagarlo a la capa anterior.
En la primera capa de la red, ese gradiente es el gradiente de la pérdida respecto a los
PÍXELES de la imagen, y hoy se descarta (nadie lee el `dX_anterior` que devuelve la primera
capa, no hay una capa antes a la que pasárselo). FGSM es exactamente ese gradiente, usado a
propósito: en vez de mover los PESOS en la dirección que reduce el error (entrenamiento
normal), mueve los PÍXELES de la imagen en la dirección que MÁS lo aumenta --
`X_adv = clip(X + epsilon * signo(dL/dX), 0, 1)` -- una imagen que a simple vista sigue
pareciendo el mismo dígito, pero que la red ya no reconoce.

El backward() de la crafting se llama con learning_rate=0 (mismo truco que ya usa
tests/test_gradients.py) para extraer el gradiente sin mover ni un peso de la red ya
entrenada -- la red que ataca es la misma en todos los epsilon, no una versión ligeramente
reentrenada en cada paso.

Uso: python fgsm.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from capas import OptimizadorAdam, predecir

from digit_classifier_full import cargar_datos
from sgd_vs_adam_full import LEARNING_RATE_ADAM, crear_red, entrenar

RESULTS_DIR = Path(__file__).parent / "results_fgsm"
RESULTS_DIR.mkdir(exist_ok=True)

EPSILONS = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
EPSILON_EJEMPLOS = 0.15  # epsilon usado para la ilustración visual de abajo


def gradiente_respecto_a_entrada(red, X, Y_onehot):
    """Forward + backward con learning_rate=0 -- reutiliza exactamente el mismo bucle de
    entrenamiento (mismo cálculo de gradiente softmax+entropía cruzada, mismo orden de capas),
    pero capturando dX_anterior de la ÚLTIMA capa del bucle (la primera de la red) en vez de
    descartarlo."""
    activacion = X
    for capa in red:
        activacion = capa.forward(activacion, entrenando=True)
    A = activacion

    gradiente = (A - Y_onehot) / Y_onehot.shape[0]
    for capa in reversed(red[:-1]):
        gradiente = capa.backward(gradiente, learning_rate=0.0)
    return gradiente  # dL/dX de la primera capa == dL/dX de la imagen de entrada


def fgsm(red, X, Y_onehot, epsilon):
    if epsilon == 0.0:
        return X
    dX = gradiente_respecto_a_entrada(red, X, Y_onehot)
    return np.clip(X + epsilon * np.sign(dX), 0.0, 1.0)


def graficar_accuracy_vs_epsilon(epsilons, accuracies, ruta_salida):
    plt.figure(figsize=(7, 4.5))
    plt.plot(epsilons, accuracies, marker="o", color="#C44E52")
    plt.xlabel("Epsilon (magnitud de la perturbación)")
    plt.ylabel("Accuracy en test")
    plt.title("FGSM: accuracy frente a epsilon (red sin defensa)", fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.ylim(-0.02, 1.02)
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150)
    plt.close()


def graficar_ejemplos(red, X_test, Y_test_num, Y_test_onehot, epsilon, ruta_salida, n=6, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_test), size=n, replace=False)
    X_muestra = X_test[idx]
    Y_muestra_onehot = Y_test_onehot[idx]
    Y_muestra_num = Y_test_num[idx]

    X_adv = fgsm(red, X_muestra.copy(), Y_muestra_onehot, epsilon)
    pred_original = np.argmax(predecir(red, X_muestra), axis=1)
    pred_adv = np.argmax(predecir(red, X_adv), axis=1)

    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 5))
    for col in range(n):
        axes[0, col].imshow(X_muestra[col].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(f"real={Y_muestra_num[col]}\npred={pred_original[col]}", fontsize=9)
        axes[0, col].axis("off")

        axes[1, col].imshow(X_adv[col].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        color = "red" if pred_adv[col] != Y_muestra_num[col] else "black"
        axes[1, col].set_title(f"pred={pred_adv[col]}", fontsize=9, color=color)
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=10)
    axes[1, 0].set_ylabel(f"FGSM ε={epsilon}", fontsize=10)
    plt.suptitle(f"Misma imagen, con y sin perturbación FGSM (ε={epsilon}) -- sigue siendo\n"
                 "el mismo dígito a simple vista, aunque la red ya no lo reconozca", fontweight="bold")
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
    entrenar("fgsm_base", red, X_train, Y_train, X_val, Y_val, seed_modelo, LEARNING_RATE_ADAM, quiet=quiet)

    accuracies = []
    for epsilon in EPSILONS:
        X_adv = fgsm(red, X_test.copy(), Y_test_onehot, epsilon)
        A_adv = predecir(red, X_adv)
        acc = float(np.mean(np.argmax(A_adv, axis=1) == Y_test_num))
        accuracies.append(acc)
        if not quiet:
            print(f"  epsilon={epsilon:.2f}  accuracy={acc:.4f}")

    metrics = {
        "seed_split": seed_split,
        "seed_modelo": seed_modelo,
        "epsilons": EPSILONS,
        "accuracies": accuracies,
        "n_test": int(len(X_test)),
    }

    if not guardar_graficas:
        return metrics

    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    graficar_accuracy_vs_epsilon(EPSILONS, accuracies, RESULTS_DIR / "accuracy_vs_epsilon.png")
    graficar_ejemplos(red, X_test, Y_test_num, Y_test_onehot, EPSILON_EJEMPLOS,
                       RESULTS_DIR / "ejemplos_fgsm.png")

    if not quiet:
        print(f"Resultados guardados en {RESULTS_DIR}")

    return metrics


if __name__ == "__main__":
    main()
