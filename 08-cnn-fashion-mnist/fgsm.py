"""
FGSM sobre la configuración de referencia (Adam, mini-batch, variante `baseline`) -- ver
README raíz para el porqué. Misma idea que
[`../07-reconocimiento-digitos/fgsm.py`](../07-reconocimiento-digitos/fgsm.py): el
`dX_anterior` que la primera capa de la red ya calcula (y descarta) en cada `backward()` es
literalmente el gradiente de la pérdida respecto a los píxeles de la imagen -- FGSM lo usa a
propósito para perturbar la imagen en la dirección que más aumenta el error, en vez de
descartarlo. Reutiliza `crear_red()` y `entrenar()` de `sgd_vs_adam_full.py` sin tocar ese
script.

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
from capas import OptimizadorAdam
from capas_cnn import predecir_cnn

from cnn_fashion_mnist_full import NOMBRES_CLASES, cargar_datos
from sgd_vs_adam_full import LEARNING_RATE_ADAM, crear_red, entrenar

RESULTS_DIR = Path(__file__).parent / "results_fgsm"
RESULTS_DIR.mkdir(exist_ok=True)

EPSILONS = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
EPSILON_EJEMPLOS = 0.15


def gradiente_respecto_a_entrada(red, X, Y_onehot):
    activacion = X
    for capa in red:
        activacion = capa.forward(activacion, entrenando=True)
    probs = activacion

    gradiente = (probs - Y_onehot) / Y_onehot.shape[0]
    for capa in reversed(red[:-1]):
        gradiente = capa.backward(gradiente, learning_rate=0.0)
    return gradiente


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
    plt.title("FGSM: accuracy frente a epsilon (CNN sin defensa)", fontweight="bold")
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
    pred_original = np.argmax(predecir_cnn(red, X_muestra), axis=1)
    pred_adv = np.argmax(predecir_cnn(red, X_adv), axis=1)

    fig, axes = plt.subplots(2, n, figsize=(2.4 * n, 5.3))
    for col in range(n):
        axes[0, col].imshow(X_muestra[col, :, :, 0], cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(f"real={NOMBRES_CLASES[Y_muestra_num[col]]}\n"
                                f"pred={NOMBRES_CLASES[pred_original[col]]}", fontsize=8)
        axes[0, col].axis("off")

        axes[1, col].imshow(X_adv[col, :, :, 0], cmap="gray", vmin=0, vmax=1)
        color = "red" if pred_adv[col] != Y_muestra_num[col] else "black"
        axes[1, col].set_title(f"pred={NOMBRES_CLASES[pred_adv[col]]}", fontsize=8, color=color)
        axes[1, col].axis("off")

    plt.suptitle(f"Misma imagen, con y sin perturbación FGSM (ε={epsilon}) -- sigue siendo\n"
                 "la misma prenda a simple vista, aunque la red ya no la reconozca", fontweight="bold")
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150)
    plt.close()


def main(seed_split=42, seed_modelo=42, quiet=False, guardar_graficas=True) -> dict:
    X_train, Y_train, Y_train_num, X_val, Y_val_num, X_test, Y_test_num = cargar_datos(seed_split, quiet=quiet)
    Y_test_onehot = np.eye(10)[Y_test_num]

    if not quiet:
        print("=== Entrenando la red de referencia (Adam, mini-batch, baseline) ===")
    rng_modelo = np.random.default_rng(seed_modelo)
    red = crear_red(rng_modelo, OptimizadorAdam)
    rng_aug = np.random.default_rng(seed_modelo)
    rng_batches = np.random.default_rng(seed_modelo)
    entrenar("fgsm_base", red, X_train, Y_train, X_val, Y_val_num, False, rng_aug, rng_batches,
              LEARNING_RATE_ADAM, quiet=quiet)

    accuracies = []
    for epsilon in EPSILONS:
        X_adv = fgsm(red, X_test.copy(), Y_test_onehot, epsilon)
        probs_adv = predecir_cnn(red, X_adv)
        acc = float(np.mean(np.argmax(probs_adv, axis=1) == Y_test_num))
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
