"""
PGD sobre la configuración de referencia (Adam, mini-batch, `baseline`). Misma idea que
[`../07-reconocimiento-digitos/pgd.py`](../07-reconocimiento-digitos/pgd.py) -- léelo primero
para la explicación completa del mecanismo. Reutiliza `gradiente_respecto_a_entrada()` de
`fgsm.py` sin duplicarlo.

Uso: python pgd.py
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

from cnn_fashion_mnist_full import cargar_datos
from fgsm import EPSILONS, gradiente_respecto_a_entrada
from sgd_vs_adam_full import LEARNING_RATE_ADAM, crear_red, entrenar

RESULTS_DIR = Path(__file__).parent / "results_pgd"
RESULTS_DIR.mkdir(exist_ok=True)

N_PASOS = 10


def pgd(red, X, Y_onehot, epsilon, n_pasos=N_PASOS):
    if epsilon == 0.0:
        return X
    alpha = epsilon / 4
    X_original = X.copy()
    X_adv = X.copy()
    for _ in range(n_pasos):
        dX = gradiente_respecto_a_entrada(red, X_adv, Y_onehot)
        X_adv = X_adv + alpha * np.sign(dX)
        X_adv = np.clip(X_adv, X_original - epsilon, X_original + epsilon)
        X_adv = np.clip(X_adv, 0.0, 1.0)
    return X_adv


def graficar_comparativa(epsilons, acc_fgsm, acc_pgd, ruta_salida):
    plt.figure(figsize=(7, 4.5))
    plt.plot(epsilons, acc_fgsm, marker="o", color="#4C72B0", label="FGSM (1 paso)")
    plt.plot(epsilons, acc_pgd, marker="o", color="#C44E52", label=f"PGD ({N_PASOS} pasos)")
    plt.xlabel("Epsilon (radio de la perturbación)")
    plt.ylabel("Accuracy en test")
    plt.title("FGSM vs PGD: accuracy frente a epsilon (CNN sin defensa)", fontweight="bold")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.ylim(-0.02, 1.02)
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
    entrenar("pgd_base", red, X_train, Y_train, X_val, Y_val_num, False, rng_aug, rng_batches,
              LEARNING_RATE_ADAM, quiet=quiet)

    accuracies_pgd = []
    for epsilon in EPSILONS:
        X_adv = pgd(red, X_test.copy(), Y_test_onehot, epsilon)
        probs_adv = predecir_cnn(red, X_adv)
        acc = float(np.mean(np.argmax(probs_adv, axis=1) == Y_test_num))
        accuracies_pgd.append(acc)
        if not quiet:
            print(f"  epsilon={epsilon:.2f}  accuracy (PGD)={acc:.4f}")

    ruta_fgsm = Path(__file__).parent / "results_fgsm" / "metrics.json"
    metrics_fgsm = json.loads(ruta_fgsm.read_text(encoding="utf-8"))
    accuracies_fgsm = metrics_fgsm["accuracies"]

    metrics = {
        "seed_split": seed_split,
        "seed_modelo": seed_modelo,
        "n_pasos": N_PASOS,
        "epsilons": EPSILONS,
        "accuracies_pgd": accuracies_pgd,
        "accuracies_fgsm": accuracies_fgsm,
        "n_test": int(len(X_test)),
    }

    if not guardar_graficas:
        return metrics

    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    graficar_comparativa(EPSILONS, accuracies_fgsm, accuracies_pgd, RESULTS_DIR / "fgsm_vs_pgd.png")

    if not quiet:
        print(f"Resultados guardados en {RESULTS_DIR}")

    return metrics


if __name__ == "__main__":
    main()
