"""
PGD (Projected Gradient Descent, Madry et al. 2017 -- https://arxiv.org/abs/1706.06083) sobre
la configuración de referencia. Versión iterativa de FGSM ([`fgsm.py`](fgsm.py), léelo primero
para la explicación del mecanismo base): en vez de un solo paso grande en la dirección del
signo del gradiente, da varios pasos pequeños, y tras cada uno **proyecta** la imagen de vuelta
dentro de la bola de radio epsilon alrededor del original -- de ahí el nombre. Reutiliza
`gradiente_respecto_a_entrada()` de `fgsm.py` sin duplicarlo.

`alpha` (tamaño de cada paso) sigue la regla práctica de Madry et al.: `alpha = epsilon / 4`,
con `N_PASOS = 10` -- con estos dos valores, el ataque puede recorrer hasta 2.5 veces el radio
epsilon en total (10 pasos x 0.25 epsilon), de sobra para explorar bien la bola de radio
epsilon antes de la proyección final.

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
from capas import OptimizadorAdam, predecir

from digit_classifier_full import cargar_datos
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
        X_adv = np.clip(X_adv, X_original - epsilon, X_original + epsilon)  # proyección a la bola de radio epsilon
        X_adv = np.clip(X_adv, 0.0, 1.0)  # rango de píxel válido
    return X_adv


def graficar_comparativa(epsilons, acc_fgsm, acc_pgd, ruta_salida):
    plt.figure(figsize=(7, 4.5))
    plt.plot(epsilons, acc_fgsm, marker="o", color="#4C72B0", label="FGSM (1 paso)")
    plt.plot(epsilons, acc_pgd, marker="o", color="#C44E52", label=f"PGD ({N_PASOS} pasos)")
    plt.xlabel("Epsilon (radio de la perturbación)")
    plt.ylabel("Accuracy en test")
    plt.title("FGSM vs PGD: accuracy frente a epsilon (red sin defensa)", fontweight="bold")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.ylim(-0.02, 1.02)
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
    entrenar("pgd_base", red, X_train, Y_train, X_val, Y_val, seed_modelo, LEARNING_RATE_ADAM, quiet=quiet)

    accuracies_pgd = []
    for epsilon in EPSILONS:
        X_adv = pgd(red, X_test.copy(), Y_test_onehot, epsilon)
        A_adv = predecir(red, X_adv)
        acc = float(np.mean(np.argmax(A_adv, axis=1) == Y_test_num))
        accuracies_pgd.append(acc)
        if not quiet:
            print(f"  epsilon={epsilon:.2f}  accuracy (PGD)={acc:.4f}")

    # Reutiliza los resultados de FGSM ya guardados (misma red de referencia, mismas semillas)
    # en vez de repetir el ataque de un solo paso.
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
