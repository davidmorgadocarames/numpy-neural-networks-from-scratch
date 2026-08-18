"""
Entrenamiento adversario para la CNN -- misma idea que
[`../07-reconocimiento-digitos/entrenamiento_adversario.py`](../07-reconocimiento-digitos/entrenamiento_adversario.py),
léelo primero para la explicación completa. Cada batch se craftea con FGSM (no PGD, por coste --
ver ese README) usando los pesos actuales antes de entrenar sobre él. Variante `baseline`
(sin augmentation) de la configuración de referencia.

Uso: python entrenamiento_adversario.py
"""

import json
import sys
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
    generar_batches,
)
from fgsm import EPSILONS, fgsm
from pgd import pgd
from sgd_vs_adam_full import LEARNING_RATE_ADAM, crear_red

RESULTS_DIR = Path(__file__).parent / "results_entrenamiento_adversario"
RESULTS_DIR.mkdir(exist_ok=True)

EPSILON_ENTRENAMIENTO = 0.1


def entrenar_adversario(red, X_train, Y_train, X_val, Y_val_num, seed_modelo, epsilon, quiet=False):
    Y_val_onehot = np.eye(10)[Y_val_num]
    historial_loss_train, historial_loss_val = [], []
    capas_con_pesos = [c for c in red if isinstance(c, (CapaDensa, CapaConv2D))]
    mejor_loss_val = np.inf
    mejor_epoca = None
    mejores_pesos = None
    rng_batches = np.random.default_rng(seed_modelo)

    for epoch in range(EPOCHS_MAX):
        loss_train_acumulada = 0.0
        n_vistas = 0
        for Xb, Yb in generar_batches(X_train, Y_train, BATCH_SIZE, rng_batches):
            Xb_adv = fgsm(red, Xb, Yb, epsilon)

            activacion = Xb_adv
            for capa in red:
                activacion = capa.forward(activacion, entrenando=True)
            probs_train = activacion

            loss_batch = -np.mean(np.sum(Yb * np.log(probs_train + 1e-15), axis=1))
            loss_train_acumulada += loss_batch * Xb.shape[0]
            n_vistas += Xb.shape[0]

            gradiente = (probs_train - Yb) / Yb.shape[0]
            for capa in reversed(red[:-1]):
                gradiente = capa.backward(gradiente, LEARNING_RATE_ADAM)

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
            print(f"  época {epoch}: loss_train={loss_train:.4f} loss_val={loss_val:.4f}")

        if epoch >= PACIENCIA_EARLY_STOP:
            loss_val_referencia = historial_loss_val[epoch - PACIENCIA_EARLY_STOP]
            mejora_relativa = (loss_val_referencia - loss_val) / loss_val_referencia
            if mejora_relativa < MEJORA_MINIMA_RELATIVA:
                if not quiet:
                    print(f"  early stopping en la época {epoch + 1}")
                break
    else:
        if not quiet:
            print(f"  completó las {EPOCHS_MAX} épocas sin activar el early stopping")

    for capa, (W, b) in zip(capas_con_pesos, mejores_pesos):
        capa.W, capa.b = W, b
    if not quiet:
        print(f"  pesos restaurados al mínimo de validación: época {mejor_epoca + 1} "
              f"(loss_val={mejor_loss_val:.6f})")

    return historial_loss_train, historial_loss_val, mejor_epoca, mejor_loss_val


def graficar_comparativa(epsilons, acc_sin_defensa, acc_con_defensa, ataque, ruta_salida):
    plt.figure(figsize=(7, 4.5))
    plt.plot(epsilons, acc_sin_defensa, marker="o", color="#4C72B0", label="Sin defensa")
    plt.plot(epsilons, acc_con_defensa, marker="o", color="#55A868", label="Con entrenamiento adversario")
    plt.xlabel("Epsilon (radio de la perturbación)")
    plt.ylabel("Accuracy en test")
    plt.title(f"Entrenamiento adversario: accuracy frente a epsilon ({ataque})", fontweight="bold")
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
        print(f"=== Entrenamiento adversario (FGSM, epsilon={EPSILON_ENTRENAMIENTO}, baseline) ===")
    rng_modelo = np.random.default_rng(seed_modelo)
    red = crear_red(rng_modelo, OptimizadorAdam)
    entrenar_adversario(red, X_train, Y_train, X_val, Y_val_num, seed_modelo, EPSILON_ENTRENAMIENTO, quiet=quiet)

    acc_limpia = float(np.mean(np.argmax(predecir_cnn(red, X_test), axis=1) == Y_test_num))
    if not quiet:
        print(f"Accuracy en test, datos limpios: {acc_limpia:.4f}")

    acc_fgsm_defendida, acc_pgd_defendida = [], []
    for epsilon in EPSILONS:
        X_adv_fgsm = fgsm(red, X_test.copy(), Y_test_onehot, epsilon)
        acc_fgsm_defendida.append(float(np.mean(np.argmax(predecir_cnn(red, X_adv_fgsm), axis=1) == Y_test_num)))

        X_adv_pgd = pgd(red, X_test.copy(), Y_test_onehot, epsilon)
        acc_pgd_defendida.append(float(np.mean(np.argmax(predecir_cnn(red, X_adv_pgd), axis=1) == Y_test_num)))

        if not quiet:
            print(f"  epsilon={epsilon:.2f}  FGSM={acc_fgsm_defendida[-1]:.4f}  PGD={acc_pgd_defendida[-1]:.4f}")

    metrics_fgsm_sin_defensa = json.loads((Path(__file__).parent / "results_fgsm" / "metrics.json")
                                           .read_text(encoding="utf-8"))
    metrics_pgd_sin_defensa = json.loads((Path(__file__).parent / "results_pgd" / "metrics.json")
                                          .read_text(encoding="utf-8"))

    metrics = {
        "seed_split": seed_split,
        "seed_modelo": seed_modelo,
        "epsilon_entrenamiento": EPSILON_ENTRENAMIENTO,
        "accuracy_test_limpia": acc_limpia,
        "epsilons": EPSILONS,
        "accuracy_fgsm_con_defensa": acc_fgsm_defendida,
        "accuracy_fgsm_sin_defensa": metrics_fgsm_sin_defensa["accuracies"],
        "accuracy_pgd_con_defensa": acc_pgd_defendida,
        "accuracy_pgd_sin_defensa": metrics_pgd_sin_defensa["accuracies_pgd"],
    }

    if not guardar_graficas:
        return metrics

    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    graficar_comparativa(EPSILONS, metrics_fgsm_sin_defensa["accuracies"], acc_fgsm_defendida,
                          "ataque FGSM", RESULTS_DIR / "defensa_vs_fgsm.png")
    graficar_comparativa(EPSILONS, metrics_pgd_sin_defensa["accuracies_pgd"], acc_pgd_defendida,
                          "ataque PGD", RESULTS_DIR / "defensa_vs_pgd.png")

    if not quiet:
        print(f"Resultados guardados en {RESULTS_DIR}")

    return metrics


if __name__ == "__main__":
    main()
