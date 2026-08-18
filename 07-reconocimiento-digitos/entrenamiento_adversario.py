"""
Entrenamiento adversario -- la defensa frente a FGSM/PGD ([`fgsm.py`](fgsm.py),
[`pgd.py`](pgd.py), léelos primero). La idea: si la red solo ve imágenes limpias durante el
entrenamiento, nunca aprende a lidiar con las perturbadas. Cada paso de entrenamiento fabrica
primero un ejemplo adversario con FGSM (usando los pesos actuales de la red, en ese momento
del entrenamiento) y entrena sobre ESE en vez de sobre la imagen limpia -- así la red ve
ejemplos adversarios en cada época, no solo al final.

Se usa FGSM para craftear (no PGD): PGD necesita 10 pasadas extra de gradiente por batch, que
sobre CADA batch de CADA época dispararía el coste de entrenar -- FGSM necesita solo 1 pasada
extra, ~2x el coste de un entrenamiento normal en vez de ~11x. Es una elección de coste
consciente, no una limitación técnica del método.

epsilon_entrenamiento=0.1 -- el mismo valor donde ya se documentó el colapso de la red sin
defensa (7.30% con FGSM, 3.67% con PGD).

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
from capas import CapaDensa, OptimizadorAdam, predecir

from digit_classifier_full import (
    BATCH_SIZE,
    EPOCHS_MAX,
    MEJORA_MINIMA_RELATIVA,
    PACIENCIA_EARLY_STOP,
    cargar_datos,
    generar_batches,
)
from fgsm import EPSILONS, fgsm, gradiente_respecto_a_entrada
from pgd import pgd
from sgd_vs_adam_full import LEARNING_RATE_ADAM, crear_red

RESULTS_DIR = Path(__file__).parent / "results_entrenamiento_adversario"
RESULTS_DIR.mkdir(exist_ok=True)

EPSILON_ENTRENAMIENTO = 0.1


def entrenar_adversario(red, X_train, Y_train, X_val, Y_val, seed_modelo, epsilon, quiet=False):
    """Mismo bucle mini-batch que sgd_vs_adam_full.py, con un paso extra al principio de cada
    batch: craftear su versión FGSM con los pesos ACTUALES antes de entrenar sobre ella. La
    validación se mide sobre imágenes LIMPIAS (igual que en el resto del repo la augmentation
    nunca se aplica a validación) -- interesa saber si la red sigue funcionando bien en el caso
    normal, no solo bajo ataque."""
    historial_loss_train, historial_loss_val = [], []
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
            Ab = activacion

            loss_batch = -np.mean(np.sum(Yb * np.log(Ab + 1e-15), axis=1))
            loss_train_acumulada += loss_batch * Xb.shape[0]
            n_vistas += Xb.shape[0]

            gradiente = (Ab - Yb) / Yb.shape[0]
            for capa in reversed(red[:-1]):
                gradiente = capa.backward(gradiente, LEARNING_RATE_ADAM)

        loss_train = loss_train_acumulada / n_vistas
        historial_loss_train.append(loss_train)

        A_val = predecir(red, X_val)
        loss_val = -np.mean(np.sum(Y_val * np.log(A_val + 1e-15), axis=1))
        historial_loss_val.append(loss_val)

        if loss_val < mejor_loss_val:
            mejor_loss_val = loss_val
            mejor_epoca = epoch
            mejores_pesos = [(c.W.copy(), c.b.copy()) for c in red if isinstance(c, CapaDensa)]

        if not quiet and epoch % 2 == 0:
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

    for capa, (W, b) in zip([c for c in red if isinstance(c, CapaDensa)], mejores_pesos):
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
    X_train, Y_train, Y_train_num, X_val, Y_val, Y_val_num, X_test, Y_test_num = cargar_datos(seed_split, quiet=quiet)
    Y_test_onehot = np.eye(10)[Y_test_num]

    if not quiet:
        print(f"=== Entrenamiento adversario (FGSM, epsilon={EPSILON_ENTRENAMIENTO}) ===")
    rng_modelo = np.random.default_rng(seed_modelo)
    red = crear_red(rng_modelo, OptimizadorAdam)
    entrenar_adversario(red, X_train, Y_train, X_val, Y_val, seed_modelo, EPSILON_ENTRENAMIENTO, quiet=quiet)

    acc_limpia = float(np.mean(np.argmax(predecir(red, X_test), axis=1) == Y_test_num))
    if not quiet:
        print(f"Accuracy en test, datos limpios: {acc_limpia:.4f}")

    acc_fgsm_defendida, acc_pgd_defendida = [], []
    for epsilon in EPSILONS:
        X_adv_fgsm = fgsm(red, X_test.copy(), Y_test_onehot, epsilon)
        acc_fgsm_defendida.append(float(np.mean(np.argmax(predecir(red, X_adv_fgsm), axis=1) == Y_test_num)))

        X_adv_pgd = pgd(red, X_test.copy(), Y_test_onehot, epsilon)
        acc_pgd_defendida.append(float(np.mean(np.argmax(predecir(red, X_adv_pgd), axis=1) == Y_test_num)))

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
