"""
Validación cruzada (k-fold) -- alternativa metodológica al barrido de semillas Monte Carlo que
ya mide "Robustez frente a la semilla" en el README de este proyecto. Aplicada aquí (no en
07/08) porque es donde k-fold tiene más sentido: con solo 450 puntos en todo el dataset,
reservar un 20% fijo para test en cada repetición (como hace el barrido de semillas) deja
fuera del entrenamiento 90 puntos que nunca se aprovechan como test en las demás repeticiones.
K-fold, en cambio, hace que CADA punto sirva de test exactamente una vez por cada vuelta
completa a los K pliegues -- usa todo el dataset para evaluar, no solo una fracción fija.

Diseño, respetando la misma regla de "test se toca una sola vez" que el resto del repo: para
cada uno de los K pliegues, ESE pliegue es el test (nunca entra en el entrenamiento de esa
vuelta); el resto de los datos (K-1 pliegues) se reparte 80/20 en train/validación para
decidir el early stopping -- el test de cada vuelta sigue evaluándose una única vez, con la
red ya entrenada y congelada, igual que en cualquier otro proyecto del repo. No es "k-fold
clásico" en el sentido más estricto (que normalmente no reserva una validación aparte dentro
de cada pliegue), es la adaptación de k-fold a la metodología de train/val/test de tres partes
que ya usa el resto del repo.

Reutiliza crear_red() y entrenar() de sgd_vs_adam.py (variante SGD, para comparar en igualdad
de condiciones con "Robustez frente a la semilla", que también usa SGD) sin tocar ese script
ni spiral_classifier.py.

Uso: python k_fold.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from capas import OptimizadorSGD, predecir

from spiral_classifier import K_CLASES, generar_datos
from sgd_vs_adam import LEARNING_RATE_SGD, crear_red, entrenar

RESULTS_DIR = Path(__file__).parent / "results_k_fold"
RESULTS_DIR.mkdir(exist_ok=True)

K_PLIEGUES = 5
N_REPETICIONES = 4  # 4 x 5 = 20 entrenamientos en total, mismo N que el barrido de semillas
SEED_MODELO_FIJO = 0  # fijo en todas las vueltas -- aísla el efecto de "qué es test", no el de la inicialización


def pliegues_estratificados(Y_num, k, seed):
    """Reparte los índices de cada clase en k partes lo más iguales posible, para que cada
    pliegue tenga la misma proporción de las 3 clases que el dataset completo."""
    rng = np.random.default_rng(seed)
    pliegues = [[] for _ in range(k)]
    for clase in range(K_CLASES):
        idx_clase = np.where(Y_num == clase)[0]
        rng.shuffle(idx_clase)
        for i, parte in enumerate(np.array_split(idx_clase, k)):
            pliegues[i].extend(parte.tolist())
    return [np.array(p) for p in pliegues]


def entrenar_y_evaluar_pliegue(X, Y, Y_num, indices_test, indices_pool, seed_val, seed_modelo):
    rng_val = np.random.default_rng(seed_val)
    indices_pool = indices_pool.copy()
    rng_val.shuffle(indices_pool)
    corte = int(0.8 * len(indices_pool))
    indices_train, indices_val = indices_pool[:corte], indices_pool[corte:]

    X_train, X_val, X_test = X[indices_train], X[indices_val], X[indices_test]
    Y_train, Y_val = Y[indices_train], Y[indices_val]
    Y_num_test = Y_num[indices_test]

    rng_modelo = np.random.default_rng(seed_modelo)
    red = crear_red(rng_modelo, OptimizadorSGD)
    entrenar("k_fold", red, X_train, Y_train, X_val, Y_val, LEARNING_RATE_SGD, quiet=True)

    A_test = predecir(red, X_test)
    pred_test = np.argmax(A_test, axis=1)
    return float(np.mean(pred_test == Y_num_test))


def graficar_comparativa(valores_k_fold, media_seed_sweep, std_seed_sweep, ruta_salida):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    rng = np.random.default_rng(0)
    y_jitter = rng.uniform(-0.1, 0.1, size=len(valores_k_fold))
    ax.scatter(valores_k_fold, np.zeros(len(valores_k_fold)) + y_jitter, color="#55A868",
               edgecolors="black", s=60, zorder=3, alpha=0.85, label="k-fold (20 pliegues)")
    media_kfold = np.mean(valores_k_fold)
    ax.axvline(media_kfold, color="#2d5f3f", linestyle="--", linewidth=1.3)
    ax.axvline(media_seed_sweep, color="#333333", linestyle=":", linewidth=1.3,
               label=f"media barrido de semillas = {media_seed_sweep:.4f}")
    ax.set_yticks([])
    ax.set_ylim(-0.3, 0.3)
    ax.set_xlabel("Accuracy en test")
    ax.set_title("k-fold vs. barrido de semillas: accuracy en test (06, SGD)", fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150)
    plt.close()


def main(quiet=False):
    X, Y, Y_num = generar_datos()

    valores = []
    for repeticion in range(N_REPETICIONES):
        pliegues = pliegues_estratificados(Y_num, K_PLIEGUES, seed=1_000 + repeticion)
        for i in range(K_PLIEGUES):
            indices_test = pliegues[i]
            indices_pool = np.concatenate([pliegues[j] for j in range(K_PLIEGUES) if j != i])
            acc = entrenar_y_evaluar_pliegue(X, Y, Y_num, indices_test, indices_pool,
                                              seed_val=2_000 + repeticion * K_PLIEGUES + i,
                                              seed_modelo=SEED_MODELO_FIJO)
            valores.append(acc)
            if not quiet:
                print(f"  repetición {repeticion + 1}/{N_REPETICIONES}, pliegue {i + 1}/{K_PLIEGUES}: "
                      f"accuracy={acc:.4f}")

    media_kfold = float(np.mean(valores))
    std_kfold = float(np.std(valores, ddof=1))
    if not quiet:
        print(f"\nk-fold: media={media_kfold:.4f} desviación típica={std_kfold:.4f} (N={len(valores)})")

    ruta_seed_sweep = Path(__file__).parent / "results" / "metrics_seed_sweep.json"
    datos_seed_sweep = json.loads(ruta_seed_sweep.read_text(encoding="utf-8"))
    media_seed_sweep = datos_seed_sweep["unico"]["media"]
    std_seed_sweep = datos_seed_sweep["unico"]["desviacion_tipica"]

    resultado = {
        "k_pliegues": K_PLIEGUES,
        "n_repeticiones": N_REPETICIONES,
        "seed_modelo_fijo": SEED_MODELO_FIJO,
        "valores_k_fold": valores,
        "media_k_fold": media_kfold,
        "desviacion_tipica_k_fold": std_kfold,
        "media_barrido_semillas": media_seed_sweep,
        "desviacion_tipica_barrido_semillas": std_seed_sweep,
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    graficar_comparativa(valores, media_seed_sweep, std_seed_sweep, RESULTS_DIR / "k_fold_vs_seed_sweep.png")

    if not quiet:
        print(f"Barrido de semillas (referencia): media={media_seed_sweep:.4f} "
              f"desviación típica={std_seed_sweep:.4f}")
        print(f"Resultados guardados en {RESULTS_DIR}")

    return resultado


if __name__ == "__main__":
    main()
