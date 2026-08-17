"""
SGD vs Adam sobre la CNN de Fashion-MNIST (muestra reducida, full-batch: mismo dataset que
cnn_fashion_mnist.py, 2.400/600/1.000). No toca cnn_fashion_mnist.py -- reutiliza su
cargar_datos(), evaluar() y augmentar_lote(), y guarda sus propios resultados en
results_sgd_vs_adam/. Mismo patrón que 06-zonas-espirales/sgd_vs_adam.py y
07-reconocimiento-digitos/sgd_vs_adam*.py.

Combina las 3 variantes de augmentation del proyecto original (baseline/augmented/
augmented_sin_flip) con los 2 optimizadores -- 6 combinaciones por ejecución. Usa
--solo-augmentacion para restringir a un subconjunto durante calibración/pruebas rápidas.

Uso:
    python sgd_vs_adam.py                              # las 6 combinaciones
    python sgd_vs_adam.py --solo-augmentacion baseline  # calibración rápida
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from capas import ActivacionLeakyReLU, ActivacionSoftmax, CapaDensa, OptimizadorAdam, OptimizadorSGD
from capas_cnn import CapaConv2D, CapaDropout, CapaFlatten, CapaMaxPool2D, augmentar_lote, predecir_cnn

from cnn_fashion_mnist import EPOCHS_MAX, MEJORA_MINIMA_RELATIVA, NOMBRES_CLASES, PACIENCIA_EARLY_STOP, cargar_datos, evaluar, graficar_confusion

RESULTS_DIR = Path(__file__).parent / "results_sgd_vs_adam"
RESULTS_DIR.mkdir(exist_ok=True)

LEARNING_RATE_SGD = 0.12  # igual que cnn_fashion_mnist.py
LEARNING_RATE_ADAM = 0.001  # calibrado a mano

VARIANTES_AUMENTACION = [
    ("baseline", False, 0.5),
    ("augmented", True, 0.5),
    ("augmented_sin_flip", True, 0.0),
]
OPTIMIZADORES = [("sgd", OptimizadorSGD, LEARNING_RATE_SGD), ("adam", OptimizadorAdam, LEARNING_RATE_ADAM)]
COLORES_VARIANTES = {"sgd": "#4C72B0", "adam": "#55A868"}


def crear_red(rng_modelo, clase_optimizador):
    """Misma arquitectura que crear_red() en cnn_fashion_mnist.py, con optimizador
    intercambiable en cada capa con pesos (Conv2D y Densa)."""
    return [
        CapaConv2D(3, 3, 1, 8, rng=rng_modelo, optimizador=clase_optimizador()), ActivacionLeakyReLU(), CapaMaxPool2D(2, 2),
        CapaConv2D(3, 3, 8, 16, rng=rng_modelo, optimizador=clase_optimizador()), ActivacionLeakyReLU(), CapaMaxPool2D(2, 2),
        CapaFlatten(),
        CapaDensa(400, 64, semilla_he=True, rng=rng_modelo, optimizador=clase_optimizador()), ActivacionLeakyReLU(),
        CapaDropout(0.3, rng=rng_modelo),
        CapaDensa(64, 10, semilla_he=True, rng=rng_modelo, optimizador=clase_optimizador()), ActivacionSoftmax(),
    ]


def entrenar(nombre, red, X_train, Y_train, X_val, Y_val_num, usar_augmentation, rng_aug, learning_rate,
             prob_flip=0.5, quiet=False):
    """Mismo bucle que entrenar() en cnn_fashion_mnist.py (ver su docstring para la explicación
    completa de augmentation/checkpoint), parametrizado en learning_rate para poder usar uno
    distinto por optimizador."""
    Y_val_onehot = np.eye(10)[Y_val_num]
    historial_loss_train, historial_loss_val, historial_acc_val = [], [], []
    capas_con_pesos = [c for c in red if isinstance(c, (CapaDensa, CapaConv2D))]
    mejor_loss_val = np.inf
    mejor_epoca = None
    mejores_pesos = None
    t0 = time.time()

    for epoch in range(EPOCHS_MAX):
        X_epoca = augmentar_lote(X_train, rng_aug, prob_flip=prob_flip) if usar_augmentation else X_train

        activacion = X_epoca
        for capa in red:
            activacion = capa.forward(activacion, entrenando=True)
        probs_train = activacion
        loss_train = -np.mean(np.sum(Y_train * np.log(probs_train + 1e-15), axis=1))
        historial_loss_train.append(loss_train)

        probs_val = predecir_cnn(red, X_val)
        loss_val = -np.mean(np.sum(Y_val_onehot * np.log(probs_val + 1e-15), axis=1))
        historial_loss_val.append(loss_val)
        acc_val = float(np.mean(np.argmax(probs_val, axis=1) == Y_val_num))
        historial_acc_val.append(acc_val)

        if loss_val < mejor_loss_val:
            mejor_loss_val = loss_val
            mejor_epoca = epoch
            mejores_pesos = [(c.W.copy(), c.b.copy()) for c in capas_con_pesos]

        gradiente = (probs_train - Y_train) / Y_train.shape[0]
        for capa in reversed(red[:-1]):
            gradiente = capa.backward(gradiente, learning_rate)

        if not quiet and epoch % 50 == 0:
            print(f"  [{nombre}] época {epoch}: loss_train={loss_train:.4f} "
                  f"loss_val={loss_val:.4f} acc_val={acc_val:.4f}")

        if epoch >= PACIENCIA_EARLY_STOP:
            loss_val_referencia = historial_loss_val[epoch - PACIENCIA_EARLY_STOP]
            mejora_relativa = (loss_val_referencia - loss_val) / loss_val_referencia
            if mejora_relativa < MEJORA_MINIMA_RELATIVA:
                if not quiet:
                    print(f"  [{nombre}] early stopping en la época {epoch + 1} ({time.time() - t0:.0f}s)")
                break
    else:
        if not quiet:
            print(f"  [{nombre}] completó las {EPOCHS_MAX} épocas sin activar el early stopping "
                  f"({time.time() - t0:.0f}s)")

    for capa, (W, b) in zip(capas_con_pesos, mejores_pesos):
        capa.W, capa.b = W, b
    if not quiet:
        print(f"  [{nombre}] pesos restaurados al mínimo de validación: época {mejor_epoca + 1} "
              f"(loss_val={mejor_loss_val:.6f}), frente a la época de corte {len(historial_loss_train)}")

    return historial_loss_train, historial_loss_val, historial_acc_val, mejor_epoca, mejor_loss_val


def graficar_curva_comparativa(curvas, ruta_salida, titulo):
    plt.figure(figsize=(9, 5))
    estilos = {"baseline": "-", "augmented": "--", "augmented_sin_flip": ":"}
    for nombre, historial in curvas.items():
        aug_variante, optimizador = nombre.rsplit("_", 1)
        plt.plot(historial, color=COLORES_VARIANTES[optimizador], linestyle=estilos[aug_variante],
                 label=nombre)
    plt.yscale("log")
    plt.title(titulo, fontweight="bold")
    plt.xlabel("Época")
    plt.ylabel("Pérdida de validación (escala log)")
    plt.legend(fontsize=8)
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150)
    plt.close()


def main(seed_split=42, seed_modelo=42, quiet=False, guardar_graficas=True, solo_augmentacion=None,
         solo_optimizador=None) -> dict:
    X_train, Y_train, Y_train_num, X_val, Y_val_num, X_test, Y_test_num = cargar_datos(seed_split, quiet=quiet)
    if not quiet:
        print(f"Entrenando con {len(X_train)} imágenes, validando con {len(X_val)}, "
              f"evaluando sobre {len(X_test)} de test...")

    variantes_aug = VARIANTES_AUMENTACION if solo_augmentacion is None else [
        v for v in VARIANTES_AUMENTACION if v[0] in solo_augmentacion
    ]
    optimizadores = OPTIMIZADORES if solo_optimizador is None else [
        o for o in OPTIMIZADORES if o[0] in solo_optimizador
    ]

    resultados = {}
    curvas = {}
    for nombre_aug, usar_aug, prob_flip in variantes_aug:
        for nombre_opt, clase_optimizador, learning_rate in optimizadores:
            nombre = f"{nombre_aug}_{nombre_opt}"
            if not quiet:
                print(f"\n=== Entrenando '{nombre}' (augmentation={usar_aug}, prob_flip={prob_flip}, "
                      f"lr={learning_rate}) ===")
            rng_modelo = np.random.default_rng(seed_modelo)  # mismos pesos iniciales en las 6 combinaciones
            red = crear_red(rng_modelo, clase_optimizador)
            rng_aug = np.random.default_rng(seed_modelo)
            hist_train, hist_val, hist_acc_val, mejor_epoca, mejor_loss_val = entrenar(
                nombre, red, X_train, Y_train, X_val, Y_val_num, usar_aug, rng_aug, learning_rate,
                prob_flip=prob_flip, quiet=quiet
            )
            accuracy, matriz_confusion = evaluar(red, X_test, Y_test_num)
            if not quiet:
                print(f"  [{nombre}] accuracy final en test: {accuracy:.4f}")

            if guardar_graficas:
                graficar_confusion(matriz_confusion, accuracy, f"Matriz de confusión test — {nombre}",
                                    RESULTS_DIR / f"confusion_matrix_{nombre}.png")

            resultados[nombre] = {
                "learning_rate": learning_rate,
                "epochs_entrenadas": len(hist_train),
                "epoca_mejor_val": mejor_epoca + 1,
                "accuracy_test": accuracy,
                "loss_train_final": float(hist_train[mejor_epoca]),
                "loss_val_final": float(mejor_loss_val),
                "matriz_confusion": matriz_confusion.tolist(),
                "historial_loss_val": hist_val,
            }
            curvas[nombre] = hist_val

    metrics = {
        "seed_split": seed_split,
        "seed_modelo": seed_modelo,
        "epochs_max_configuradas": EPOCHS_MAX,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "clases": NOMBRES_CLASES,
        "resultados": resultados,
    }

    if not guardar_graficas:
        return metrics

    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    graficar_curva_comparativa(curvas, RESULTS_DIR / "learning_curve_comparativa.png",
                                "SGD vs Adam — CNN Fashion-MNIST (muestra reducida, full-batch)")

    if not quiet:
        print("\n--- Resumen ---")
        for nombre, datos in resultados.items():
            print(f"{nombre:28s} accuracy={datos['accuracy_test']:.4f}  "
                  f"epocas={datos['epochs_entrenadas']}  epoca_mejor_val={datos['epoca_mejor_val']}")
        print(f"Resultados guardados en {RESULTS_DIR}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--solo-augmentacion", nargs="+", default=None,
                         choices=["baseline", "augmented", "augmented_sin_flip"])
    parser.add_argument("--solo-optimizador", nargs="+", default=None, choices=["sgd", "adam"])
    args = parser.parse_args()
    main(solo_augmentacion=args.solo_augmentacion, solo_optimizador=args.solo_optimizador,
         guardar_graficas=args.solo_optimizador is None and args.solo_augmentacion is None)
