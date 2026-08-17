"""
BatchNorm sobre la configuración de referencia (Adam, mini-batch, variante `baseline`) -- ver
README raíz para por qué esta es la configuración de referencia para todo lo nuevo del repo.

Compara la arquitectura de siempre contra la misma arquitectura con una CapaBatchNorm2D tras
cada CapaConv2D y una CapaBatchNorm tras la primera capa densa (Conv/Dense -> BatchNorm ->
LeakyReLU, el orden del paper original de Ioffe & Szegedy, 2015). No toca
cnn_fashion_mnist_full.py ni sgd_vs_adam_full.py.

Uso: python batchnorm.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from capas import ActivacionLeakyReLU, ActivacionSoftmax, CapaBatchNorm, CapaDensa, OptimizadorAdam
from capas_cnn import CapaBatchNorm2D, CapaConv2D, CapaDropout, CapaFlatten, CapaMaxPool2D, predecir_cnn

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
from sgd_vs_adam_full import LEARNING_RATE_ADAM, crear_red as crear_red_sin_batchnorm

RESULTS_DIR = Path(__file__).parent / "results_batchnorm"
RESULTS_DIR.mkdir(exist_ok=True)

COLORES_VARIANTES = {"adam_sin_batchnorm": "#4C72B0", "adam_con_batchnorm": "#55A868"}


def crear_red_con_batchnorm(rng_modelo):
    return [
        CapaConv2D(3, 3, 1, 8, rng=rng_modelo, optimizador=OptimizadorAdam()),
        CapaBatchNorm2D(canales=8, optimizador=OptimizadorAdam()),
        ActivacionLeakyReLU(), CapaMaxPool2D(2, 2),
        CapaConv2D(3, 3, 8, 16, rng=rng_modelo, optimizador=OptimizadorAdam()),
        CapaBatchNorm2D(canales=16, optimizador=OptimizadorAdam()),
        ActivacionLeakyReLU(), CapaMaxPool2D(2, 2),
        CapaFlatten(),
        CapaDensa(400, 64, semilla_he=True, rng=rng_modelo, optimizador=OptimizadorAdam()),
        CapaBatchNorm(dim=64, optimizador=OptimizadorAdam()),
        ActivacionLeakyReLU(),
        CapaDropout(0.3, rng=rng_modelo),
        CapaDensa(64, 10, semilla_he=True, rng=rng_modelo, optimizador=OptimizadorAdam()),
        ActivacionSoftmax(),
    ]


def _copiar_parametros(capa):
    if isinstance(capa, (CapaDensa, CapaConv2D)):
        return (capa.W.copy(), capa.b.copy())
    return (capa.gamma.copy(), capa.beta.copy())


def _restaurar_parametros(capa, parametros):
    p1, p2 = parametros
    if isinstance(capa, (CapaDensa, CapaConv2D)):
        capa.W, capa.b = p1, p2
    else:
        capa.gamma, capa.beta = p1, p2


def entrenar(nombre, red, X_train, Y_train, X_val, Y_val_num, rng_batches, quiet=False):
    """Igual que entrenar() en sgd_vs_adam_full.py (sin augmentation, variante baseline), con
    el checkpoint de mejor validación extendido a CapaBatchNorm/CapaBatchNorm2D además de
    CapaDensa/CapaConv2D -- si no se restauraran gamma/beta junto con W/b, la red quedaría con
    una mezcla inconsistente de pesos del mínimo de validación y normalización de la última
    época."""
    Y_val_onehot = np.eye(10)[Y_val_num]
    historial_loss_train, historial_loss_val = [], []
    capas_con_parametros = [c for c in red if isinstance(c, (CapaDensa, CapaConv2D, CapaBatchNorm, CapaBatchNorm2D))]
    mejor_loss_val = np.inf
    mejor_epoca = None
    mejores_parametros = None

    for epoch in range(EPOCHS_MAX):
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
                gradiente = capa.backward(gradiente, LEARNING_RATE_ADAM)

        loss_train = loss_train_acumulada / n_vistas
        historial_loss_train.append(loss_train)

        probs_val = predecir_cnn(red, X_val)
        loss_val = -np.mean(np.sum(Y_val_onehot * np.log(probs_val + 1e-15), axis=1))
        historial_loss_val.append(loss_val)

        if loss_val < mejor_loss_val:
            mejor_loss_val = loss_val
            mejor_epoca = epoch
            mejores_parametros = [_copiar_parametros(c) for c in capas_con_parametros]

        if not quiet:
            print(f"  [{nombre}] época {epoch}: loss_val={loss_val:.4f}")

        if epoch >= PACIENCIA_EARLY_STOP:
            loss_val_referencia = historial_loss_val[epoch - PACIENCIA_EARLY_STOP]
            mejora_relativa = (loss_val_referencia - loss_val) / loss_val_referencia
            if mejora_relativa < MEJORA_MINIMA_RELATIVA:
                if not quiet:
                    print(f"  [{nombre}] early stopping en la época {epoch + 1}")
                break
    else:
        if not quiet:
            print(f"  [{nombre}] completó las {EPOCHS_MAX} épocas sin activar el early stopping")

    for capa, parametros in zip(capas_con_parametros, mejores_parametros):
        _restaurar_parametros(capa, parametros)
    if not quiet:
        print(f"  [{nombre}] pesos restaurados al mínimo de validación: época {mejor_epoca + 1} "
              f"(loss_val={mejor_loss_val:.6f}), frente a la época de corte {len(historial_loss_train)}")

    return historial_loss_train, historial_loss_val, mejor_epoca, mejor_loss_val


def graficar_curva_comparativa(curvas_val, ruta_salida):
    plt.figure(figsize=(8, 4.5))
    for nombre, historial in curvas_val.items():
        plt.plot(historial, color=COLORES_VARIANTES[nombre], label=nombre)
    plt.yscale("log")
    plt.title("BatchNorm vs sin BatchNorm (Adam, mini-batch, CNN baseline)", fontweight="bold")
    plt.xlabel("Época")
    plt.ylabel("Pérdida de validación (escala log)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150)
    plt.close()


def main(seed_split=42, seed_modelo=42, quiet=False, guardar_graficas=True) -> dict:
    X_train, Y_train, Y_train_num, X_val, Y_val_num, X_test, Y_test_num = cargar_datos(seed_split, quiet=quiet)

    resultados = {}
    curvas_val = {}
    for nombre, constructor in [
        ("adam_sin_batchnorm", lambda rng: crear_red_sin_batchnorm(rng, OptimizadorAdam)),
        ("adam_con_batchnorm", crear_red_con_batchnorm),
    ]:
        if not quiet:
            print(f"\n=== Entrenando '{nombre}' ===")
        rng_modelo = np.random.default_rng(seed_modelo)
        red = constructor(rng_modelo)
        rng_batches = np.random.default_rng(seed_modelo)
        hist_train, hist_val, mejor_epoca, mejor_loss_val = entrenar(
            nombre, red, X_train, Y_train, X_val, Y_val_num, rng_batches, quiet=quiet
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
        }
        curvas_val[nombre] = hist_val

    metrics = {
        "seed_split": seed_split,
        "seed_modelo": seed_modelo,
        "learning_rate": LEARNING_RATE_ADAM,
        "variante_augmentation": "baseline",
        "resultados": resultados,
    }

    if not guardar_graficas:
        return metrics

    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    graficar_curva_comparativa(curvas_val, RESULTS_DIR / "batchnorm_comparativa.png")

    if not quiet:
        print("\n--- Resumen ---")
        for nombre, datos in resultados.items():
            print(f"{nombre:20s} accuracy={datos['accuracy_test']:.4f}  "
                  f"epocas={datos['epochs_entrenadas']}  epoca_mejor_val={datos['epoca_mejor_val']}")
        print(f"Resultados guardados en {RESULTS_DIR}")

    return metrics


if __name__ == "__main__":
    main()
