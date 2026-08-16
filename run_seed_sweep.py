"""
Barrido multi-semilla sobre los 8 proyectos del repo (13 unidades: los 6 proyectos de capas
densas sin variantes, las 2 variantes de 07-reconocimiento-digitos, las 2 variantes de
08-cnn-fashion-mnist -- cada llamada a estas últimas ya entrena las 3 sub-variantes de
augmentation internamente, así que 2 llamadas cubren las 6 combinaciones -- y 3 unidades
"sgd-vs-adam" (06, y las 2 escalas de 07) que entrenan las 2 sub-variantes de optimizador sobre
el mismo problema de su unidad hermana sin Adam).

Para cada unidad, se ejecuta N veces con seed_split y seed_modelo sorteados de forma
INDEPENDIENTE en cada ejecución (no atados entre sí, no en rejilla) -- es una estimación Monte
Carlo estándar de la media y la varianza del resultado final frente a la semilla, no una
técnica con nombre propio (ver conversación/README para la discusión de por qué). La semilla de
datos (SEED_DATOS, definida dentro de cada script) se queda siempre fija -- ver README raíz para
por qué no tendría sentido variarla también.

Escribe results/metrics_seed_sweep.json en la carpeta de cada proyecto (o results_full/ /
results_sgd_vs_adam/ donde corresponda), sin tocar los results/metrics.json del run canónico ya
documentado.

Uso:
    python run_seed_sweep.py                    # las 13 unidades, N por defecto
    python run_seed_sweep.py --n 5               # N=5 en todas
    python run_seed_sweep.py --solo 08-cnn-fullbatch --n 3
"""

import argparse
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
N_SEEDS_DEFAULT = 20
COLORES_VARIANTES = {"baseline": "#4C72B0", "augmented": "#55A868", "augmented_sin_flip": "#C44E52",
                      "unico": "#4C72B0", "sgd": "#4C72B0", "adam": "#55A868"}


def cargar_modulo(nombre, ruta):
    # python script.py añade automáticamente la carpeta del script a sys.path (para imports
    # entre hermanos, p. ej. cnn_fashion_mnist.py -> capas_cnn.py); importlib no lo hace solo,
    # así que hay que replicarlo a mano antes de ejecutar el módulo.
    carpeta = str(ruta.parent)
    if carpeta not in sys.path:
        sys.path.insert(0, carpeta)
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def semillas_independientes(n, offset):
    """n pares (seed_split, seed_modelo) sorteados de forma independiente entre sí -- simple
    muestreo Monte Carlo del espacio conjunto, no una rejilla ni semillas atadas."""
    rng = np.random.default_rng(1_000_000 + offset)
    splits = rng.integers(0, 2**31 - 1, size=n)
    modelos = rng.integers(0, 2**31 - 1, size=n)
    return list(zip(splits.tolist(), modelos.tolist()))


def metrica_simple(campo):
    def extractor(metrics):
        return {"unico": metrics[campo]}
    return extractor


def metrica_variantes(metrics):
    """Para unidades con varias variantes entrenadas en la misma llamada a main() (08:
    baseline/augmented/augmented_sin_flip; 06-sgd-vs-adam: sgd/adam) -- todas comparten el
    mismo esquema metrics["resultados"][nombre]["accuracy_test"]."""
    return {nombre: datos["accuracy_test"] for nombre, datos in metrics["resultados"].items()}


def historial_simple(campo):
    def extractor(metrics):
        return {"unico": metrics[campo]}
    return extractor


def historial_variantes(metrics):
    return {nombre: datos["historial_loss_val"] for nombre, datos in metrics["resultados"].items()}


# (nombre_unidad, ruta_script relativa a ROOT, carpeta_results relativa al proyecto,
#  extractor de métrica final, extractor de historial de pérdida de validación)
UNIDADES = [
    ("01-xor", "01-compuertas-logicas-xor/xor_gate.py", "results",
     metrica_simple("loss_final"), historial_simple("historial_loss")),
    ("02-celsius", "02-celsius-fahrenheit/celsius_fahrenheit.py", "results",
     metrica_simple("mae_test_grados_F"), historial_simple("historial_loss_val")),
    ("03-clientes", "03-tipos-clientes/customer_classifier.py", "results",
     metrica_simple("accuracy_test"), historial_simple("historial_loss_val")),
    ("04-temperatura", "04-prediccion-temperatura-dia-noche/temperatura_dia_noche.py", "results",
     metrica_simple("mae_test_celsius"), historial_simple("historial_loss_val")),
    ("05-casas", "05-precio-casas/house_price.py", "results",
     metrica_simple("mae_test_euros"), historial_simple("historial_loss_val")),
    ("06-espirales", "06-zonas-espirales/spiral_classifier.py", "results",
     metrica_simple("accuracy_test"), historial_simple("historial_loss_val")),
    ("06-sgd-vs-adam", "06-zonas-espirales/sgd_vs_adam.py", "results_sgd_vs_adam",
     metrica_variantes, historial_variantes),
    ("07-digitos-fullbatch", "07-reconocimiento-digitos/digit_classifier.py", "results",
     metrica_simple("accuracy_test"), historial_simple("historial_loss_val")),
    ("07-fullbatch-sgd-vs-adam", "07-reconocimiento-digitos/sgd_vs_adam.py", "results_sgd_vs_adam",
     metrica_variantes, historial_variantes),
    ("07-digitos-minibatch", "07-reconocimiento-digitos/digit_classifier_full.py", "results_full",
     metrica_simple("accuracy_test"), historial_simple("historial_loss_val")),
    ("07-minibatch-sgd-vs-adam", "07-reconocimiento-digitos/sgd_vs_adam_full.py", "results_full_sgd_vs_adam",
     metrica_variantes, historial_variantes),
    ("08-cnn-fullbatch", "08-cnn-fashion-mnist/cnn_fashion_mnist.py", "results",
     metrica_variantes, historial_variantes),
    ("08-cnn-minibatch", "08-cnn-fashion-mnist/cnn_fashion_mnist_full.py", "results_full",
     metrica_variantes, historial_variantes),
]


def graficar_curvas_superpuestas(por_variante_historial, titulo, ruta_salida):
    """Una línea por semilla, pérdida de validación en escala log (eje Y) vs épocas (eje X).
    Con semillas independientes no hay relación de orden entre "semilla 3" y "semilla 4" -- lo
    que sí es informativo es la FORMA de cada curva (dónde converge, si corta antes, si se
    dispara) superpuesta con el resto, no una tabla de valores por época."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for variante, historiales in por_variante_historial.items():
        color = COLORES_VARIANTES.get(variante, "#4C72B0")
        for i, historial in enumerate(historiales):
            ax.plot(range(1, len(historial) + 1), historial, color=color, alpha=0.55, linewidth=1.3,
                    label=variante if i == 0 else None)
    ax.set_yscale("log")
    ax.set_xlabel("Época")
    ax.set_ylabel("Pérdida de validación (escala log)")
    ax.set_title(titulo, fontweight="bold", fontsize=11)
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    if len(por_variante_historial) > 1:
        ax.legend(fontsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    Path(ruta_salida).parent.mkdir(exist_ok=True)
    plt.savefig(ruta_salida, dpi=150)
    plt.close()


def resumir(valores):
    return {
        "n": len(valores),
        "media": statistics.fmean(valores),
        "desviacion_tipica": statistics.stdev(valores) if len(valores) > 1 else 0.0,
        "min": min(valores),
        "max": max(valores),
        "valores": valores,
    }


def ejecutar_unidad(nombre, ruta_script, carpeta_results, extractor, extractor_historial, n, offset):
    ruta_absoluta = ROOT / ruta_script
    modulo = cargar_modulo(f"sweep_{nombre.replace('-', '_')}", ruta_absoluta)

    pares = semillas_independientes(n, offset)
    por_variante = {}
    por_variante_historial = {}
    t0 = time.time()
    for i, (seed_split, seed_modelo) in enumerate(pares):
        metrics = modulo.main(seed_split=int(seed_split), seed_modelo=int(seed_modelo),
                               quiet=True, guardar_graficas=False)
        resultado = extractor(metrics)
        for variante, valor in resultado.items():
            por_variante.setdefault(variante, []).append(valor)
        for variante, historial in extractor_historial(metrics).items():
            por_variante_historial.setdefault(variante, []).append(historial)
        print(f"  [{nombre}] {i + 1}/{n} (seed_split={seed_split}, seed_modelo={seed_modelo}) -- "
              f"{time.time() - t0:.0f}s acumulados")

    resumen = {variante: resumir(valores) for variante, valores in por_variante.items()}
    carpeta_proyecto = ROOT / ruta_script.split("/")[0]
    salida = carpeta_proyecto / carpeta_results / "metrics_seed_sweep.json"
    salida.write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")

    ruta_curvas = carpeta_proyecto / carpeta_results / "seed_sweep_curvas.png"
    graficar_curvas_superpuestas(por_variante_historial, f"{nombre}: pérdida de validación por semilla",
                                  ruta_curvas)

    print(f"[{nombre}] guardado en {salida} y {ruta_curvas} ({time.time() - t0:.0f}s total)")
    return resumen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=N_SEEDS_DEFAULT)
    parser.add_argument("--solo", type=str, default=None, help="nombre de una sola unidad a ejecutar")
    args = parser.parse_args()

    unidades = UNIDADES if args.solo is None else [u for u in UNIDADES if u[0] == args.solo]
    if not unidades:
        print(f"Unidad '{args.solo}' no encontrada. Unidades disponibles: {[u[0] for u in UNIDADES]}")
        sys.exit(1)

    for offset, (nombre, ruta_script, carpeta_results, extractor, extractor_historial) in enumerate(unidades):
        print(f"\n=== Unidad: {nombre} (N={args.n}) ===")
        ejecutar_unidad(nombre, ruta_script, carpeta_results, extractor, extractor_historial, args.n, offset)


if __name__ == "__main__":
    main()
