"""
Esquema B: split de train/val/test FIJO (siempre la misma partición, seed_split=SEED_SPLIT_FIJO)
frente al esquema "Actual" que ya mide run_seed_sweep.py (split y modelo, cada uno con su
propia semilla, ambos variando de forma independiente). Solo aplica a 07 y 08 -- no hay
SEED_DATOS real en esos dos proyectos (MNIST/Fashion-MNIST se descargan, no se generan), así que
"fijar los datos" en la práctica es fijar el split, que es lo que decide qué imágenes concretas
caen en train/val/test. En 01-06 no hace falta (decisión tomada en conversación: allí ya existe
SEED_DATOS de verdad, y esta comparación concreta -- split fijo vs libre -- no aporta nada nuevo
sobre lo que ya mide el esquema actual).

Aísla una pregunta distinta a la de "Robustez frente a la semilla" de cada README: con
EXACTAMENTE la misma partición de datos siempre, ¿cuánto cambia el resultado solo por
reinicializar los pesos (y, donde aplica, el orden de los mini-batches/augmentation)?

Reutiliza cargar_modulo(), resumir() y graficar_curvas_superpuestas() de run_seed_sweep.py --
mismo motor de entrenamiento (sgd_vs_adam*.py), solo cambia qué semillas se le pasan desde
fuera. Escribe metrics_seed_sweep_esquemaB.json / seed_sweep_curvas_esquemaB.png en la misma
carpeta de resultados que el esquema actual, sin pisar sus archivos.

Uso:
    python run_seed_sweep_esquemaB.py                    # las 4 unidades, N por defecto
    python run_seed_sweep_esquemaB.py --n 5
    python run_seed_sweep_esquemaB.py --solo 08-fullbatch-sgd-vs-adam --n 3
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from run_seed_sweep import (
    N_SEEDS_DEFAULT,
    ROOT,
    UNIDADES,
    cargar_modulo,
    graficar_curvas_superpuestas,
    resumir,
)

SEED_SPLIT_FIJO = 42  # nunca varía en este esquema -- misma partición train/val/test siempre

# Solo las unidades sgd-vs-adam de 07/08 -- ver docstring del módulo para el porqué.
NOMBRES_ESQUEMA_B = {"07-fullbatch-sgd-vs-adam", "07-minibatch-sgd-vs-adam",
                      "08-fullbatch-sgd-vs-adam", "08-minibatch-sgd-vs-adam"}
UNIDADES_ESQUEMA_B = [u for u in UNIDADES if u[0] in NOMBRES_ESQUEMA_B]


def semillas_modelo(n, offset):
    """n semillas de modelo independientes -- solo esta varía en el esquema B."""
    rng = np.random.default_rng(2_000_000 + offset)
    return rng.integers(0, 2**31 - 1, size=n).tolist()


def ejecutar_unidad(nombre, ruta_script, carpeta_results, extractor, extractor_historial, n, offset):
    ruta_absoluta = ROOT / ruta_script
    modulo = cargar_modulo(f"esquemaB_{nombre.replace('-', '_')}", ruta_absoluta)

    semillas = semillas_modelo(n, offset)
    por_variante = {}
    por_variante_historial = {}
    t0 = time.time()
    for i, seed_modelo in enumerate(semillas):
        metrics = modulo.main(seed_split=SEED_SPLIT_FIJO, seed_modelo=int(seed_modelo),
                               quiet=True, guardar_graficas=False)
        resultado = extractor(metrics)
        for variante, valor in resultado.items():
            por_variante.setdefault(variante, []).append(valor)
        for variante, historial in extractor_historial(metrics).items():
            por_variante_historial.setdefault(variante, []).append(historial)
        print(f"  [{nombre} esquemaB] {i + 1}/{n} (seed_split={SEED_SPLIT_FIJO} fijo, "
              f"seed_modelo={seed_modelo}) -- {time.time() - t0:.0f}s acumulados")

    resumen = {variante: resumir(valores) for variante, valores in por_variante.items()}
    carpeta_proyecto = ROOT / ruta_script.split("/")[0]
    salida = carpeta_proyecto / carpeta_results / "metrics_seed_sweep_esquemaB.json"
    salida.write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")

    ruta_curvas = carpeta_proyecto / carpeta_results / "seed_sweep_curvas_esquemaB.png"
    graficar_curvas_superpuestas(por_variante_historial, f"{nombre} (esquema B, split fijo): "
                                  "pérdida de validación por semilla", ruta_curvas)

    print(f"[{nombre} esquemaB] guardado en {salida} y {ruta_curvas} ({time.time() - t0:.0f}s total)")
    return resumen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=N_SEEDS_DEFAULT)
    parser.add_argument("--solo", type=str, default=None, help="nombre de una sola unidad a ejecutar")
    args = parser.parse_args()

    unidades = UNIDADES_ESQUEMA_B if args.solo is None else [u for u in UNIDADES_ESQUEMA_B if u[0] == args.solo]
    if not unidades:
        print(f"Unidad '{args.solo}' no encontrada en esquema B. Disponibles: "
              f"{[u[0] for u in UNIDADES_ESQUEMA_B]}")
        sys.exit(1)

    for offset, (nombre, ruta_script, carpeta_results, extractor, extractor_historial) in enumerate(unidades):
        print(f"\n=== Unidad (esquema B): {nombre} (N={args.n}) ===")
        ejecutar_unidad(nombre, ruta_script, carpeta_results, extractor, extractor_historial, args.n, offset)


if __name__ == "__main__":
    main()
