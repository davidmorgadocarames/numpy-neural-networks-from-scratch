"""
Test de permutación: ¿la red aprendió el ciclo día/noche real, o llegaría a un error parecido
con cualquier ruido que tuviera la misma media y varianza? Es una pregunta distinta a la de
"Robustez frente a la semilla" (run_seed_sweep.py) -- esa varía la inicialización de pesos sobre
los MISMOS datos para medir si el entrenamiento es estable; esta mantiene la inicialización
FIJA y varía el orden de los datos para medir si el patrón que explota la red es genuino.

Metodología: se toma la serie de 192 temperaturas reales y se baraja por completo (mismos
valores, misma media/varianza, pero sin ciclo día/noche) N veces con órdenes distintos. Cada
barajado se entrena con la misma receta exacta que el proyecto original (misma arquitectura,
mismo split 60/20/20, misma normalización por min/max de train, mismo early stopping) y SIEMPRE
con la misma seed_modelo -- así la única diferencia entre repeticiones es el orden de los datos,
no el punto de partida del entrenamiento. Si el MAE real queda muy por debajo de la distribución
de MAEs barajados, el error bajo no es un artefacto (autocorrelación falsa por el solapamiento
de ventanas, ruido aprovechado por casualidad): la red está explotando el ciclo día/noche
genuino, porque es lo único que las versiones barajadas no tienen.

Es un barajado GLOBAL (no por bloques): destruye toda la estructura temporal, no solo la de
largo plazo. Una alternativa más conservadora sería un block-shuffle que preserve tramos cortos
intactos: es una prueba más laxa, útil si se quisiera acotar a "la red no aprovecha solo
correlación de muy corto plazo" en vez de "la red no aprovecha ninguna estructura temporal".

Uso:
    python permutation_test.py              # N=1000 por defecto
    python permutation_test.py --n 50        # para probar rápido cuánto tarda
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
from capas import ActivacionLeakyReLU, CapaDensa, predecir

from temperatura_dia_noche import (
    AMPLITUD_C,
    MEJORA_MINIMA_RELATIVA,
    N_DIAS,
    PACIENCIA_EARLY_STOP,
    PUNTOS_POR_DIA,
    RUIDO_STD_C,
    SEED_DATOS,
    TEMP_MEDIA_C,
    VENTANA,
)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SEED_MODELO_FIJO = 42  # misma seed_modelo que el run canónico -- fija en TODAS las repeticiones
SEED_PERMUTACION = 2024  # gobierna solo el orden de los barajados, independiente de SEED_MODELO_FIJO
N_PERMUTACIONES_DEFAULT = 1000


def generar_serie_real():
    """Misma generación que temperatura_dia_noche.py, con SEED_DATOS -- reproduce exactamente
    la misma serie de 192 temperaturas."""
    np.random.seed(SEED_DATOS)
    n_puntos = N_DIAS * PUNTOS_POR_DIA
    dias = np.linspace(0, N_DIAS, n_puntos, endpoint=False)
    fase = 2 * np.pi * dias
    return TEMP_MEDIA_C - AMPLITUD_C * np.cos(fase) + np.random.normal(0, RUIDO_STD_C, n_puntos)


def entrenar_evaluar(serie_temperatura, seed_modelo):
    """Misma receta que main() de temperatura_dia_noche.py (ventana deslizante, split 60/20/20
    cronológico por posición, normalización con min/max de train, red 3->6->1, early stopping
    con checkpoint del mínimo de validación) pero recibiendo la serie ya barajada (o la real)
    como parámetro en vez de generarla -- así ambos casos comparten exactamente el mismo código
    de entrenamiento y solo cambia el ORDEN de los valores de entrada."""
    n_puntos = len(serie_temperatura)
    n = n_puntos - VENTANA
    split_train = int(n * 0.6)
    split_val = int(n * 0.8)
    raw_fin_train = split_train + VENTANA

    t_min = np.min(serie_temperatura[:raw_fin_train])
    t_max = np.max(serie_temperatura[:raw_fin_train])
    serie_norm = (serie_temperatura - t_min) / (t_max - t_min)

    X_lista, Y_lista = [], []
    for i in range(len(serie_norm) - VENTANA):
        X_lista.append(serie_norm[i : i + VENTANA])
        Y_lista.append(serie_norm[i + VENTANA])
    X = np.array(X_lista)
    Y = np.array(Y_lista).reshape(-1, 1)

    X_train, X_val, X_test = X[:split_train], X[split_train:split_val], X[split_val:]
    Y_train, Y_val, Y_test = Y[:split_train], Y[split_train:split_val], Y[split_val:]

    rng_modelo = np.random.default_rng(seed_modelo)
    red = [
        CapaDensa(dim_entrada=VENTANA, dim_salida=6, semilla_he=False, rng=rng_modelo),
        ActivacionLeakyReLU(),
        CapaDensa(dim_entrada=6, dim_salida=1, semilla_he=False, rng=rng_modelo),
    ]

    learning_rate = 0.1
    epochs = 3000
    historial_loss_val = []
    mejor_loss_val = np.inf
    mejor_epoca = None
    mejores_pesos = None

    for epoch in range(epochs):
        activacion = X_train
        for capa in red:
            activacion = capa.forward(activacion)
        A2_train = activacion

        A2_val = predecir(red, X_val)
        loss_val = np.mean((A2_val - Y_val) ** 2)
        historial_loss_val.append(loss_val)

        if loss_val < mejor_loss_val:
            mejor_loss_val = loss_val
            mejor_epoca = epoch
            mejores_pesos = [(c.W.copy(), c.b.copy()) for c in red if isinstance(c, CapaDensa)]

        gradiente = 2 * (A2_train - Y_train) / Y_train.shape[0]
        for capa in reversed(red):
            gradiente = capa.backward(gradiente, learning_rate)

        if epoch >= PACIENCIA_EARLY_STOP:
            loss_val_referencia = historial_loss_val[epoch - PACIENCIA_EARLY_STOP]
            mejora_relativa = (loss_val_referencia - loss_val) / loss_val_referencia
            if mejora_relativa < MEJORA_MINIMA_RELATIVA:
                break

    for capa, (W, b) in zip([c for c in red if isinstance(c, CapaDensa)], mejores_pesos):
        capa.W, capa.b = W, b

    A2_test = predecir(red, X_test)
    Y_test_c = Y_test * (t_max - t_min) + t_min
    Y_pred_c = A2_test * (t_max - t_min) + t_min
    return float(np.mean(np.abs(Y_pred_c - Y_test_c)))


def graficar_histograma(mae_permutados, mae_real, p_valor, ruta_salida):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(mae_permutados, bins=40, color="#4C72B0", edgecolor="black", alpha=0.85,
            label=f"MAE con datos barajados (N={len(mae_permutados)})")
    ax.axvline(mae_real, color="#C44E52", linestyle="--", linewidth=2,
                label=f"MAE con datos reales = {mae_real:.2f} °C")
    cota_p = 1 / (len(mae_permutados) + 1)
    texto_p = f"p < {cota_p:.3g}" if p_valor <= cota_p else f"p = {p_valor:.3g}"
    ax.text(mae_real, ax.get_ylim()[1] * 0.92, f"  {texto_p}", color="#C44E52", fontsize=9, va="top")
    ax.set_xlabel("MAE en test (°C)")
    ax.set_ylabel("Nº de barajados")
    ax.set_title("Test de permutación: ¿es real el ciclo día/noche aprendido?",
                 fontweight="bold", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150)
    plt.close()


def main(n_permutaciones=N_PERMUTACIONES_DEFAULT):
    serie_real = generar_serie_real()

    print(f"Entrenando sobre datos reales (seed_modelo={SEED_MODELO_FIJO})...")
    mae_real = entrenar_evaluar(serie_real, SEED_MODELO_FIJO)
    print(f"MAE en test, datos reales: {mae_real:.4f} °C")

    rng_permutacion = np.random.default_rng(SEED_PERMUTACION)
    mae_permutados = []
    t0 = time.time()
    for i in range(n_permutaciones):
        serie_barajada = rng_permutacion.permutation(serie_real)
        mae = entrenar_evaluar(serie_barajada, SEED_MODELO_FIJO)
        mae_permutados.append(mae)
        if (i + 1) % 50 == 0 or (i + 1) == n_permutaciones:
            print(f"  {i + 1}/{n_permutaciones} barajados -- {time.time() - t0:.0f}s acumulados")

    mae_permutados = np.array(mae_permutados)
    n_iguala_o_mejora = int(np.sum(mae_permutados <= mae_real))
    p_valor = (n_iguala_o_mejora + 1) / (n_permutaciones + 1)

    resumen = {
        "seed_modelo_fijo": SEED_MODELO_FIJO,
        "seed_permutacion": SEED_PERMUTACION,
        "n_permutaciones": n_permutaciones,
        "mae_test_real_celsius": mae_real,
        "mae_test_permutado": {
            "media": float(np.mean(mae_permutados)),
            "desviacion_tipica": float(np.std(mae_permutados, ddof=1)),
            "min": float(np.min(mae_permutados)),
            "max": float(np.max(mae_permutados)),
            "valores": mae_permutados.tolist(),
        },
        "n_permutaciones_que_igualan_o_superan_lo_real": n_iguala_o_mejora,
        "p_valor_empirico": p_valor,
    }

    salida_json = RESULTS_DIR / "metrics_permutation_test.json"
    salida_json.write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")

    ruta_grafica = RESULTS_DIR / "permutation_test.png"
    graficar_histograma(mae_permutados, mae_real, p_valor, ruta_grafica)

    print(f"\nMAE real: {mae_real:.4f} °C")
    print(f"MAE barajado: media={resumen['mae_test_permutado']['media']:.4f}, "
          f"min={resumen['mae_test_permutado']['min']:.4f}, "
          f"max={resumen['mae_test_permutado']['max']:.4f}")
    print(f"p-valor empírico: {p_valor:.4g} ({n_iguala_o_mejora}/{n_permutaciones} barajados "
          f"igualan o superan el resultado real)")
    print(f"Guardado en {salida_json} y {ruta_grafica}")

    return resumen


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=N_PERMUTACIONES_DEFAULT)
    args = parser.parse_args()
    main(n_permutaciones=args.n)
