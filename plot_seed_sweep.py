"""
Genera una gráfica por proyecto a partir de su results/metrics_seed_sweep.json (ver
run_seed_sweep.py) -- un dot plot de los N valores obtenidos con semillas independientes, en
vez de una tabla o un heatmap: con semillas aleatorias no hay ningún orden con sentido entre
"semilla 3" y "semilla 4" (ver README raíz), así que mostrar cada valor individual como un punto
es más honesto que un histograma agrupado o una barra con error -- un solo vistazo deja ver si
la dispersión es uniforme o si hay valores atípicos (como el colapso de 03-tipos-clientes),
que una barra con error estándar escondería dentro del intervalo.

Uso: python plot_seed_sweep.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent

# (nombre_json, xlabel, título, ruta_salida)
UNIDADES_SIMPLES = [
    ("01-compuertas-logicas-xor/results/metrics_seed_sweep.json", "Loss final",
     "01 — XOR: loss final sobre 20 semillas", "01-compuertas-logicas-xor/results/seed_sweep.png"),
    ("02-celsius-fahrenheit/results/metrics_seed_sweep.json", "MAE en test (°F)",
     "02 — Celsius→Fahrenheit: MAE sobre 20 semillas", "02-celsius-fahrenheit/results/seed_sweep.png"),
    ("03-tipos-clientes/results/metrics_seed_sweep.json", "Accuracy en test",
     "03 — Tipos de cliente: accuracy sobre 20 semillas", "03-tipos-clientes/results/seed_sweep.png"),
    ("04-prediccion-temperatura-dia-noche/results/metrics_seed_sweep.json", "MAE en test (°C)",
     "04 — Temperatura día/noche: MAE sobre 20 semillas", "04-prediccion-temperatura-dia-noche/results/seed_sweep.png"),
    ("05-precio-casas/results/metrics_seed_sweep.json", "MAE en test (€)",
     "05 — Precio de una casa: MAE sobre 20 semillas", "05-precio-casas/results/seed_sweep.png"),
    ("06-zonas-espirales/results/metrics_seed_sweep.json", "Accuracy en test",
     "06 — Zonas de espirales: accuracy sobre 20 semillas", "06-zonas-espirales/results/seed_sweep.png"),
]

COLOR_UNICO = "#4C72B0"
COLORES_VARIANTES = {"baseline": "#4C72B0", "augmented": "#55A868", "augmented_sin_flip": "#C44E52",
                      "full-batch": "#4C72B0", "mini-batch": "#DD8452"}


def _formato_valor(v):
    return f"{v:.4g}" if abs(v) < 1000 else f"{v:,.0f}"


def dotplot_simple(valores, xlabel, titulo, ruta_salida):
    valores = np.array(valores)
    rng = np.random.default_rng(0)
    y_jitter = rng.uniform(-0.15, 0.15, size=len(valores))
    media = valores.mean()

    fig, ax = plt.subplots(figsize=(6.5, 2.3))
    ax.scatter(valores, y_jitter, color=COLOR_UNICO, edgecolors="black", s=70, zorder=3, alpha=0.85,
               linewidths=0.8)
    ax.axvline(media, color="#333333", linestyle="--", linewidth=1.3, zorder=2)
    ax.text(media, 0.30, f"media = {_formato_valor(media)}", ha="center", fontsize=9, color="#333333")
    ax.set_yticks([])
    ax.set_ylim(-0.4, 0.4)
    ax.set_xlabel(xlabel)
    ax.set_title(titulo, fontweight="bold", fontsize=11)
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    Path(ruta_salida).parent.mkdir(exist_ok=True)
    plt.savefig(ruta_salida, dpi=150)
    plt.close()


def dotplot_variantes(series, xlabel, titulo, ruta_salida):
    """series: lista de (nombre, valores) -- una fila por variante, mismo eje X."""
    fig, ax = plt.subplots(figsize=(8.5, 0.9 * len(series) + 1.0))
    rng = np.random.default_rng(0)
    for i, (nombre, valores) in enumerate(series):
        valores = np.array(valores)
        y = len(series) - 1 - i
        y_jitter = y + rng.uniform(-0.15, 0.15, size=len(valores))
        color = COLORES_VARIANTES.get(nombre, "#4C72B0")
        ax.scatter(valores, y_jitter, color=color, edgecolors="black", s=60, zorder=3, alpha=0.85,
                   linewidths=0.8, label=nombre)
        media = valores.mean()
        ax.plot([media, media], [y - 0.28, y + 0.28], color="#333333", linestyle="--", linewidth=1.3, zorder=2)
        ax.text(media, y + 0.34, f"media = {_formato_valor(media)}", ha="center", fontsize=8, color="#333333")
    ax.set_yticks(range(len(series)))
    ax.set_yticklabels([s[0] for s in reversed(series)])
    ax.set_xlabel(xlabel)
    ax.set_title(titulo, fontweight="bold", fontsize=11)
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    Path(ruta_salida).parent.mkdir(exist_ok=True)
    plt.savefig(ruta_salida, dpi=150)
    plt.close()


def main():
    for ruta_json, xlabel, titulo, ruta_salida in UNIDADES_SIMPLES:
        ruta_json_abs = ROOT / ruta_json
        if not ruta_json_abs.exists():
            print(f"[saltado] {ruta_json} no existe todavía")
            continue
        datos = json.loads(ruta_json_abs.read_text(encoding="utf-8"))
        valores = datos["unico"]["valores"]
        dotplot_simple(valores, xlabel, titulo, ROOT / ruta_salida)
        print(f"[ok] {ruta_salida}")

    # 07: dos variantes en carpetas distintas (results / results_full), un gráfico cada una
    for ruta_json, nombre_var, xlabel, titulo, ruta_salida in [
        ("07-reconocimiento-digitos/results/metrics_seed_sweep.json", "full-batch", "Accuracy en test",
         "07 — Dígitos (muestra reducida, full-batch): accuracy sobre 20 semillas",
         "07-reconocimiento-digitos/results/seed_sweep.png"),
        ("07-reconocimiento-digitos/results_full/metrics_seed_sweep.json", "mini-batch", "Accuracy en test",
         "07 — Dígitos (dataset completo, mini-batch): accuracy sobre 20 semillas",
         "07-reconocimiento-digitos/results_full/seed_sweep.png"),
    ]:
        ruta_json_abs = ROOT / ruta_json
        if not ruta_json_abs.exists():
            print(f"[saltado] {ruta_json} no existe todavía")
            continue
        datos = json.loads(ruta_json_abs.read_text(encoding="utf-8"))
        valores = datos["unico"]["valores"]
        dotplot_simple(valores, xlabel, titulo, ROOT / ruta_salida)
        print(f"[ok] {ruta_salida}")

    # 08: 3 variantes por escala, un gráfico combinado por escala
    for ruta_json, escala, ruta_salida in [
        ("08-cnn-fashion-mnist/results/metrics_seed_sweep.json", "muestra reducida, full-batch",
         "08-cnn-fashion-mnist/results/seed_sweep.png"),
        ("08-cnn-fashion-mnist/results_full/metrics_seed_sweep.json", "dataset completo, mini-batch",
         "08-cnn-fashion-mnist/results_full/seed_sweep.png"),
    ]:
        ruta_json_abs = ROOT / ruta_json
        if not ruta_json_abs.exists():
            print(f"[saltado] {ruta_json} no existe todavía (barrido de 08 en curso)")
            continue
        datos = json.loads(ruta_json_abs.read_text(encoding="utf-8"))
        orden = ["baseline", "augmented", "augmented_sin_flip"]
        series = [(nombre, datos[nombre]["valores"]) for nombre in orden if nombre in datos]
        dotplot_variantes(series, "Accuracy en test",
                           f"08 — CNN Fashion-MNIST ({escala})",
                           ROOT / ruta_salida)
        print(f"[ok] {ruta_salida}")


if __name__ == "__main__":
    main()
