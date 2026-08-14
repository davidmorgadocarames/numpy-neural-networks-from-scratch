"""
Predicción del precio de una casa (NumPy puro) a partir de 2 variables: metros cuadrados y
número de habitaciones. Red modular (Densa -> LeakyReLU -> Densa lineal) entrenada con un
split train/validación/test para poder detectar overfitting y decidir cuándo parar sin tocar
el conjunto de test: el early stopping compara el error en train contra el de VALIDACIÓN
época a época -- si el error de validación empezara a subir mientras el de train sigue
bajando, sería la señal clásica de que la red está memorizando en vez de generalizar. El test
se evalúa una única vez, al final, con la red ya congelada.

Es un problema de REGRESIÓN (predice un precio en euros, no una categoría), así que no aplica
una matriz de confusión -- el equivalente honesto es el mapa de precios aprendido y el error
en euros sobre casas de test nunca vistas en el entrenamiento.

El min/max de la normalización se calcula solo con el tramo de train (igual que en
03-tipos-clientes) y se aplica después a validación y test -- calcularlo con los 150 ejemplos
completos antes del split sería una fuga de información: la red vería el rango de precios y
metros cuadrados de validación/test antes de tiempo.

Uso: python house_price.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from capas import ActivacionLeakyReLU, CapaDensa, predecir

SEED_DATOS = 42  # gobierna solo la generación de datos -- fijo siempre, nunca es argumento
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Early stopping: para el entrenamiento en cuanto el error de VALIDACIÓN deja de mejorar de
# verdad (ver README para más detalle de por qué se usa una ventana en vez de comparar época a
# época).
PACIENCIA_EARLY_STOP = 200
MEJORA_MINIMA_RELATIVA = 0.005


def main(seed_split=42, seed_modelo=42, quiet=False, guardar_graficas=True) -> dict:
    """seed_split y seed_modelo son independientes entre sí y de SEED_DATOS -- separar las
    fuentes de aleatoriedad (qué datos existen / cómo se reparten / cómo se inicializa la red)
    permite medir la varianza de cada una por separado en vez de mezclarlas bajo una sola
    semilla global (ver README para el análisis de robustez sobre múltiples semillas)."""
    # Datos: SIEMPRE con SEED_DATOS, vía la API legacy de np.random -- así el dataset (qué 150
    # casas existen) no cambia nunca, sea cual sea seed_split/seed_modelo.
    np.random.seed(SEED_DATOS)
    metros_cuadrados = np.random.uniform(40, 240, (150, 1))
    habitaciones = np.random.randint(1, 6, (150, 1)).astype(float)
    X_puro = np.hstack([metros_cuadrados, habitaciones])

    precio_base = metros_cuadrados * 1500 + habitaciones * 25000 + 50000
    ruido = np.random.normal(0, 15000, (150, 1))
    Y_puro = precio_base + ruido

    # 90 train (60%) / 30 validación (20%, decide el early stopping) / 30 test (20%, se toca
    # una sola vez, al final). El split se hace ANTES de normalizar para que min/max salgan solo
    # de train (igual que en 03-tipos-clientes) -- si no, la red vería el rango de val/test.
    rng_split = np.random.default_rng(seed_split)
    indices = np.arange(150)
    rng_split.shuffle(indices)
    n_train, n_val = 90, 30
    idx_train, idx_val, idx_test = indices[:n_train], indices[n_train:n_train + n_val], indices[n_train + n_val:]
    X_train_raw, X_val_raw, X_test_raw = X_puro[idx_train], X_puro[idx_val], X_puro[idx_test]
    Y_train_raw, Y_val_raw, Y_test_raw = Y_puro[idx_train], Y_puro[idx_val], Y_puro[idx_test]

    X_min, X_max = np.min(X_train_raw, axis=0), np.max(X_train_raw, axis=0)
    Y_min, Y_max = np.min(Y_train_raw), np.max(Y_train_raw)
    X_train, X_val, X_test = ((X_train_raw - X_min) / (X_max - X_min),
                               (X_val_raw - X_min) / (X_max - X_min),
                               (X_test_raw - X_min) / (X_max - X_min))
    Y_train, Y_val, Y_test = ((Y_train_raw - Y_min) / (Y_max - Y_min),
                               (Y_val_raw - Y_min) / (Y_max - Y_min),
                               (Y_test_raw - Y_min) / (Y_max - Y_min))

    rng_modelo = np.random.default_rng(seed_modelo)
    red = [
        CapaDensa(dim_entrada=2, dim_salida=16, semilla_he=False, rng=rng_modelo),
        ActivacionLeakyReLU(),
        CapaDensa(dim_entrada=16, dim_salida=1, semilla_he=False, rng=rng_modelo),
    ]

    learning_rate = 0.05
    epochs = 4000
    historial_loss_train, historial_loss_val = [], []

    # Checkpoint del mejor punto de validación: el early stopping corta ~PACIENCIA_EARLY_STOP
    # épocas DESPUÉS del mínimo real de loss_val, así que quedarse con los pesos de la época de
    # corte sería quedarse con pesos peores que los del mínimo. Se guarda una copia de los pesos
    # de cada CapaDensa cada vez que loss_val marca un nuevo mínimo, y se restauran al salir del
    # bucle. Importante: .copy(), no una referencia -- si no, "restaurar" acabaría dejando los
    # pesos finales (los arrays se siguen modificando in-place en cada paso de gradiente).
    mejor_loss_val = np.inf
    mejor_epoca = None
    mejores_pesos = None

    for epoch in range(epochs):
        activacion = X_train
        for capa in red:
            activacion = capa.forward(activacion)
        A2_train = activacion

        loss_train = np.mean((A2_train - Y_train) ** 2)
        historial_loss_train.append(loss_train)

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
                if not quiet:
                    print(f"Early stopping en la época {epoch + 1}: el error de validación lleva "
                          f"{PACIENCIA_EARLY_STOP} épocas sin mejorar un {MEJORA_MINIMA_RELATIVA:.1%}")
                break
    else:
        if not quiet:
            print(f"Entrenamiento completado sin activar el early stopping (llegó a la época {epochs})")

    epocas_entrenadas = len(historial_loss_train)

    for capa, (W, b) in zip([c for c in red if isinstance(c, CapaDensa)], mejores_pesos):
        capa.W, capa.b = W, b
    if not quiet:
        print(f"Pesos restaurados al mínimo de validación: época {mejor_epoca + 1} "
              f"(loss_val={mejor_loss_val:.6f}), frente a la época de corte {epocas_entrenadas}")

    # === Única vez que se toca el conjunto de test, ya con la red entrenada (pesos del mínimo
    # de validación, no los de la última época entrenada) ===
    A2_test = predecir(red, X_test)
    loss_test = float(np.mean((A2_test - Y_test) ** 2))
    precio_pred_test = A2_test * (Y_max - Y_min) + Y_min
    precio_real_test = Y_test * (Y_max - Y_min) + Y_min
    mae_euros = float(np.mean(np.abs(precio_pred_test - precio_real_test)))

    metrics = {
        "seed_datos": SEED_DATOS,
        "seed_split": seed_split,
        "seed_modelo": seed_modelo,
        "epochs_configuradas": epochs,
        "epochs_entrenadas": epocas_entrenadas,
        "epoca_mejor_val": mejor_epoca + 1,
        "loss_train_final": float(historial_loss_train[mejor_epoca]),
        "loss_val_final": float(mejor_loss_val),
        "loss_test_final": loss_test,
        "mae_test_euros": mae_euros,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
    }

    if not guardar_graficas:
        return {**metrics, "historial_loss_val": historial_loss_val}

    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if not quiet:
        print(f"MAE en test: {mae_euros:.2f} EUR")

    # === Gráfico 1: curva de aprendizaje (train + validación, en euros desnormalizados) ===
    error_euros_train = np.sqrt(historial_loss_train) * (Y_max - Y_min)
    error_euros_val = np.sqrt(historial_loss_val) * (Y_max - Y_min)
    plt.figure(figsize=(6, 4))
    plt.plot(error_euros_train, color="blue", label="Error entrenamiento (Train)")
    plt.plot(error_euros_val, color="orange", linestyle="--", label="Error generalización (Validación)")
    plt.title("Curva de aprendizaje: detección de overfitting", fontweight="bold")
    plt.xlabel("Épocas")
    plt.ylabel("Error promedio (EUR)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "learning_curve.png", dpi=150)
    plt.close()

    # === Gráfico 2: visualización de datos = mapa de precios aprendido ===
    xx, yy = np.meshgrid(np.arange(0, 1.02, 0.02), np.arange(0, 1.02, 0.02))
    rejilla = np.c_[xx.ravel(), yy.ravel()]
    precios_rejilla = (predecir(red, rejilla) * (Y_max - Y_min) + Y_min).reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    fondo = plt.contourf(xx, yy, precios_rejilla, levels=20, alpha=0.8, cmap="coolwarm")
    plt.colorbar(fondo, label="Precio estimado (EUR)")
    plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train * (Y_max - Y_min) + Y_min,
                cmap="coolwarm", edgecolors="black", marker="o", s=50, label="Train")
    plt.scatter(X_val[:, 0], X_val[:, 1], c=Y_val * (Y_max - Y_min) + Y_min,
                cmap="coolwarm", edgecolors="darkorange", marker="^", s=90, label="Validación")
    plt.scatter(X_test[:, 0], X_test[:, 1], c=Y_test * (Y_max - Y_min) + Y_min,
                cmap="coolwarm", edgecolors="yellow", marker="*", s=120, label="Test")
    plt.title("Mapa de tasación aprendido por la red", fontweight="bold")
    plt.xlabel("Metros cuadrados (normalizado)")
    plt.ylabel("Número de habitaciones (normalizado)")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "data_visualization.png", dpi=150)
    plt.close()

    # === "Matriz": no aplica confusión (regresión). Predicho vs real en test. ===
    plt.figure(figsize=(5, 5))
    plt.scatter(precio_real_test, precio_pred_test, color="teal", edgecolors="k", s=80)
    lims = [min(precio_real_test.min(), precio_pred_test.min()),
            max(precio_real_test.max(), precio_pred_test.max())]
    plt.plot(lims, lims, color="gray", linestyle="--", label="Predicción perfecta")
    plt.title(f"Predicho vs real en test (MAE={mae_euros:.0f} EUR)", fontweight="bold")
    plt.xlabel("Precio real (EUR)")
    plt.ylabel("Precio predicho (EUR)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "predicted_vs_real.png", dpi=150)
    plt.close()

    if not quiet:
        print(f"Resultados guardados en {RESULTS_DIR}")

    return {**metrics, "historial_loss_val": historial_loss_val}


if __name__ == "__main__":
    main()
