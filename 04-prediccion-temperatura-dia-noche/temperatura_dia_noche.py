"""
Predicción de temperatura día/noche (NumPy puro): una red densa mira las últimas 3 horas de
temperatura y predice la hora siguiente. Es un problema de REGRESIÓN sobre una ventana
deslizante de historia -- no confunde categorías, así que no aplica una matriz de confusión;
el "veredicto" es cuánto se parece la predicción de la IA al futuro real que nunca vio durante
el entrenamiento.

La serie es sintética pero en unidades físicas reales: temperatura en grados Celsius, con
lecturas horarias (24 puntos = 1 día) siguiendo un ciclo día/noche (mínimo de madrugada,
máximo a mediodía) más ruido gaussiano. Cada "paso" en los gráficos de predicción es
literalmente 1 hora hacia el futuro, no un día -- con VENTANA=3 la red mira las últimas 3
horas para predecir la siguiente.

El split respeta el orden temporal en sus TRES partes -- train / validación / test, en ese
orden cronológico -- porque mezclar aleatoriamente aquí sería hacer trampa: la red podría
"memorizar" un instante rodeado de datos de entrenamiento en vez de predecir el futuro. El
early stopping decide cuándo parar mirando el error de VALIDACIÓN (el tramo de tiempo
intermedio); el tramo final de test no se toca hasta la evaluación final.

Uso: python temperatura_dia_noche.py
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

SEED = 42
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
VENTANA = 3  # horas de historia que mira la red para predecir la siguiente

N_DIAS = 8
PUNTOS_POR_DIA = 24  # 1 lectura por hora
TEMP_MEDIA_C = 18.0
AMPLITUD_C = 7.0  # oscilación día/noche: mínimo ~11°C de madrugada, máximo ~25°C a mediodía
RUIDO_STD_C = 0.7

# Early stopping: para el entrenamiento en cuanto el error de VALIDACIÓN deja de mejorar de
# verdad (ver README de 02-celsius-fahrenheit y 05-precio-casas para la explicación completa).
PACIENCIA_EARLY_STOP = 200
MEJORA_MINIMA_RELATIVA = 0.005


def main() -> None:
    np.random.seed(SEED)

    n_puntos = N_DIAS * PUNTOS_POR_DIA
    dias = np.linspace(0, N_DIAS, n_puntos, endpoint=False)  # 0.0, 1/24, 2/24, ... (en días)
    fase = 2 * np.pi * dias  # 1 ciclo completo = 1 día (fase=0 -> medianoche, fase=pi -> mediodía)
    temperatura_c = TEMP_MEDIA_C - AMPLITUD_C * np.cos(fase) + np.random.normal(0, RUIDO_STD_C, n_puntos)

    t_min, t_max = np.min(temperatura_c), np.max(temperatura_c)
    serie_norm = (temperatura_c - t_min) / (t_max - t_min)

    X_lista, Y_lista = [], []
    for i in range(len(serie_norm) - VENTANA):
        X_lista.append(serie_norm[i : i + VENTANA])
        Y_lista.append(serie_norm[i + VENTANA])
    X = np.array(X_lista)
    Y = np.array(Y_lista).reshape(-1, 1)

    # 60% train / 20% validación / 20% test, en ese orden cronológico -- validación y test son
    # siempre posteriores en el tiempo a los datos que la red usó para ajustar pesos.
    n = len(X)
    split_train = int(n * 0.6)
    split_val = int(n * 0.8)
    X_train, X_val, X_test = X[:split_train], X[split_train:split_val], X[split_val:]
    Y_train, Y_val, Y_test = Y[:split_train], Y[split_train:split_val], Y[split_val:]

    red = [
        CapaDensa(dim_entrada=VENTANA, dim_salida=6, semilla_he=False),
        ActivacionLeakyReLU(),
        CapaDensa(dim_entrada=6, dim_salida=1, semilla_he=False),
    ]

    learning_rate = 0.1
    epochs = 3000
    historial_loss_train, historial_loss_val = [], []

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

        gradiente = 2 * (A2_train - Y_train) / Y_train.shape[0]
        for capa in reversed(red):
            gradiente = capa.backward(gradiente, learning_rate)

        if epoch >= PACIENCIA_EARLY_STOP:
            loss_val_referencia = historial_loss_val[epoch - PACIENCIA_EARLY_STOP]
            mejora_relativa = (loss_val_referencia - loss_val) / loss_val_referencia
            if mejora_relativa < MEJORA_MINIMA_RELATIVA:
                print(f"Early stopping en la época {epoch + 1}: el error de validación lleva "
                      f"{PACIENCIA_EARLY_STOP} épocas sin mejorar un {MEJORA_MINIMA_RELATIVA:.1%}")
                break
    else:
        print(f"Entrenamiento completado sin activar el early stopping (llegó a la época {epochs})")

    epocas_entrenadas = len(historial_loss_train)

    # === Única vez que se toca el conjunto de test, ya con la red entrenada ===
    A2_test = predecir(red, X_test)
    loss_test = float(np.mean((A2_test - Y_test) ** 2))
    Y_test_c = Y_test * (t_max - t_min) + t_min
    Y_pred_c = A2_test * (t_max - t_min) + t_min
    mae_test_c = float(np.mean(np.abs(Y_pred_c - Y_test_c)))

    # Horas de test, para etiquetar el eje X del gráfico de predicción con tiempo real (no
    # con un índice 0,1,2... sin unidades): cada muestra de test predice la hora en
    # dias[i + VENTANA], así que la primera hora de test es la primera predicha después del
    # corte validación/test.
    horas_test = (dias[VENTANA + split_val : VENTANA + split_val + len(Y_test)] - dias[VENTANA + split_val]) * 24

    metrics = {
        "epochs_configuradas": epochs,
        "epochs_entrenadas": epocas_entrenadas,
        "ventana_horas": VENTANA,
        "loss_train_final": float(historial_loss_train[-1]),
        "loss_val_final": float(historial_loss_val[-1]),
        "loss_test_final": loss_test,
        "mae_test_celsius": mae_test_c,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"MAE en test: {mae_test_c:.4f} °C")

    # === Gráfico 1: curva de aprendizaje (train + validación, que es lo que decide cuándo
    # parar) ===
    plt.figure(figsize=(6, 4))
    plt.plot(historial_loss_train, color="blue", label="Train")
    plt.plot(historial_loss_val, color="orange", linestyle="--", label="Validación")
    plt.title("Curva de aprendizaje: temperatura día/noche", fontweight="bold")
    plt.xlabel("Épocas")
    plt.ylabel("Error cuadrático medio (escala normalizada 0-1)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "learning_curve.png", dpi=150)
    plt.close()

    # === Gráfico 2: visualización de datos = la serie completa (train + val + test), en °C
    # y días ===
    plt.figure(figsize=(8, 4))
    plt.plot(dias, temperatura_c, color="gray", alpha=0.6, label="Temperatura simulada (con ruido)")
    plt.axvline(x=dias[VENTANA + split_train], color="darkorange", linestyle=":", label="Inicio de validación")
    plt.axvline(x=dias[VENTANA + split_val], color="black", linestyle=":", label="Inicio del test")
    plt.title("Temperatura día/noche simulada", fontweight="bold")
    plt.xlabel("Tiempo (días)")
    plt.ylabel("Temperatura (°C)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "data_visualization.png", dpi=150)
    plt.close()

    # === "Matriz": no aplica confusión (regresión). Real vs predicho en test, el
    # equivalente honesto para series temporales. Eje X en horas reales desde el inicio del
    # test, no en un índice sin unidades. ===
    plt.figure(figsize=(7, 4))
    plt.plot(horas_test, Y_test_c, color="black", label="Futuro real (nunca visto)", linewidth=2)
    plt.plot(horas_test, Y_pred_c, color="red", linestyle="--", label="Predicción de la IA")
    plt.title(f"Predicción vs realidad en test (MAE={mae_test_c:.2f} °C)", fontweight="bold")
    plt.xlabel("Horas desde el inicio del test")
    plt.ylabel("Temperatura (°C)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "predicted_vs_real.png", dpi=150)
    plt.close()

    print(f"Resultados guardados en {RESULTS_DIR}")


if __name__ == "__main__":
    main()
