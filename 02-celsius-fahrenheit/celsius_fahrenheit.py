"""
Una sola neurona lineal (NumPy puro) que aprende la fórmula de conversión Celsius -> Fahrenheit
únicamente a partir de ejemplos (temperatura en C, temperatura en F), sin que se le diga nunca
la fórmula real. Es el ejemplo más simple posible de regresión: sin capas ocultas ni
activaciones no lineales, solo A = X*W + b, para ver con total claridad qué es lo que el
descenso de gradiente está ajustando.

Split en tres partes -- train / validación / test -- en vez de solo train/test: el early
stopping decide cuándo parar mirando el error de VALIDACIÓN, nunca el de test, para que la
cifra final de test sea una estimación limpia de generalización (test no participa en ninguna
decisión de entrenamiento, se evalúa una sola vez al final).

Es un problema de REGRESIÓN (predice un número continuo, no una clase), así que no aplica una
matriz de confusión -- el "veredicto" aquí es cuánto se acerca la fórmula aprendida por la red
a la fórmula física real (F = C*1.8 + 32), medido en el conjunto de test.

Uso: python celsius_fahrenheit.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SEED = 42
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Early stopping: para el entrenamiento en cuanto el error de VALIDACIÓN deja de mejorar de
# verdad, en vez de gastar épocas fijas de las que la mayoría no aportan nada. Se compara el
# loss de validación actual contra el de hace PACIENCIA_EARLY_STOP épocas -- si en toda esa
# ventana no ha mejorado al menos un MEJORA_MINIMA_RELATIVA (0.5%), se considera que ya no
# puede mejorar más y se corta ahí. Una comparación época-contra-época no serviría: el loss
# puede quedarse igual varias épocas seguidas y luego seguir bajando, así que hace falta una
# ventana.
PACIENCIA_EARLY_STOP = 200
MEJORA_MINIMA_RELATIVA = 0.005


def main() -> None:
    np.random.seed(SEED)

    # 30 ejemplos: 18 para entrenar (60%), 6 de validación (20%, decide el early stopping) y
    # 6 de test (20%, la red no los ve nunca hasta la evaluación final).
    X_todo = np.random.uniform(-20, 40, (30, 1))
    Y_todo = X_todo * 1.8 + 32

    indices = np.arange(30)
    np.random.shuffle(indices)
    n_train, n_val = 18, 6
    idx_train, idx_val, idx_test = indices[:n_train], indices[n_train:n_train + n_val], indices[n_train + n_val:]
    X_train, X_val, X_test = X_todo[idx_train], X_todo[idx_val], X_todo[idx_test]
    Y_train, Y_val, Y_test = Y_todo[idx_train], Y_todo[idx_val], Y_todo[idx_test]

    W = np.random.uniform(-1, 1, (1, 1))
    b = np.zeros((1, 1))

    learning_rate = 0.001
    epochs = 5000
    historial_loss_train, historial_loss_val = [], []

    # Checkpoint del mejor punto de validación: el early stopping corta ~PACIENCIA_EARLY_STOP
    # épocas DESPUÉS del mínimo real de loss_val (necesita esa ventana para confirmar que ya no
    # mejora), así que quedarse con W/b de la época de corte sería quedarse con pesos peores que
    # los del mínimo. Se guarda una copia de W/b cada vez que loss_val marca un nuevo mínimo, y
    # se restaura al salir del bucle -- tanto si se corta por early stopping como si se agota
    # epochs. Importante: .copy(), no una referencia -- si no, "restaurar" acabaría dejando los
    # pesos finales (los arrays se siguen modificando in-place en cada paso de gradiente).
    mejor_loss_val = np.inf
    mejor_epoca = None
    W_mejor, b_mejor = None, None

    for epoch in range(epochs):
        A_train = np.dot(X_train, W) + b
        loss_train = np.mean((A_train - Y_train) ** 2)
        historial_loss_train.append(loss_train)

        A_val = np.dot(X_val, W) + b
        loss_val = np.mean((A_val - Y_val) ** 2)
        historial_loss_val.append(loss_val)

        if loss_val < mejor_loss_val:
            mejor_loss_val = loss_val
            mejor_epoca = epoch
            W_mejor, b_mejor = W.copy(), b.copy()

        dLoss_dA = 2 * (A_train - Y_train) / Y_train.shape[0]
        dW = np.dot(X_train.T, dLoss_dA)
        db = np.sum(dLoss_dA, axis=0, keepdims=True)
        W -= learning_rate * dW
        b -= learning_rate * db

        if epoch >= PACIENCIA_EARLY_STOP:
            loss_val_referencia = historial_loss_val[epoch - PACIENCIA_EARLY_STOP]
            mejora_relativa = (loss_val_referencia - loss_val) / loss_val_referencia
            if mejora_relativa < MEJORA_MINIMA_RELATIVA:
                print(f"Early stopping en la época {epoch + 1}: el error de validación lleva "
                      f"{PACIENCIA_EARLY_STOP} épocas sin mejorar un {MEJORA_MINIMA_RELATIVA:.1%}")
                break
    else:
        print(f"Entrenamiento completado sin activar el early stopping: el error de "
              f"validación seguía mejorando más de un {MEJORA_MINIMA_RELATIVA:.1%} cada "
              f"{PACIENCIA_EARLY_STOP} épocas incluso al llegar a la época {epochs}")

    epocas_entrenadas = len(historial_loss_train)

    W, b = W_mejor, b_mejor
    print(f"Pesos restaurados al mínimo de validación: época {mejor_epoca + 1} "
          f"(loss_val={mejor_loss_val:.6f}), frente a la época de corte {epocas_entrenadas}")

    # === Única vez que se toca el conjunto de test, ya con W y b congelados (los del mínimo
    # de validación, no los de la última época entrenada) ===
    A_test = np.dot(X_test, W) + b
    mse_test = float(np.mean((A_test - Y_test) ** 2))
    mae_test = float(np.mean(np.abs(A_test - Y_test)))

    metrics = {
        "epochs_configuradas": epochs,
        "epochs_entrenadas": epocas_entrenadas,
        "epoca_mejor_val": mejor_epoca + 1,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "W_aprendido": float(W[0][0]),
        "b_aprendido": float(b[0][0]),
        "W_real": 1.8,
        "b_real": 32.0,
        "mse_train_final": float(historial_loss_train[mejor_epoca]),
        "mse_val_final": float(mejor_loss_val),
        "mse_test_final": mse_test,
        "mae_test_grados_F": mae_test,
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"W aprendido: {W[0][0]:.4f} (real: 1.8) | b aprendido: {b[0][0]:.4f} (real: 32.0)")
    print(f"MAE en test: {mae_test:.4f} grados F")

    # === Gráfico 1: curva de aprendizaje (train + validación, que es lo que decide cuándo
    # parar) ===
    plt.figure(figsize=(6, 4))
    plt.plot(historial_loss_train, color="blue", label="Train")
    plt.plot(historial_loss_val, color="orange", linestyle="--", label="Validación")
    plt.title("Curva de aprendizaje (Celsius -> Fahrenheit)", fontweight="bold")
    plt.xlabel("Épocas")
    plt.ylabel("Error cuadrático medio")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "learning_curve.png", dpi=150)
    plt.close()

    # === Gráfico 2: visualización de datos = recta aprendida vs datos reales ===
    plt.figure(figsize=(6, 4))
    plt.scatter(X_train, Y_train, color="red", label="Datos de entrenamiento", zorder=5)
    plt.scatter(X_val, Y_val, color="mediumseagreen", edgecolors="k", label="Datos de validación", zorder=5, marker="^", s=90)
    plt.scatter(X_test, Y_test, color="gold", edgecolors="k", label="Datos de test", zorder=5, marker="*", s=120)
    X_linea = np.linspace(-30, 50, 100).reshape(-1, 1)
    plt.plot(X_linea, np.dot(X_linea, W) + b, color="black", linewidth=2, label="Fórmula aprendida por la IA")
    plt.title("Celsius vs Fahrenheit", fontweight="bold")
    plt.xlabel("Celsius (°C)")
    plt.ylabel("Fahrenheit (°F)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "data_visualization.png", dpi=150)
    plt.close()

    # === "Matriz": no aplica una matriz de confusión (regresión, no clasificación). En su
    # lugar, predicho vs real en el conjunto de test -- el equivalente honesto para regresión. ===
    plt.figure(figsize=(5, 5))
    plt.scatter(Y_test, A_test, color="teal", edgecolors="k", s=80)
    lims = [min(Y_test.min(), A_test.min()), max(Y_test.max(), A_test.max())]
    plt.plot(lims, lims, color="gray", linestyle="--", label="Predicción perfecta")
    plt.title(f"Predicho vs Real en test (MAE={mae_test:.3f} °F)", fontweight="bold")
    plt.xlabel("Fahrenheit real")
    plt.ylabel("Fahrenheit predicho")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "predicted_vs_real.png", dpi=150)
    plt.close()

    print(f"Resultados guardados en {RESULTS_DIR}")


if __name__ == "__main__":
    main()
