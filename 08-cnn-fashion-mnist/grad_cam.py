"""
Grad-CAM (Selvaraju et al., 2017 -- https://arxiv.org/abs/1610.02391): el equivalente de los
mapas de saliencia (`../07-reconocimiento-digitos/mapas_saliencia.py`) para una red con
estructura espacial interna. En vez del gradiente respecto a los PÍXELES de entrada (que en
una CNN da un mapa tan ruidoso como en una red densa, por el mismo motivo -- ver ese README),
Grad-CAM usa el gradiente respecto al último MAPA DE ACTIVACIONES convolucional: cada canal de
ese mapa ya es la respuesta de un filtro aprendido a una zona de la imagen, así que el
gradiente ahí es mucho menos ruidoso y señala directamente "qué región" importó, no "qué
píxel suelto".

Mecánica: en vez de retropropagar hasta la entrada (como hacen FGSM/saliencia), se para justo
después de la última capa convolucional -- se captura el gradiente que llegaría a esa capa,
sin seguir bajando hasta los píxeles. Se usa la salida de la 2ª CapaConv2D tras su LeakyReLU
(11×11×16, justo antes del 2º MaxPool2D) -- el último punto con resolución espacial razonable
antes de que el flatten la destruya.

`alpha_c = media(dL/dA_c)` sobre alto y ancho (cuánto le importa a la red, en promedio, subir
la activación de ese canal) pondera cada canal del mapa de activaciones; se suman, se aplica
ReLU (solo interesa lo que empuja HACIA la clase, no en contra) y se reescala a la imagen
original con vecino más cercano (sin dependencias de procesado de imagen, igual que
`augmentar_lote()` en `capas_cnn.py`).

Uso: python grad_cam.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from capas import OptimizadorAdam
from capas_cnn import predecir_cnn

from cnn_fashion_mnist_full import NOMBRES_CLASES, cargar_datos
from sgd_vs_adam_full import LEARNING_RATE_ADAM, crear_red, entrenar

RESULTS_DIR = Path(__file__).parent / "results_grad_cam"
RESULTS_DIR.mkdir(exist_ok=True)

IDX_CAPA_OBJETIVO = 4  # salida de LeakyReLU tras la 2ª CapaConv2D, ver docstring del módulo


def grad_cam(red, X, Y_onehot, idx_capa_objetivo=IDX_CAPA_OBJETIVO):
    activacion = X
    activaciones = []
    for capa in red:
        activacion = capa.forward(activacion, entrenando=True)
        activaciones.append(activacion)
    probs = activacion
    A = activaciones[idx_capa_objetivo]  # (N, H, W, C)

    gradiente = (probs - Y_onehot) / Y_onehot.shape[0]
    # se retropropaga solo hasta justo DESPUÉS de la capa objetivo -- no hasta la entrada,
    # a diferencia de FGSM/saliencia. learning_rate=0 para no mover pesos ni contaminar Adam.
    for capa in reversed(red[idx_capa_objetivo + 1:-1]):
        gradiente = capa.backward(gradiente, learning_rate=0.0)
    dA = gradiente  # (N, H, W, C) -- dL/dA

    alpha = dA.mean(axis=(1, 2), keepdims=True)  # (N, 1, 1, C)
    cam = np.maximum(np.sum(alpha * A, axis=3), 0.0)  # (N, H, W), ReLU

    maximos = cam.reshape(cam.shape[0], -1).max(axis=1)
    maximos[maximos == 0] = 1.0
    return cam / maximos[:, None, None]


def redimensionar_nn(mapa, alto, ancho):
    """Escala un mapa 2D a (alto, ancho) por vecino más cercano -- sin dependencias de
    procesado de imagen, mismo espíritu que el resto del repo."""
    h, w = mapa.shape
    filas = (np.arange(alto) * h / alto).astype(int)
    columnas = (np.arange(ancho) * w / ancho).astype(int)
    return mapa[np.ix_(filas, columnas)]


def graficar_grad_cam(red, X_test, Y_test_num, Y_test_onehot, ruta_salida, n=6, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_test), size=n, replace=False)
    X_muestra = X_test[idx]
    Y_muestra_onehot = Y_test_onehot[idx]
    Y_muestra_num = Y_test_num[idx]

    cams = grad_cam(red, X_muestra.copy(), Y_muestra_onehot)
    pred = np.argmax(predecir_cnn(red, X_muestra), axis=1)

    fig, axes = plt.subplots(2, n, figsize=(2.4 * n, 5.3))
    for col in range(n):
        imagen = X_muestra[col, :, :, 0]
        axes[0, col].imshow(imagen, cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(f"real={NOMBRES_CLASES[Y_muestra_num[col]]}\n"
                                f"pred={NOMBRES_CLASES[pred[col]]}", fontsize=8)
        axes[0, col].axis("off")

        mapa_grande = redimensionar_nn(cams[col], 28, 28)
        axes[1, col].imshow(imagen, cmap="gray", vmin=0, vmax=1)
        axes[1, col].imshow(mapa_grande, cmap="jet", alpha=0.5)
        axes[1, col].axis("off")

    plt.suptitle("Grad-CAM: qué zona de la imagen activó la decisión (superpuesto en rojo/amarillo)",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150)
    plt.close()


def main(seed_split=42, seed_modelo=42, quiet=False, guardar_graficas=True) -> dict:
    X_train, Y_train, Y_train_num, X_val, Y_val_num, X_test, Y_test_num = cargar_datos(seed_split, quiet=quiet)
    Y_test_onehot = np.eye(10)[Y_test_num]

    if not quiet:
        print("=== Entrenando la red de referencia (Adam, mini-batch, baseline) ===")
    rng_modelo = np.random.default_rng(seed_modelo)
    red = crear_red(rng_modelo, OptimizadorAdam)
    rng_aug = np.random.default_rng(seed_modelo)
    rng_batches = np.random.default_rng(seed_modelo)
    entrenar("gradcam_base", red, X_train, Y_train, X_val, Y_val_num, False, rng_aug, rng_batches,
              LEARNING_RATE_ADAM, quiet=quiet)

    if not guardar_graficas:
        return {"seed_split": seed_split, "seed_modelo": seed_modelo}

    graficar_grad_cam(red, X_test, Y_test_num, Y_test_onehot, RESULTS_DIR / "grad_cam.png")

    if not quiet:
        print(f"Resultados guardados en {RESULTS_DIR}")

    return {"seed_split": seed_split, "seed_modelo": seed_modelo}


if __name__ == "__main__":
    main()
