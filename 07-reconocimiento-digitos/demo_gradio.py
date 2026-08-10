"""
Demo interactiva (Gradio) del reconocedor de dígitos NumPy: carga los pesos ya entrenados por
digit_classifier.py (results/red_pesos.npz, no reentrena) y expone un Sketchpad para dibujar
un dígito a mano y clasificarlo en vivo con la red construida desde cero.

El preprocesado (escala de grises, detección de fondo, umbral de ruido y centrado por centro
de masa en una caja de 20x20) replica el de MNIST real -- sin él, un trazo dibujado a mano
queda muy descentrado respecto a los datos de entrenamiento y la precisión cae mucho aunque
la red esté bien entrenada.

Uso: python demo_gradio.py (requiere haber ejecutado antes digit_classifier.py)
"""

import sys
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from capas import ActivacionLeakyReLU, ActivacionSoftmax, CapaDensa, predecir

RESULTS_DIR = Path(__file__).parent / "results"
PESOS_PATH = RESULTS_DIR / "red_pesos.npz"


def cargar_red() -> list:
    pesos = np.load(PESOS_PATH)
    capa1 = CapaDensa(dim_entrada=784, dim_salida=128)
    capa1.W, capa1.b = pesos["W1"], pesos["b1"]
    capa2 = CapaDensa(dim_entrada=128, dim_salida=10)
    capa2.W, capa2.b = pesos["W2"], pesos["b2"]
    return [capa1, ActivacionLeakyReLU(), capa2, ActivacionSoftmax()]


def centrar_por_masa(matriz: np.ndarray, tamano: int = 28, tamano_caja: int = 20) -> np.ndarray:
    filas_no_vacias = np.where(np.any(matriz > 0, axis=1))[0]
    columnas_no_vacias = np.where(np.any(matriz > 0, axis=0))[0]
    if len(filas_no_vacias) == 0 or len(columnas_no_vacias) == 0:
        return matriz

    fila_min, fila_max = filas_no_vacias[[0, -1]]
    columna_min, columna_max = columnas_no_vacias[[0, -1]]
    recorte = matriz[fila_min : fila_max + 1, columna_min : columna_max + 1]

    alto, ancho = recorte.shape
    escala = tamano_caja / max(alto, ancho)
    nuevo_alto = max(1, round(alto * escala))
    nuevo_ancho = max(1, round(ancho * escala))
    recorte_redimensionado = np.array(
        Image.fromarray(recorte).resize((nuevo_ancho, nuevo_alto), Image.Resampling.LANCZOS)
    )

    lienzo = np.zeros((tamano, tamano), dtype=np.float64)
    fila_offset = (tamano - nuevo_alto) // 2
    columna_offset = (tamano - nuevo_ancho) // 2
    lienzo[fila_offset : fila_offset + nuevo_alto, columna_offset : columna_offset + nuevo_ancho] = recorte_redimensionado

    filas_idx, columnas_idx = np.indices(lienzo.shape)
    masa_total = lienzo.sum()
    if masa_total > 0:
        centro_fila = (filas_idx * lienzo).sum() / masa_total
        centro_columna = (columnas_idx * lienzo).sum() / masa_total
        lienzo = np.roll(lienzo, int(round(tamano / 2 - centro_fila)), axis=0)
        lienzo = np.roll(lienzo, int(round(tamano / 2 - centro_columna)), axis=1)

    return lienzo


def build_predict_fn(red: list):
    def predecir_dibujo(imagen_dict) -> str:
        if isinstance(imagen_dict, dict) and "composite" in imagen_dict:
            imagen_pil = imagen_dict["composite"]
        else:
            imagen_pil = imagen_dict

        if imagen_pil is None:
            return "Pizarra vacía. ¡Dibuja un número más grande o marcado!"
        if not isinstance(imagen_pil, Image.Image):
            imagen_pil = Image.fromarray(np.uint8(imagen_pil))

        imagen_gris = imagen_pil.convert("L").resize((28, 28), Image.Resampling.LANCZOS)
        matriz = np.array(imagen_gris, dtype=np.float64)

        if np.mean(matriz) > 127:
            matriz = 255.0 - matriz
        matriz = np.where(matriz > 40, matriz, 0)

        if np.max(matriz) == 0 or np.sum(matriz) < 500:
            return "Pizarra vacía. ¡Dibuja un número más grande o marcado!"

        matriz = centrar_por_masa(matriz)
        vector_entrada = (matriz / 255.0).reshape(1, 784)

        probabilidades = predecir(red, vector_entrada)[0]
        ranking = np.argsort(probabilidades)[::-1]

        lineas = [f"Predicción: {ranking[0]} ({probabilidades[ranking[0]] * 100:.1f}% de confianza)"]
        for puesto, digito in enumerate(ranking[1:4], start=2):
            lineas.append(f"  {puesto}ª opción: {digito} ({probabilidades[digito] * 100:.1f}%)")
        return "\n".join(lineas)

    return predecir_dibujo


def main() -> None:
    if not PESOS_PATH.exists():
        raise SystemExit(
            f"No se encuentra {PESOS_PATH}. Ejecuta primero 'python digit_classifier.py' "
            "para entrenar y guardar los pesos."
        )

    red = cargar_red()
    interfaz = gr.Interface(
        fn=build_predict_fn(red),
        inputs=gr.Sketchpad(canvas_size=(280, 280), type="pil", label="Dibuja un número aquí"),
        outputs=gr.Textbox(label="Predicción de la red NumPy (escrita desde cero)", lines=5),
        title="Reconocedor de dígitos — red neuronal NumPy desde cero",
        description=(
            "Dibuja un dígito del 0 al 9, centrado y grande, y pulsa 'Submit'. Red densa "
            "(784->128->10, LeakyReLU+Softmax) implementada a mano en NumPy, sin frameworks "
            "de deep learning, ~91.7% accuracy en test."
        ),
        live=False,
    )
    interfaz.launch(inbrowser=True)


if __name__ == "__main__":
    main()
