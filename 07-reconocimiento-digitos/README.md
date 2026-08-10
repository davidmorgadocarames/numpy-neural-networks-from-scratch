# 07 — Reconocimiento de dígitos manuscritos (MNIST), 100% NumPy

El proyecto más avanzado del conjunto: reconoce dígitos manuscritos (0-9) del dataset MNIST
con una red neuronal escrita **completamente desde cero en NumPy** — sin TensorFlow, sin
Keras, sin PyTorch. Red modular 784 → 128 (LeakyReLU) → 10 (Softmax), usando el mismo
mini-framework de [`../capas.py`](../capas.py) que el resto de proyectos de este repo.

Se separa un conjunto de test estratificado por dígito, nunca visto en el entrenamiento, para
reportar una accuracy y una matriz de confusión honestas.

## 1. Entrenamiento — `digit_classifier.py`

Entrena la red sobre 1200 imágenes (120 por dígito) y evalúa sobre 300 de test (30 por dígito,
nunca vistas en el entrenamiento). Guarda los pesos entrenados en `results/red_pesos.npz` para
que la demo interactiva no tenga que reentrenar.

**Resultado**: **91.67% accuracy en test** tras 800 épocas — con una red minúscula (784→128→10,
~101k parámetros) entrenada sobre solo 1200 imágenes y sin ningún tipo de aumento de datos ni
regularización.

![Curva de aprendizaje](results/learning_curve.png)

**Visualización de datos** — muestra del dataset de entrenamiento:

![Muestra de dígitos](results/data_visualization.png)

**Matriz de confusión (test)**:

![Matriz de confusión](results/confusion_matrix.png)

**Lectura de la matriz**: la diagonal domina (272 de 300 aciertos) y casi todos los errores
son casos sueltos (1 imagen), lo esperable con solo 1200 imágenes de entrenamiento y sin
aumento de datos. Dos patrones sí destacan por repetirse más de una vez:

- **"3" es el dígito más difícil** (25/30 = 83%, el peor de los 10): se confunde con 2, 5, 7,
  8 y 9, una imagen cada uno. Tiene sentido — un "3" mal trazado comparte curvas con varios
  otros dígitos según el estilo de letra, mientras que dígitos con forma más rígida (0, 1, 6,
  9) apenas se confunden con nada.
- **"5"→"8" (2 casos) y "8"→"6" (2 casos)**: confusiones clásicas de MNIST por trazo
  incompleto — un "5" cuyo trazo superior se cierra un poco de más empieza a parecer un "8", y
  un "8" cuyo lazo superior no queda del todo cerrado se lee como un "6".

Con una red más grande, más datos o aumento de datos (rotaciones/desplazamientos pequeños)
estos casos límite mejorarían.

## 2. Demo interactiva — `demo_gradio.py`

Carga los pesos ya entrenados y abre un Sketchpad donde se puede dibujar un dígito a mano para
clasificarlo en vivo. Reutiliza el mismo preprocesado de centrado por centro de masa (caja
20x20 dentro de un lienzo 28x28) que replica cómo está construido MNIST realmente — sin ese
centrado, un trazo dibujado a mano y simplemente reescalado queda muy descentrado respecto a
los datos de entrenamiento y la precisión cae mucho aunque la red esté bien entrenada.

![Demo interactiva: dibujar un dígito y clasificarlo con la red NumPy](results/demo_sketchpad.gif)

## Reproducir

```bash
pip install -r ../requirements.txt
python digit_classifier.py   # entrena y guarda los pesos (~1-2 min, descarga MNIST la primera vez)
python demo_gradio.py        # demo interactiva, requiere haber ejecutado antes digit_classifier.py
```

## Limitaciones

- Solo 1200 imágenes de entrenamiento (frente a las 60.000 de MNIST completo) para que el
  entrenamiento sea rápido en CPU sin frameworks optimizados — con más datos la accuracy
  subiría, pero no es el objetivo de este proyecto (demostrar el mecanismo interno, no
  maximizar el benchmark).
- Sin aumento de datos, regularización (dropout) ni ajuste de hiperparámetros.
