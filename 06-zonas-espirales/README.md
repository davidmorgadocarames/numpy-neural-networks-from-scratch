# 06 — Zonas de espirales: frontera de decisión curva

El benchmark clásico para demostrar que una red con suficientes capas ocultas puede aprender
fronteras de decisión **curvas**, no solo rectas — 3 brazos de una espiral entrelazados, que
ningún clasificador lineal podría separar. Red modular profunda: 2 → 64 (LeakyReLU) → 64
(LeakyReLU) → 3 (Softmax).

Se separa un 20% como test estratificado por brazo, para medir generalización real con una
matriz de confusión sobre puntos nunca vistos en el entrenamiento.

## Resultado

**Accuracy en test: 97.78%** (90 puntos de test, de 450 en total), tras 5000 épocas.

![Curva de aprendizaje](results/learning_curve.png)

**Visualización de datos — zonas aprendidas**: el fondo de color muestra la frontera curva que
la red dibujó para separar los 3 brazos. Los puntos pequeños son entrenamiento, las estrellas
grandes son test.

![Zonas de espirales](results/data_visualization.png)

**Matriz de confusión (test)**: 88 aciertos de 90, solo 2 errores, ambos entre brazos
consecutivos (Brazo 1→Brazo 0 y Brazo 2→Brazo 1) — nunca entre el Brazo 0 y el Brazo 2, que
son los más alejados entre sí en la espiral. Mirando el gráfico de zonas de arriba, tiene
sentido: los 3 brazos se enroscan hacia un centro común, así que cerca del origen (donde las
3 espirales pasan muy cerca unas de otras) es donde la frontera de decisión tiene que curvarse
más y con menos margen, y es justo ahí donde caen los 2 puntos de test mal clasificados. Lejos
del centro, donde los brazos están claramente separados, no hay ningún error.

![Matriz de confusión](results/confusion_matrix.png)

## Reproducir

```bash
pip install -r ../requirements.txt
python spiral_classifier.py
```
