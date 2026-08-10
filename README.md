# RRNN — Redes neuronales desde cero, en NumPy puro

Ocho redes neuronales, de la más simple a la más compleja, **implementadas desde cero en
NumPy** — sin TensorFlow, sin Keras, sin PyTorch. Cada capa (densa, convolucional, pooling,
dropout, LeakyReLU, Sigmoide, Softmax) tiene su propio `forward()` y `backward()` escritos a
mano, para entender exactamente qué calcula el descenso de gradiente en cada paso, en vez de
delegarlo en un framework de producción.

Cada una de estas redes se escribió primero como un script suelto y se fue refinando hasta un
mini-framework orientado a objetos reutilizable ([`capas.py`](capas.py)) — primero entender el
mecanismo por dentro a mano, antes de usar un framework de producción como TensorFlow/Keras
sabiendo qué hace realmente por debajo.

## Resultados de un vistazo

| # | Proyecto | Tipo | Resultado en test |
|---|---|---|---|
| 01 | [Compuerta XOR](01-compuertas-logicas-xor/) | Clasificación binaria | 4/4 aciertos, loss 0.0021 |
| 02 | [Celsius → Fahrenheit](02-celsius-fahrenheit/) | Regresión | MAE 0.0032 °F |
| 03 | [Tipos de cliente](03-tipos-clientes/) | Clasificación (3 clases) | 100% accuracy |
| 04 | [Temperatura día/noche](04-prediccion-temperatura-dia-noche/) | Regresión (serie temporal) | MAE 0.90 °C |
| 05 | [Precio de una casa](05-precio-casas/) | Regresión | MAE 11.452 € |
| 06 | [Zonas de espirales](06-zonas-espirales/) | Clasificación (3 clases) | 97.78% accuracy |
| 07 | [Dígitos manuscritos (MNIST)](07-reconocimiento-digitos/) | Clasificación (10 clases) | 91.67% accuracy |
| 08 | [CNN Fashion-MNIST: baseline vs augmentation](08-cnn-fashion-mnist/) | Clasificación (10 clases) | 79.20% (baseline) vs 72.70% (augmented) |

## Los 8 proyectos, de lo más simple a lo más complejo

Cada carpeta tiene su propio script reproducible, README con resultados reales (no solo
descritos: cada script se ejecutó para verificar las cifras) y tres visualizaciones
consistentes: **curva de aprendizaje**, **visualización de los datos** y **la matriz** —
matriz de confusión en los 5 proyectos de clasificación (01, 03, 06, 07, 08); en los 3 de
regresión (02, 04, 05) una matriz de confusión no tiene sentido matemático, así que se
sustituye por el equivalente honesto (predicho vs real), explicado en cada README. Debajo, una
imagen representativa de cada uno — en 02, 03, 07 y 08 es una animación (Manim) mostrando
arquitectura, forward, backward y resultado con la red real entrenándose en el momento; el
resto de gráficas y detalles están en el README de cada carpeta.

### 01 — [Compuerta XOR](01-compuertas-logicas-xor/)
Clasificación binaria no separable linealmente. **4/4 aciertos, loss 0.0021.**

![Frontera de decisión XOR](01-compuertas-logicas-xor/results/data_visualization.png)

### 02 — [Celsius → Fahrenheit](02-celsius-fahrenheit/)
Regresión con una sola neurona lineal. **MAE 0.0032 °F en test** (prácticamente exacto).

![Entrenamiento: arquitectura, forward, backward y resultado](02-celsius-fahrenheit/results/training.gif)

### 03 — [Tipos de cliente](03-tipos-clientes/)
Clasificación de 3 categorías. **100% accuracy en test.**

![Entrenamiento: arquitectura, forward, backward y resultado](03-tipos-clientes/results/training.gif)

### 04 — [Predicción de temperatura día/noche](04-prediccion-temperatura-dia-noche/)
Regresión con ventana deslizante sobre un ciclo día/noche con ruido. **MAE 0.90 °C en test.**

![Predicción vs realidad](04-prediccion-temperatura-dia-noche/results/predicted_vs_real.png)

### 05 — [Precio de una casa](05-precio-casas/)
Regresión con 2 variables (metros, habitaciones). **MAE 11.452 € en test.**

![Predicho vs real](05-precio-casas/results/predicted_vs_real.png)

### 06 — [Zonas de espirales](06-zonas-espirales/)
Clasificación con frontera curva, red profunda de 2 capas ocultas. **97.78% accuracy en test.**

![Zonas de espirales](06-zonas-espirales/results/data_visualization.png)

### 07 — [Dígitos manuscritos (MNIST)](07-reconocimiento-digitos/)
Clasificación de 10 clases + demo interactiva para dibujar y reconocer en vivo. **91.67%
accuracy en test.**

![Entrenamiento: arquitectura, forward, backward y resultado](07-reconocimiento-digitos/results/training.gif)

![Demo interactiva: dibujar un dígito y clasificarlo con la red NumPy](07-reconocimiento-digitos/results/demo_sketchpad.gif)

### 08 — [CNN sobre Fashion-MNIST: baseline vs data augmentation](08-cnn-fashion-mnist/)
Convolución + pooling + dropout, también 100% NumPy (técnica im2col). Entrenada dos veces con
los mismos pesos iniciales para medir el efecto real de la augmentation. **79.20% accuracy
(baseline) vs 72.70% (con augmentation)** — con este presupuesto de épocas, la augmentation
empeora el resultado (ver README del proyecto para el análisis completo).

![Entrenamiento: arquitectura, forward, backward y resultado](08-cnn-fashion-mnist/results/training.gif)

![Matriz de confusión — baseline](08-cnn-fashion-mnist/results/confusion_matrix_baseline.png)

## El mini-framework — `capas.py`

Las clases `CapaDensa`, `ActivacionLeakyReLU`, `ActivacionSigmoide` y `ActivacionSoftmax` se
escriben **una sola vez** en [`capas.py`](capas.py) y se reutilizan en los 7 primeros
proyectos, en vez de copiar y pegar el mismo bucle de entrenamiento cada vez. El proyecto 08
amplía el mismo patrón con capas convolucionales propias en
[`08-cnn-fashion-mnist/capas_cnn.py`](08-cnn-fashion-mnist/capas_cnn.py) (`CapaConv2D`,
`CapaMaxPool2D`, `CapaFlatten`, `CapaDropout`), reutilizando `CapaDensa` y las activaciones de
`capas.py` para su parte totalmente conectada. Cada capa sabe calcular su propio gradiente y
actualizar sus propios pesos; el bucle de entrenamiento no necesita saber qué hay dentro de
cada capa, solo encadenarlas:

```python
# Forward: pasar los datos por cada capa en orden
activacion = X
for capa in red:
    activacion = capa.forward(activacion)

# Backward: recorrer las capas al revés, cada una devuelve el gradiente de la anterior
gradiente = ...  # gradiente de la función de pérdida
for capa in reversed(red):
    gradiente = capa.backward(gradiente, learning_rate)
```

Es, en miniatura, el mismo patrón que usan `Sequential` y `model.fit()` en Keras — aquí escrito
a mano para verlo por dentro.

## Reproducir cualquier proyecto

```bash
pip install -r requirements.txt
cd 01-compuertas-logicas-xor && python xor_gate.py
```

(cada carpeta tiene su propio script principal; ver el README de cada una).

## Verificar los gradientes escritos a mano

[`tests/test_gradients.py`](tests/test_gradients.py) comprueba, por diferencias finitas, que
el `backward()` de cada capa (`CapaDensa`, las activaciones, y `CapaConv2D`/`CapaMaxPool2D`
del proyecto 08) calcula el gradiente correcto — no una afirmación en un README, un test que
se ejecuta:

```bash
pip install -r requirements.txt
pytest tests/test_gradients.py -v
```

## Metodología: por qué train / validación / test, y no solo train / test

Los proyectos con early stopping (02, 04, 05 y 08) usan un split de **tres** partes, no dos:
**train** (ajusta los pesos), **validación** (decide cuándo activar el early stopping) y
**test** (se evalúa una única vez, con la red ya entrenada y congelada, y no participa en
ninguna decisión anterior). Es el fix al error clásico de "fuga de información vía el
conjunto de test" (usar el propio test para decidir cuándo parar de entrenar, lo que
contamina la cifra final que se reporta como si fuera una estimación limpia de
generalización). En el proyecto 08 en concreto, este split resuelve además el problema de
raíz de la versión anterior: baseline y augmented ya no necesitan que nadie iguale
manualmente su criterio de parada para poder compararse de forma justa — cada uno para solo
cuando su propia validación dice que ya no mejora, sin que el experimentador tenga que mirar
ninguna cifra de test para decidir nada. El detalle de cada split (tamaños, y por qué en 04 el
split respeta el orden temporal) está en el README de cada proyecto.

## Limitaciones generales

- Todos los datasets son sintéticos o muestras reducidas (excepto MNIST en el proyecto 07 y
  Fashion-MNIST en el 08, que son reales pero se usa solo una muestra de cada uno) — el
  objetivo es demostrar el mecanismo de aprendizaje correctamente implementado, no maximizar
  el rendimiento en un benchmark.
- Sin GPU — todo corre en CPU, incluida la CNN del proyecto 08 (la convolución se vectoriza
  con la técnica im2col, ver su README, en vez de con bucles Python por píxel); no escalaría a
  datasets grandes sin reescribir en un framework de producción (TensorFlow/Keras, PyTorch...).
- Con datasets tan pequeños como los de 02 (30 ejemplos) o 05 (150), separar un tercer split
  de validación dificulta aún más la varianza estadística de la cifra final: con otra semilla
  de reparto, el MAE de test podría salir sensiblemente distinto. El objetivo de estos
  proyectos es demostrar el mecanismo y la metodología correcta, no una estimación de error
  robusta a gran escala.

## Licencia

[MIT](LICENSE).
