# 08 — CNN sobre Fashion-MNIST, 100% NumPy: baseline vs data augmentation

El proyecto más avanzado del repo: una **CNN completa (convolución + max pooling + dropout)
escrita desde cero en NumPy**, sin TensorFlow, sin Keras, sin PyTorch — ni siquiera para la
parte convolucional. Los 7 proyectos anteriores usan solo capas densas; este añade
convolución 2D, pooling y dropout implementados a mano en
[`capas_cnn.py`](capas_cnn.py) mediante la técnica **im2col** (convertir cada ventana de la
imagen en una fila de una matriz, para que la convolución se resuelva con una sola
multiplicación de matrices en vez de con bucles Python por píxel — lo que hace viable
entrenar la red entera en CPU sin GPU). Los gradientes de `CapaConv2D`, `CapaMaxPool2D` y del
resto de capas de [`../capas.py`](../capas.py) se verifican numéricamente contra diferencias
finitas en [`../tests/test_gradients.py`](../tests/test_gradients.py) — no es una afirmación
sin más, se puede ejecutar y comprobar:

```bash
pip install -r ../requirements.txt
pytest ../tests/test_gradients.py -v
```

La arquitectura se entrena **dos veces con los mismos pesos iniciales**: una vez tal cual
(*baseline*) y otra con data augmentation (flip horizontal, rotación, zoom y desplazamiento
aleatorios, reimplementados a mano en NumPy — ver `augmentar_lote()` en `capas_cnn.py`), para
medir el efecto real de la técnica en vez de solo mencionarla.

## Arquitectura

```
Entrada 28×28×1
  → Conv2D 3×3, 8 filtros → LeakyReLU → MaxPool 2×2      (26×26×8 → 13×13×8)
  → Conv2D 3×3, 16 filtros → LeakyReLU → MaxPool 2×2     (11×11×16 → 5×5×16)
  → Flatten (400)
  → Densa 400→64 → LeakyReLU → Dropout(0.3)
  → Densa 64→10 → Softmax
```

~27.500 parámetros. Entrenamiento full-batch (mismo estilo que el resto del repo, sin
mini-batches), con split de tres partes estratificado por clase: **2400 imágenes de train**
(240/clase), **600 de validación** (60/clase, decide el early stopping) y **1000 de test**
(100/clase, se evalúan una sola vez, después de entrenar) — mismo patrón de estratificación
que `07-reconocimiento-digitos`, con el split de validación añadido (ver "Metodología" en el
[README raíz](../README.md) para el porqué).

## Los datos y la augmentation

![Muestra de datos y augmentation](results/data_visualization.png)

Fila 1: una prenda de cada una de las 10 clases de Fashion-MNIST. Filas 2 y 3: la **misma**
imagen (una camiseta) sometida a 10 augmentations aleatorias distintas cada vez — se ve el
flip horizontal, la rotación, el zoom y el desplazamiento actuando de forma independiente en
cada copia, tal como los vería la red en épocas distintas. La validación y el test **nunca**
pasan por `augmentar_lote()` — es una técnica de entrenamiento, no algo que la red vaya a ver
en producción.

## Resultado: la augmentation empeoró el resultado con este presupuesto de épocas

El early stopping de cada versión decide cuándo parar mirando su propio loss de
**validación** — no hace falta igualar nada a mano entre baseline y augmented, cada una para
sola cuando su validación deja de mejorar. En este presupuesto de 400 épocas máximas, ninguna
de las dos activó el early stopping (ambas llegaron al límite todavía mejorando ligeramente):

| | Épocas | Accuracy en test | Loss validación final |
|---|---|---|---|
| **Baseline** | 400 (todas) | **79.20%** | 0.4770 |
| **Con augmentation** | 400 (todas) | **72.70%** | 0.5923 |

![Curva de aprendizaje](results/learning_curve.png)

La curva de arriba muestra accuracy de **validación** (lo que efectivamente se registra época
a época; el test es un único número final por versión, calculado una sola vez). La verde
(augmented) no cruza a la azul (baseline) en ningún punto del entrenamiento.

**Por qué tiene sentido, no solo "la augmentation no funcionó"**:

1. **Cada época de augmentation es una época distinta de facto.** El baseline refina el mismo
   dataset de 2400 imágenes una y otra vez; la versión con augmentation ve una transformación
   aleatoria distinta en cada época, así que en la práctica optimiza sobre un objetivo más
   ruidoso y le cuesta más ajustar los pesos por época — se nota en la curva verde, claramente
   más ruidosa que la azul. Con un presupuesto de épocas fijo, ese coste de optimización se
   paga entero pero el beneficio de generalización (que necesita más tiempo para notarse) no
   llega a compensarlo.
2. **Apenas había hueco de overfitting que cerrar.** El loss de validación del baseline sigue
   bajando en paralelo al de train hasta la época 400 sin señales de divergencia (ver
   `results/metrics.json`) — la red ya generaliza razonablemente bien gracias al Dropout(0.3)
   que ambas versiones comparten. La augmentation ayuda más cuando el modelo memoriza (train
   mucho mejor que validación); aquí ese síntoma casi no existe, así que añadir encima una
   augmentation agresiva (rotación ±15°, zoom ±10%, desplazamiento ±2px) solo añade dificultad
   de optimización sin un problema real que resolver.
3. **Fashion-MNIST ya viene centrado y recortado de forma consistente** (a diferencia de fotos
   reales de un catálogo, por ejemplo), así que hay menos variación de pose/posición que
   "enseñar a ignorar" con augmentation geométrica — el margen de mejora que esta técnica
   puede aportar aquí es más pequeño de partida que en un dataset con más variabilidad real.

Este es un resultado conocido en la práctica: con un presupuesto de épocas limitado, la
augmentation puede empeorar la accuracy en lugar de mejorarla, porque añade dificultad de
optimización antes de que el beneficio de generalización llegue a notarse.

## Matrices de confusión: dónde falla cada versión

**Baseline (79.20%)**:

![Matriz de confusión — baseline](results/confusion_matrix_baseline.png)

**Con augmentation (72.70%)**:

![Matriz de confusión — augmented](results/confusion_matrix_augmented.png)

En **ambas** versiones, "Camisa" es con diferencia la clase más difícil (26/100 y 15/100
aciertos respectivamente) — es un resultado bien conocido de Fashion-MNIST: una camisa
comparte silueta con la camiseta, el jersey y el abrigo, y sin color ni textura (son imágenes
en escala de grises de 28×28) el contorno es prácticamente toda la información disponible
para distinguirlas.

Lo que sí cambia entre versiones es "Jersey": en el baseline acierta 66/100, pero con
augmentation cae a 37/100, con 53 de esas 100 imágenes confundidas específicamente con
"Abrigo" (que en cambio se mantiene estable, 82/100 baseline vs 83/100 augmented). Es
coherente con el punto 1 de arriba: jersey y abrigo ya se parecen bastante en silueta, y
difuminar esa distinción con rotación/zoom parece haberle costado a la red precisamente la
señal fina que necesitaba para separar esas dos clases, dentro del mismo presupuesto de
entrenamiento.

## Reproducir

```bash
pip install -r ../requirements.txt
python cnn_fashion_mnist.py   # ~11 min en CPU: entrena baseline y augmented, 400 épocas cada uno
```

## Limitaciones

- Solo 2400 imágenes de entrenamiento (de las 60.000 de Fashion-MNIST completo) y sin
  mini-batches (full-batch, igual que el resto del repo) — con más datos y/o más presupuesto
  de épocas es posible que la augmentation sí llegara a superar al baseline; el resultado
  reportado aquí es específico a este presupuesto de cómputo, no una afirmación general de que
  "la augmentation no sirve".
- Muestreo por vecino más cercano en `augmentar_lote()` (no interpolación bilineal como
  Keras/TensorFlow) — más simple de implementar a mano, con una pérdida de calidad de imagen
  menor pero no nula.
- Sin ajuste de hiperparámetros de la augmentation (ángulo, zoom, desplazamiento) — se
  usaron valores razonables por intuición, no se buscó la combinación óptima.
