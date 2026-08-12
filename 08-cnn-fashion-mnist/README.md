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

## Resultado: la augmentation sigue por detrás del baseline con este presupuesto de épocas

El early stopping de cada versión decide cuándo parar mirando su propio loss de
**validación** — no hace falta igualar nada a mano entre baseline y augmented, cada una para
sola cuando su validación deja de mejorar. En este presupuesto de 400 épocas máximas (techo de
seguridad, no un objetivo), ninguna de las dos activa el early stopping — ambas agotan el
techo, todavía mejorando ligeramente. Los pesos usados para evaluar son los del mínimo de
`loss_val` de cada una, restaurados por checkpoint (ver "Checkpoint del mejor punto de
validación" en el [README raíz](../README.md)):

| | Épocas (corte) | Época del mínimo de validación | Accuracy en test | Loss validación (mínimo) |
|---|---|---|---|---|
| **Baseline** | 400 (todas) | 394 | **79.00%** | 0.4689 |
| **Con augmentation** | 400 (todas) | 370 | **74.80%** | 0.5858 |

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
   llega a compensarlo del todo, aunque el checkpoint sí recupera parte de esa diferencia: la
   versión augmented es la que más se beneficia de quedarse con los pesos del mínimo de
   validación (época 370) en vez de los de la época 400, precisamente porque su curva es más
   ruidosa y el punto final no coincide con el mejor punto.
2. **Apenas había hueco de overfitting que cerrar.** El loss de validación del baseline sigue
   bajando en paralelo al de train hasta el final sin señales de divergencia clara (ver
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
augmentation puede seguir por detrás del baseline en vez de superarlo, porque añade dificultad
de optimización antes de que el beneficio de generalización llegue a notarse del todo.

## Matrices de confusión: dónde falla cada versión

**Baseline (79.00%)**:

![Matriz de confusión — baseline](results/confusion_matrix_baseline.png)

**Con augmentation (74.80%)**:

![Matriz de confusión — augmented](results/confusion_matrix_augmented.png)

En **ambas** versiones, "Camisa" es con diferencia la clase más difícil (38/100 baseline,
13/100 con augmentation) — es un resultado bien conocido de Fashion-MNIST: una camisa comparte
silueta con la camiseta, el jersey y el abrigo, y sin color ni textura (son imágenes en escala
de grises de 28×28) el contorno es prácticamente toda la información disponible para
distinguirlas. Con augmentation ese problema se agrava: 40 de las 100 camisas de test se
confunden con "Jersey", frente a 8 en el baseline.

Lo que más cambia entre versiones es "Jersey" y "Abrigo" — pero, con los pesos ya restaurados
al mínimo de validación, en la dirección contraria a lo que mostraba la versión anterior de
este README (que usaba los pesos de la última época, no los del mejor punto de validación):
"Jersey" **mejora** con augmentation (50/100 → 75/100), mientras "Abrigo" **empeora** (84/100
→ 62/100, con 29 de esas 100 imágenes ahora confundidas específicamente con "Jersey"). Sigue
siendo coherente con el punto 1 de arriba — jersey y abrigo comparten silueta y es la frontera
entre ambos la que más se mueve con la augmentation geométrica — pero el sentido concreto en
que se mueve esa frontera (a favor de una clase y en contra de la otra) depende de qué pesos
exactos se comparen, otra razón más para fijar ese punto de comparación con un criterio
objetivo (el mínimo de validación) en vez de con la última época entrenada.

## Dataset completo + mini-batches — `cnn_fashion_mnist_full.py`

`cnn_fashion_mnist.py` usa full-batch: una única actualización de pesos por época, sobre toda
la muestra de golpe. Manejable con 2.400 imágenes, pero no escala. `cnn_fashion_mnist_full.py`
es la misma arquitectura y la misma metodología (split 60/20/20, early stopping por
validación, checkpoint del mejor punto, baseline vs augmented con los mismos pesos iniciales),
entrenada en cambio por **mini-batches de 32 imágenes** (1.313 por época) sobre las **70.000**
imágenes completas de Fashion-MNIST (exactamente 7.000 por clase — a diferencia de MNIST, aquí
sí está perfectamente equilibrado).

Mismo error clásico a vigilar que en `07-reconocimiento-digitos/digit_classifier_full.py`: el
gradiente se normaliza por el tamaño del **batch actual**, no por el de todo train — dejar el
denominador del full-batch original habría hecho que el entrenamiento no convergiera, sin
lanzar ningún error explícito.

| | Train / Val / Test | Accuracy baseline | Accuracy augmented | Tiempo total |
|---|---|---|---|---|
| Muestra reducida, full-batch | 2.400 / 600 / 1.000 | 79.00% | 74.80% | ~11 min (800 épocas en total) |
| Dataset completo, mini-batch | 42.000 / 14.000 / 14.000 | **88.34%** | **85.24%** | ~6,5 min (33 épocas en total) |

Dos cosas notables:

1. **Con 17,5x más datos, el entrenamiento tardó *menos* tiempo real**, no más — la versión
   full-batch necesitaba 400 épocas por versión (800 en total) para converger con solo 1
   actualización de pesos cada una; con mini-batches (1.313 actualizaciones por época) la red
   converge en 13 épocas (baseline) y 20 (augmented), 33 en total. Cada actualización individual
   es más barata (procesa 32 imágenes, no 2.400), y hacen falta muchísimas menos épocas
   completas para llegar al mismo sitio.
2. **La augmentation sigue por detrás del baseline (85.24% < 88.34%), pero la brecha se
   estrecha según crece la muestra**: 4,20 puntos con la muestra reducida (79.00% - 74.80%)
   frente a 3,10 puntos con el dataset completo (88.34% - 85.24%). Es la misma lectura que en
   la sección anterior, ahora con datos: más muestra ayuda a que el coste de optimización de la
   augmentation (punto 1 de la sección anterior) pese relativamente menos, aunque con este
   presupuesto de épocas (early stopping, no un número fijo) tampoco llega a compensarlo del
   todo.

![Curva de aprendizaje (dataset completo)](results_full/learning_curve.png)

**Matrices de confusión (test, dataset completo)**:

**Baseline (88.34%)**:

![Matriz de confusión — baseline (dataset completo)](results_full/confusion_matrix_baseline.png)

**Con augmentation (85.24%)**:

![Matriz de confusión — augmented (dataset completo)](results_full/confusion_matrix_augmented.png)

**Lectura**: "Camisa" sigue siendo, con diferencia, la clase más difícil en ambas versiones
(68.6% baseline, 41.1% con augmentation, sobre 1.400 camisas de test) — el mismo resultado
conocido de Fashion-MNIST que en la muestra reducida, ahora con mucha más muestra para
confirmarlo: camisa comparte silueta con camiseta, jersey y abrigo, y sin color ni textura el
contorno es toda la información disponible. Con augmentation la confusión se dispara: 247 de
1.400 camisas se leen como "Jersey" (17,6%) y 237 como "Abrigo" (16,9%), frente a 117 y 74
respectivamente en el baseline — la augmentation geométrica parece difuminar precisamente los
detalles de solapa/cuello que distinguen una camisa de un jersey o un abrigo.

## ¿Ayuda o perjudica el flip horizontal? Estudio con y sin flip, a dos escalas

La augmentation de este proyecto combina 4 transformaciones (`capas_cnn.augmentar_lote`): flip
horizontal, rotación, zoom y desplazamiento. A diferencia de dígitos manuscritos, varias
prendas de Fashion-MNIST no son simétricas en la práctica -- zapatillas, sandalias y botines
tienen una orientación de puntera concreta -- así que tenía sentido dudar si el flip
específicamente ayuda o perjudica, en vez de asumirlo. Se entrenó una tercera variante,
`augmented_sin_flip` (prob_flip=0.0, mismo resto de augmentation), con la misma semilla de
augmentation que `augmented` -- como `augmentar_lote` consume los mismos números aleatorios en
el mismo orden con independencia de `prob_flip`, las rotaciones/zooms/desplazamientos son
idénticos entre ambas variantes y la única diferencia real es si se aplica el flip.

| | Baseline | Augmented (flip=0.5) | Augmented sin flip (flip=0.0) |
|---|---|---|---|
| Muestra reducida (2.400 train) | 79.00% | 74.80% | **75.70%** |
| Dataset completo (42.000 train) | 88.34% | **85.24%** | 84.85% |

La respuesta no es un simple "ayuda" o "perjudica": **a escala reducida, quitar el flip mejora
el resultado** (+0,90 puntos); **a escala completa, el flip ayuda ligeramente** (+0,39 puntos)
-- el efecto neto se invierte entre las dos escalas, y en ambos casos la diferencia es pequeña
frente a la brecha real (la que separa augmented de baseline). Mirar solo la accuracy total
esconde lo que sí es un patrón consistente por clase:

- **"Camisa" es mucho peor con flip, en las dos escalas**: 13% con flip vs 23% sin flip
  (muestra reducida); 41,1% con flip vs 48,7% sin flip (dataset completo). El flip agrava
  justo la confusión que ya era el punto más débil del modelo (ver más arriba) -- probablemente
  porque muchas camisas de Fashion-MNIST llevan el bolsillo, la solapa o el estampado en un
  lado concreto, y el flip le quita a la red esa asimetría como pista.
- **"Jersey" es mucho mejor con flip, en las dos escalas**: 75% con flip vs 52% sin flip
  (muestra reducida); 79,2% con flip vs 70,9% sin flip (dataset completo). Aquí el flip parece
  aportar variedad útil sin destruir ninguna asimetría relevante (un jersey es visualmente casi
  simétrico).
- **El resto de clases (incluidas las asimétricas por diseño -- Sandalia, Zapatilla, Botín)
  no muestran una dirección consistente entre las dos escalas**: por ejemplo Zapatilla mejora
  sin flip a escala reducida (86% vs 84%) pero empeora sin flip a escala completa (89,9% vs
  94,6%) -- el signo se invierte, lo que apunta a ruido de muestra (100 imágenes de test por
  clase a escala reducida, 1.400 a escala completa, pero aun así el efecto del flip en estas
  clases es pequeño frente a la varianza entre ejecuciones) más que a un efecto real y estable.

**Conclusión honesta**: el flip horizontal no es universalmente bueno ni malo para Fashion-
MNIST con esta arquitectura -- tiene un efecto real, grande y consistente en al menos una
clase concreta (perjudica a "Camisa") y el efecto contrario en otra ("Jersey"), pero no hay
una regla simple tipo "las prendas asimétricas se ven perjudicadas por el flip" que se sostenga
al mirar sandalias/zapatillas/botines en las dos escalas. Si el objetivo fuera maximizar
accuracy en una clase concreta (p. ej. "Camisa"), desactivar el flip sería una palanca real a
probar; como métrica global, el efecto del flip es del mismo orden que el ruido entre
ejecuciones.

## Reproducir

```bash
pip install -r ../requirements.txt
python cnn_fashion_mnist.py        # muestra reducida, full-batch (~15,4 min: 400 épocas x 3 versiones)
python cnn_fashion_mnist_full.py   # dataset completo, mini-batch (~10,8 min: 33 épocas x 3 versiones)
```

## Limitaciones

- `cnn_fashion_mnist.py` usa solo 2.400 imágenes de entrenamiento (de las 42.000 disponibles
  con el split completo) y full-batch, precisamente para que el bucle de entrenamiento sea
  simple de leer sin mini-batches de por medio — es una limitación intencionada, no un límite
  real: `cnn_fashion_mnist_full.py` entrena sobre el dataset completo con mini-batches y sube
  el baseline a 88.34% y el augmented a 85.24% (ver sección "Dataset completo + mini-batches"
  arriba). Con más muestra la brecha entre baseline y augmented se estrecha (4,20 → 3,10
  puntos) pero no se cierra dentro de este presupuesto de épocas.
- Muestreo por vecino más cercano en `augmentar_lote()` (no interpolación bilineal como
  Keras/TensorFlow) — más simple de implementar a mano, con una pérdida de calidad de imagen
  menor pero no nula.
- Sin ajuste de hiperparámetros de la augmentation (ángulo, zoom, desplazamiento) — se
  usaron valores razonables por intuición, no se buscó la combinación óptima.
