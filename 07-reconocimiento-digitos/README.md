# 07 — Reconocimiento de dígitos manuscritos (MNIST), 100% NumPy

El proyecto más avanzado del conjunto: reconoce dígitos manuscritos (0-9) del dataset MNIST
con una red neuronal escrita **completamente desde cero en NumPy** — sin TensorFlow, sin
Keras, sin PyTorch. Red modular 784 → 128 (LeakyReLU) → 10 (Softmax), usando el mismo
mini-framework de [`../capas.py`](../capas.py) que el resto de proyectos de este repo.

Split en tres partes estratificado por dígito -- train / validación / test -- con early
stopping que decide cuándo parar mirando el error de VALIDACIÓN, nunca el de test. El test se
toca una única vez, al final, con la red ya congelada, para reportar una accuracy y una matriz
de confusión honestas sobre dígitos nunca vistos en el entrenamiento.

## 1. Entrenamiento — `digit_classifier.py`

Entrena la red sobre 1200 imágenes (120 por dígito), valida sobre 300 (30 por dígito, deciden
el early stopping) y evalúa sobre otras 300 de test (30 por dígito, nunca vistas hasta la
evaluación final). Guarda los pesos entrenados en `results/red_pesos.npz` para que la demo
interactiva no tenga que reentrenar.

**Resultado**: **89.00% accuracy en test**. El early stopping corta en la época 695 de las 800
configuradas (techo de seguridad, no un objetivo), pero los pesos usados para evaluar son los
de la época 633 -- el mínimo real de `loss_val`, restaurado por checkpoint (ver "Checkpoint del
mejor punto de validación" en el [README raíz](../README.md); el early stopping necesita ver
200 épocas sin mejora para confirmar el corte, así que el punto en que corta siempre queda por
detrás del mínimo real). Con una red minúscula (784→128→10, ~101k parámetros) entrenada sobre
solo 1200 imágenes y sin ningún tipo de aumento de datos ni regularización. Antes, sin
validación, la red entrenaba las 800 épocas completas y llegaba a 91.67% en test, pero ese
número no era comparable con nada: no había forma honesta de saber si a la época 800 la red ya
estaba sobreajustando, porque el propio test se usaba de facto como referencia implícita para
fijar `epochs=800`. El 89.00% actual es más bajo pero es la cifra en la que se puede confiar,
porque la decisión de parar la tomó la validación sin haber mirado nunca el test.

![Curva de aprendizaje](results/learning_curve.png)

**Visualización de datos** — muestra del dataset de entrenamiento:

![Muestra de dígitos](results/data_visualization.png)

**Matriz de confusión (test)**:

![Matriz de confusión](results/confusion_matrix.png)

**Lectura de la matriz**: la diagonal domina (267 de 300 aciertos) y la mayoría de errores son
casos sueltos, lo esperable con solo 1200 imágenes de entrenamiento, sin aumento de datos y
parando el entrenamiento en cuanto la validación deja de mejorar. Dos patrones destacan:

- **"9" es el dígito más difícil** (23/30 = 76.7%, el peor de los 10): se confunde sobre todo
  con "3" (3 casos) y "7" (3 casos). Tiene sentido en trazos manuscritos donde el lazo superior
  del 9 queda poco cerrado o el trazo vertical se confunde con el travesaño del 7.
- **"4"→"9" es la confusión individual más repetida (6 de 30 cuatros)**: un "4" con el trazo
  superior cerrado o desplazado se parece mucho a un "9" en letra manuscrita — la red pequeña,
  entrenada con pocos datos y parada pronto por el early stopping, no llega a aprender el
  detalle fino (el ángulo abierto del 4 frente al lazo cerrado del 9) que distingue ambos casos.

Con una red más grande, más datos o aumento de datos (rotaciones/desplazamientos pequeños)
estos casos límite mejorarían.

## 2. Dataset completo + mini-batches — `digit_classifier_full.py`

`digit_classifier.py` usa full-batch: una única actualización de pesos por época, sobre toda la
muestra de golpe. Es manejable con 1.200 imágenes, pero no escala -- con las 70.000 imágenes
completas de MNIST, seguir haciendo full-batch significaría 1 sola actualización de pesos por
época sobre una matriz enorme. `digit_classifier_full.py` es la misma red y la misma
metodología (split 60/20/20, early stopping por validación, checkpoint del mejor punto),
entrenada en cambio por **mini-batches de 32 imágenes** (~1.313 actualizaciones de pesos por
época) sobre el dataset **completo**, estratificado 60/20/20 por dígito usando las 70.000
imágenes reales de MNIST (no perfectamente equilibrado entre dígitos: entre 6.313 y 7.877
imágenes según el dígito).

Detalle importante al convertir full-batch en mini-batch: el gradiente de la capa de salida se
normaliza por el tamaño del **batch actual** (`Yb.shape[0]`), no por el tamaño de todo el
conjunto de train — dejar el denominador del full-batch original es un error clásico y
silencioso (ningún except salta, el entrenamiento simplemente converge mal o absurdamente
despacio).

| | Train | Val | Test | Accuracy test |
|---|---|---|---|---|
| Muestra reducida, full-batch (`digit_classifier.py`) | 1.200 | 300 | 300 | 89.00% |
| Dataset completo, mini-batch (`digit_classifier_full.py`) | 41.995 | 13.996 | 14.009 | **97.46%** |

La mejora (+8.5 puntos) confirma que el límite real de la versión reducida no era la
arquitectura (784→128→10 sigue siendo la misma red en ambos casos) sino la combinación de poca
muestra y una sola actualización de pesos por época. Con mini-batches, la red converge además
en muchísimas menos épocas: early stopping en la época 19 (de 30 configuradas), con los pesos
restaurados de la época 15 — frente a las 695 épocas (de 800) que necesitaba la versión
full-batch, porque cada "época" aquí ya contiene ~1.313 pasos de descenso de gradiente en vez
de 1.

![Curva de aprendizaje (dataset completo)](results_full/learning_curve.png)

**Matriz de confusión (test, dataset completo)**:

![Matriz de confusión (dataset completo)](results_full/confusion_matrix.png)

**Lectura**: con 14.009 imágenes de test, la diagonal domina de forma mucho más uniforme que en
la versión reducida — el peor dígito pasa de 76.7% (el "9" en la muestra pequeña) a **95.6%**
(sigue siendo el "9" el más difícil, pero ahora seguido de cerca por "5" al 96.8% y "8" al
97.0%, no un salto brusco). La confusión "4"→"9" que era la más repetida en la versión
reducida (6 de 30 cuatros, 20%) sigue presente en términos absolutos (14 de 1.366 cuatros), pero
ahora es solo un 1.0% de los cuatros de test — el mismo patrón de confusión real (rasgos
compartidos entre 4 y 9 mal trazados) sigue ahí, simplemente con muchos menos casos porque la
red tiene mucha más muestra de la que aprender el detalle fino que los distingue.

## 3. Demo interactiva — `demo_gradio.py`

Carga los pesos ya entrenados (de `digit_classifier.py`, la versión reducida) y abre un
Sketchpad donde se puede dibujar un dígito a mano para clasificarlo en vivo. Reutiliza el mismo
preprocesado de centrado por centro de masa (caja 20x20 dentro de un lienzo 28x28) que replica
cómo está construido MNIST realmente — sin ese centrado, un trazo dibujado a mano y simplemente
reescalado queda muy descentrado respecto a los datos de entrenamiento y la precisión cae mucho
aunque la red esté bien entrenada.

![Demo interactiva: dibujar un dígito y clasificarlo con la red NumPy](results/demo_sketchpad.gif)

## Robustez frente a la semilla

Repitiendo cada variante con **10 pares (seed_split, seed_modelo) sorteados de forma
independiente** (`python run_seed_sweep.py --solo 07-digitos-fullbatch` /
`--solo 07-digitos-minibatch`, ver [README raíz](../README.md)):

| Variante | Accuracy media ± desv. típica | Rango (10 semillas) |
|---|---|---|
| Muestra reducida, full-batch | 88.50% ± 1.45% | 86.33%–91.00% |
| Dataset completo, mini-batch | 97.46% ± 0.21% | 97.18%–97.73% |

![Robustez — muestra reducida, full-batch](results/seed_sweep.png)

![Pérdida por época — muestra reducida, full-batch](results/seed_sweep_curvas.png)

![Robustez — dataset completo, mini-batch](results_full/seed_sweep.png)

![Pérdida por época — dataset completo, mini-batch](results_full/seed_sweep_curvas.png)

La versión reducida (1.200 imágenes) tiene ~7 veces más varianza entre semillas que la del
dataset completo — coherente con lo esperable: con menos datos, tanto la inicialización como
qué 300 imágenes concretas caen en test pesan proporcionalmente más en el resultado final. La
comparación de la sección 2 (+8.5 puntos a favor del dataset completo) es robusta a la semilla:
incluso en su peor caso (86.33%), la versión reducida sigue muy por debajo del peor caso de la
versión completa (97.18%) — los rangos ni siquiera se solapan.

## Reproducir

```bash
pip install -r ../requirements.txt
python digit_classifier.py        # versión reducida, full-batch (~1-2 min, descarga MNIST la primera vez)
python digit_classifier_full.py   # dataset completo, mini-batch (~pocos minutos, ver README)
python demo_gradio.py             # demo interactiva, requiere haber ejecutado antes digit_classifier.py
```

## Limitaciones

- `digit_classifier.py` usa solo 1.200 imágenes de entrenamiento (frente a las ~42.000
  disponibles con el split completo) precisamente para poder entrenar full-batch en CPU sin
  frameworks optimizados de forma rápida y sencilla de leer — es una limitación intencionada
  para ilustrar el mecanismo con claridad, no un límite real: `digit_classifier_full.py`
  entrena sobre el dataset completo con mini-batches y consigue 97.46% frente al 89.00% de la
  versión reducida (ver sección 2 arriba).
- Sin aumento de datos, regularización (dropout) ni ajuste de hiperparámetros.
