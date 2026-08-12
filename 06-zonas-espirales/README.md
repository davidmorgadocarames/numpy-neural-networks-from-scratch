# 06 — Zonas de espirales: frontera de decisión curva

El benchmark clásico para demostrar que una red con suficientes capas ocultas puede aprender
fronteras de decisión **curvas**, no solo rectas — 3 brazos de una espiral entrelazados, que
ningún clasificador lineal podría separar. Red modular profunda: 2 → 64 (LeakyReLU) → 64
(LeakyReLU) → 3 (Softmax).

## Validación y early stopping: un hallazgo metodológico

Con solo train/test (80%/20%) y esta red de ~9k parámetros, el problema sobreajusta a partir
de cierto punto del entrenamiento: el error sobre el conjunto de test deja de bajar y empieza a
subir mientras el de train sigue cayendo, la señal clásica de que la red memoriza en vez de
generalizar. Sin un tercer conjunto separado no hay forma legítima de detectar ese punto de
corte -- pararía mirando el propio test, y luego reportar la accuracy de ese mismo test sería
hacer trampa. La solución es un split en TRES partes -- 60% train / 20% validación / 20% test,
estratificado por brazo -- con early stopping que decide cuándo parar mirando el error de
VALIDACIÓN, nunca el de test.

Hechos verificados en esta ejecución (`SEED=0`, split 60/20/20): el error de **test** alcanza
su mínimo en la época ~2770 (0.0657) y sube hasta 0.0762 al terminar las 5000 épocas -- el
sobreajuste es real. El error de **validación**, en cambio, sigue bajando de forma monótona
durante las 5000 épocas completas (su mínimo cae en la época 4996, casi al final) y nunca llega
a detectar ese repunte, por lo que el early stopping no se activa en esta ejecución.

La causa es el tamaño de la muestra: con solo 30 puntos de validación por brazo, la señal es
demasiado ruidosa para detectar de forma fiable un sobreajuste que sí está presente en el
conjunto de test (otros 30 puntos, con su propia varianza). Es una limitación conocida del
early stopping basado en validación -- cuantos menos puntos, menos fiable la estimación de
"cuándo empieza a empeorar" -- no un fallo del mecanismo: la validación sigue siendo la única
señal legítima para decidir cuándo parar sin tocar el test, simplemente en esta ejecución
concreta no alcanza a capturar el sobreajuste dentro del presupuesto de 5000 épocas.

Esto no compromete la accuracy reportada: el test nunca participó en ninguna decisión de
entrenamiento. El checkpoint de mejor validación (restaura los pesos de la época con menor
`loss_val`, ver [README raíz](../README.md)) recupera aquí los pesos de la época 4996 --
prácticamente el final, así que en esta ejecución no cambia el resultado, pero es el mismo
mecanismo que sí lo habría corregido si la validación hubiera detectado el repunte a tiempo.

## Resultado

**Accuracy en test: 97.78%** (90 puntos de test, de 450 en total: 270 train / 90 val / 90
test). Entrenamiento hasta el techo de 5000 épocas configurado (un límite de seguridad, no un
objetivo a alcanzar) -- ver arriba por qué la validación no cortó antes.

![Curva de aprendizaje](results/learning_curve.png)

**Visualización de datos — zonas aprendidas**: el fondo de color muestra la frontera curva que
la red dibujó para separar los 3 brazos. Los puntos pequeños son entrenamiento, los triángulos
son validación (deciden cuándo parar, nunca entran en el gradiente) y las estrellas grandes son
test.

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
