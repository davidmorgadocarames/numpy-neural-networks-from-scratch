# 05 — Predicción del precio de una casa

Predice el precio de una vivienda a partir de 2 variables: metros cuadrados y número de
habitaciones. Red modular (Densa → LeakyReLU → Densa lineal) con un split train / validación /
test para poder comparar el error de train contra el de **validación** época a época y
detectar overfitting — si el error de validación empezara a subir mientras el de train sigue
bajando, sería la señal de que la red está memorizando en vez de generalizar. El conjunto de
test no participa en esa decisión: se evalúa una sola vez, al final, con la red ya congelada
(ver "Metodología" en el [README raíz](../README.md) para el porqué de separar validación de
test).

Es un problema de **regresión** (predice un precio en euros), así que no aplica una matriz de
confusión.

## Resultado

De las 150 casas: 90 de entrenamiento (60%), 30 de validación (20%, decide el early stopping)
y 30 de test (20%, evaluadas una sola vez):

- **MAE en test: 11.452 €** — coherente con el ruido gaussiano (desviación 15.000 €) añadido
  deliberadamente a los datos sintéticos, lo que indica que la red aprendió la relación real
  precio = f(metros, habitaciones) tan bien como el ruido de los datos permite.

![Curva de aprendizaje](results/learning_curve.png)

### Early stopping: parar en cuanto deja de mejorar de verdad

Con 4000 épocas configuradas (techo de seguridad, no un objetivo), el entrenamiento **corta en
la época 2362** — el error de **validación** lleva 200 épocas sin bajar al menos un 0.5%, así
que seguir no aporta nada. Los pesos usados para evaluar son los de la época **2361**, el
mínimo real de `loss_val`, restaurado por checkpoint (ver "Checkpoint del mejor punto de
validación" en el [README raíz](../README.md)) — prácticamente el mismo punto que el de corte,
porque aquí el error de validación deja de mejorar de forma bastante abrupta.

Esto es distinto de lo que pasa en [`02-celsius-fahrenheit`](../02-celsius-fahrenheit/): ahí los
datos no tienen ruido y el error sigue bajando (aunque de forma invisible en una gráfica lineal)
indefinidamente, así que el early stopping nunca se activa. Aquí sí hay ruido gaussiano real en
los precios (desviación 15.000 €), así que existe un suelo de error que no se puede bajar por
mucho que se entrene — en cuanto la red lo alcanza, seguir entrenando no mejora nada. El
criterio concreto: se compara el error de validación actual contra el de 200 épocas atrás, y si
la mejora relativa en toda esa ventana es menor al 0.5% se corta el entrenamiento
(`epochs_entrenadas` en `results/metrics.json` guarda la época real de corte, `epoca_mejor_val`
la de los pesos realmente usados).

**Visualización de datos — mapa de tasación aprendido**: el fondo de color muestra el precio
que la red asignaría a cualquier combinación de metros/habitaciones. Los puntos son las casas
reales (círculos = train, triángulos = validación, estrellas = test), coloreados por su precio
real — si el punto tiene un color parecido al fondo que pisa, la red acertó.

![Mapa de tasación](results/data_visualization.png)

**"Matriz" (equivalente para regresión)**: no aplica confusión. Predicho vs real en el
conjunto de test:

![Predicho vs real](results/predicted_vs_real.png)

## Reproducir

```bash
pip install -r ../requirements.txt
python house_price.py
```

## Limitaciones

- Datos sintéticos (fórmula lineal conocida + ruido gaussiano), no precios reales de mercado.
