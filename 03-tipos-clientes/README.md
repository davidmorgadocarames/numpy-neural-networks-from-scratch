# 03 — Clasificador de tipos de cliente (3 categorías)

Clasifica clientes de una tienda online en 3 categorías (Navegadores, Ocasionales, VIPs) a
partir de 2 variables: minutos navegando y productos en el carrito. Red modular
(Densa → LeakyReLU → Densa → Softmax) entrenada con entropía cruzada.

Split en tres partes -- 60% train / 20% validación / 20% test, estratificado por categoría --
con early stopping que decide cuándo parar mirando el error de VALIDACIÓN, nunca el de test.
El test se toca una única vez, al final, con la red ya congelada, para medir generalización
real con una matriz de confusión sobre clientes nunca vistos en el entrenamiento.

## Bug corregido: faltaba normalizar los datos

Una primera versión de este proyecto pasaba minutos y productos a la red **sin normalizar**,
con valores de hasta ~50. Con esa escala, el descenso de gradiente converge tan despacio que
ni siquiera llegaba a separar bien las clases (~93% de acierto evaluado sobre los propios
datos de entrenamiento). El síntoma era confundir "Ocasionales" con "VIPs" en la matriz de
confusión, pese a que las 3 categorías **no se solapan en ninguna de las 2 variables**
(minutos: 2–10 / 15–25 / 30–45; productos: 0–10 / 17–27 / 36–51) — con clases así de
separadas, una red que converge bien no debería fallar nunca. Añadiendo una normalización
min-max estándar (con el mínimo/máximo del conjunto de train, nunca del de test, para no
filtrar información) y sin tocar ningún otro hiperparámetro, la red pasa de ~90% a **100% de
accuracy en test**.

## Resultado

**Accuracy en test: 100%** (24 clientes de test, de 120 en total: 72 train / 24 val / 24
test), tras 3000 épocas -- el problema está tan bien separado que el early stopping no llega
a activarse, la red converge sin overfitting dentro del propio presupuesto de épocas.

![Curva de aprendizaje](results/learning_curve.png)

**Visualización de datos — zonas aprendidas**: el fondo de color muestra en qué categoría
clasificaría la red cualquier punto del plano. Los triángulos son los clientes de validación
(deciden cuándo parar, pero nunca entran en el gradiente) y las estrellas amarillas son los
clientes de test (nunca vistos hasta la evaluación final) — todos caen dentro de la zona de su
color correcto, y la frontera entre zonas pasa limpiamente por el hueco vacío entre las 3 nubes
de puntos.

![Zonas de clientes](results/data_visualization.png)

**Matriz de confusión (test)**: con clases perfectamente separables y la red ya convergida,
la matriz es una diagonal perfecta — cada uno de los 8 clientes de test de cada categoría cae
en su celda "real = predicción" (Navegadores→Navegadores, Ocasionales→Ocasionales,
VIPs→VIPs) y las celdas fuera de la diagonal están todas a 0, es decir, cero errores:

![Matriz de confusión](results/confusion_matrix.png)

## Robustez frente a la semilla

Con clases tan bien separadas cabría esperar 100% de accuracy con cualquier semilla. Repitiendo
el entrenamiento con **10 pares (seed_split, seed_modelo) sorteados de forma independiente**
(`python run_seed_sweep.py --solo 03-clientes`, ver [README raíz](../README.md)):

| Métrica | Media | Desv. típica | Mínimo | Máximo | N semillas |
|---|---|---|---|---|---|
| Accuracy en test | 93.75% | 19.76% | 37.50% | 100% | 10 |

![Robustez frente a la semilla](results/seed_sweep.png)

![Pérdida por época, las 10 semillas superpuestas](results/seed_sweep_curvas.png)

9 de las 10 semillas llegan a 100%, pero una colapsa a 37.5% — es la curva que corta en seco
poco después de la época 200 en la gráfica de arriba, mientras las otras 9 siguen bajando hasta
la 3000. Se diagnosticó la semilla que falla ejecutándola de forma aislada: el early stopping
se activa en la **época 201** — el corte más temprano posible, dado `PACIENCIA_EARLY_STOP=200`
— con `loss_val=1.10` (frente a ~0.19 en una ejecución normal de 3000 épocas). No es un
problema de separabilidad de los datos (la matriz de confusión de esa ejecución,
`[[0,8,0],[2,6,0],[5,0,3]]`, muestra una red que apenas ha empezado a diferenciar clases, no
una que las confunda genuinamente): con esa inicialización concreta, la mejora de la validación
en las primeras 200 épocas es tan lenta que no supera el umbral relativo del 0.5%, y el
criterio de parada (una ventana de épocas fija, no adaptativa) corta el entrenamiento casi
antes de empezar. Es una limitación real del criterio de early stopping tal como está
implementado aquí (ventana absoluta de épocas, igual para cualquier velocidad de convergencia
inicial), no del dataset ni de la arquitectura.

## Reproducir

```bash
pip install -r ../requirements.txt
python customer_classifier.py
```

## Limitaciones

- Dataset sintético con categorías generadas por rangos con algo de solape intencionado (para
  que el problema no sea trivial) — no son datos reales de una tienda.
- El criterio de early stopping (ventana fija de `PACIENCIA_EARLY_STOP` épocas) puede cortar
  prematuramente si una inicialización concreta arranca con mejora lenta en la validación —
  medido arriba: 1 de 10 semillas corta en la época 201 con un resultado muy por debajo del
  resto. Una ventana adaptativa (o un mínimo de épocas antes de poder activarse) mitigaría esto,
  pero no se ha implementado para mantener el criterio idéntico al resto del repositorio.
