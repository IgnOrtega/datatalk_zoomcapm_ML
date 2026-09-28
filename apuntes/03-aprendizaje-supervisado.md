# 1.3 — Aprendizaje supervisado

## La idea

En los dos ejemplos anteriores (precio de autos, spam) nosotros actuamos como
**profesores** del modelo: le mostramos ejemplos junto con la respuesta correcta.
De esos ejemplos el algoritmo extrae patrones y después **generaliza** a casos
nuevos.

Eso es «supervisado»: existe una **etiqueta** (el target) que supervisa el
aprendizaje.

## Notación formal

| Símbolo | Nombre | Qué es |
|---------|--------|--------|
| **X** (mayúscula) | Matriz de features (*feature matrix*) | Array 2D: filas = observaciones, columnas = features |
| **y** (minúscula) | Vector de target | Array 1D con el valor a predecir de cada fila |
| **g** | El modelo | La función que entrenamos |

Si hace rato que no ves álgebra: **y** es simplemente un arreglo de números
(`[0, 1, 0, 1, …]`) y **X** es un arreglo de arreglos, o sea una tabla.

## El objetivo

$$g(X) \approx y$$

> Entrenar un modelo es encontrar la función **g** que, aplicada a la matriz de
> features, produzca algo lo más parecido posible al target.

Ese proceso de encontrar `g` es el **entrenamiento**. Todo el curso gira alrededor
de qué forma concreta puede tomar `g`: regresión logística, árboles de decisión,
redes neuronales, etc.

Ejemplo: `g` recibe año, marca y kilometraje, y devuelve un precio lo más cercano
posible al real. Si el precio real era 11.000 y el modelo dijo 15.000, no acertó
exacto, pero puede ser suficientemente cerca para nuestro propósito.

## Los tipos de problema supervisado

Se distinguen **por el tipo de target**.

### Regresión

La salida es un **número**.

- Precio de un auto: de 0 a +∞.
- Precio de una casa a partir de m², habitaciones, distancia al centro y al metro.

En general el rango puede ser cualquiera, incluso de −∞ a +∞.

### Clasificación

La salida es una **categoría**.

- **Multiclase**: clasificar imágenes en auto / gato / perro. Pueden ser 3, 10 o
  1000 categorías.
- **Binaria**: solo dos categorías. Es el caso del spam, y es *el subtipo más
  usado en la práctica*. El modelo devuelve una probabilidad entre 0 y 1, y el
  target es un arreglo de ceros y unos.

> Vas a toparte sí o sí con un problema de clasificación binaria cuando trabajes
> con ML.

### Ranking

Ordenar ítems por un puntaje. No se cubre en profundidad en el curso, pero
conviene saber que existe.

- **Sistemas de recomendación**: se le asigna a cada producto un score de 0 a 1
  (qué tan probable es que a este usuario le guste), se ordena y se muestran los
  primeros N.
- **Buscadores**: Google puntúa los documentos por relevancia (0.9, 0.85…) y
  muestra los más relevantes primero.
- **Búsqueda en e-commerce**: alguien busca «iphone» y hay que mostrarle lo más
  relevante *para esa persona*.

## Resumen

```
        X (feature matrix)  ──►  g (modelo)  ──►  ŷ ≈ y (target)
```

Según el tipo de target:

```
supervisado
├── regresión          → número
├── clasificación      → categoría
│   ├── multiclase
│   └── binaria        ← la más usada
└── ranking            → orden por score
```

El curso se centra sobre todo en **clasificación**, con un capítulo de regresión.

---

**Anterior:** [1.2 — ML vs reglas](02-ml-vs-reglas.md) ·
**Siguiente:** [1.4 — CRISP-DM](04-crisp-dm.md)
