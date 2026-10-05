# 2.2 — Preparación de datos

## Descargar y cargar el CSV

El dataset original está en Kaggle, pero hay una copia en el repo
`mlbookcamp-code` (chapter 2, car price). Se copia la URL del archivo y se
descarga con `wget` (o con «Guardar como» en el navegador).

```python
import pandas as pd
import numpy as np

!wget <url-del-csv>        # en el notebook; o descargarlo a mano

df = pd.read_csv('data.csv')
df.head()                  # primeras 5 filas
```

> Hábito del instructor: después de cargar, mirar siempre las primeras filas con
> `head()`.

## Problema 1: nombres de columnas inconsistentes

Algunos nombres tienen mayúsculas, otros no; algunos usan `_` y otros espacios
(`Engine HP`, `Transmission Type`, `MSRP`…).

Con espacios no se puede acceder con punto: `df.Transmission Type` no funciona,
hay que usar `df['Transmission Type']`.

### Solución: minúsculas y `_` en vez de espacios

`df.columns` es un **Index** de pandas. Igual que una Series, tiene el accesor
`.str` para aplicar operaciones de texto a todos los elementos a la vez, y se
pueden encadenar:

```python
df.columns.str.lower()                          # todo a minúsculas
df.columns.str.lower().str.replace(' ', '_')    # además, espacios → _

df.columns = df.columns.str.lower().str.replace(' ', '_')   # guardar el resultado
```

## Problema 2: valores de texto inconsistentes

Lo mismo pasa con los **valores**: a veces en mayúsculas, a veces no. Hay que
normalizarlos, pero `.str` solo funciona sobre columnas de texto, así que primero
hay que encontrarlas.

### Encontrar las columnas de texto

```python
df.dtypes                       # tipo de cada columna
```

| dtype | Ejemplo | ¿Nos interesa? |
|-------|---------|----------------|
| `int64` | year | No |
| `float64` | engine_hp | No |
| `object` | make, model | **Sí** |

> Al leer un CSV, las columnas `object` son en la práctica **strings**.

```python
df.dtypes == 'object'                     # Series de booleanos
df.dtypes[df.dtypes == 'object']          # solo las columnas object
df.dtypes[df.dtypes == 'object'].index    # nos interesan los NOMBRES (el índice)

strings = list(df.dtypes[df.dtypes == 'object'].index)
```

Los valores de esa Series son todos «object», no aportan nada; lo útil es el
**índice**, que contiene los nombres de las columnas. Pasarlo a lista es solo
porque se ve más prolijo.

### Aplicar la misma limpieza a cada columna de texto

```python
for col in strings:
    df[col] = df[col].str.lower().str.replace(' ', '_')

df.head()
```

Resultado: todo en minúsculas y con `_` en vez de espacios, tanto en los nombres
de las columnas como en los valores.

---

**Anterior:** [2.1 — Proyecto: predicción del precio de autos](01-proyecto-precio-autos.md) ·
**Siguiente:** [2.3 — Análisis exploratorio](03-analisis-exploratorio.md)
