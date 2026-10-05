# 2.12 — Variables categóricas

Son variables cuyos valores son **categorías**, normalmente strings: marca
(`make`), modelo, tipo de combustible, tipo de transmisión, tracción, etc.

```python
df_train.dtypes    # las columnas de tipo object son categóricas
```

## Una categórica disfrazada: `number_of_doors`

Parece numérica, pero en realidad es categórica: autos de 2, 3 y 4 puertas son
**tipos distintos** de auto. Pandas la trata como número solo porque sus valores
son números.

> Puede ser importante: un auto de 2 puertas probablemente sea más caro que uno
> de 4.

## One-hot encoding

Una columna categórica se representa con **varias columnas binarias**, una por
valor:

| number_of_doors | num_doors_2 | num_doors_3 | num_doors_4 |
|---|---|---|---|
| 2 | 1 | 0 | 0 |
| 3 | 0 | 1 | 0 |
| 4 | 0 | 0 | 1 |
| 2 | 1 | 0 | 0 |

En Pandas, con una comparación `==` y pasando el booleano a entero:

```python
(df_train.number_of_doors == 2).astype('int')   # True/False → 1/0
```

En vez de escribir una línea por cada valor, un loop con un template de string:

```python
for v in [2, 3, 4]:
    df_train['num_doors_%s' % v] = (df_train.number_of_doors == v).astype('int')
```

`%s` se reemplaza por el valor de `v`.

## Agregarlo a `prepare_X`

```python
def prepare_X(df):
    df = df.copy()
    features = base.copy()

    df['age'] = 2017 - df.year
    features.append('age')

    for v in [2, 3, 4]:
        df['num_doors_%s' % v] = (df.number_of_doors == v).astype('int')
        features.append('num_doors_%s' % v)

    df_num = df[features]
    df_num = df_num.fillna(0)
    X = df_num.values
    return X
```

> Si se hace `append` directamente sobre `base`, cada llamada a la función le
> agrega `age` (y las demás) **otra vez** a la lista global. Por eso se trabaja
> sobre una copia (`features = base.copy()`), igual que con el DataFrame.

Resultado: la mejora es **casi despreciable**. El número de puertas no aporta
mucho.

## Agregar la marca (`make`)

`make` tiene muchos valores distintos (`df.make.nunique()`), así que se toman
solo los más populares:

```python
df.make.value_counts().head()            # top 5 marcas con su frecuencia

makes = list(df.make.value_counts().head().index)
# ['chevrolet', 'ford', 'volkswagen', 'toyota', 'dodge']
```

`value_counts()` devuelve los valores en el **índice**; con `.index` se obtienen
y se envuelven en una lista de Python.

```python
    for v in makes:
        df['make_%s' % v] = (df.make == v).astype('int')
        features.append('make_%s' % v)
```

Se agregan 5 columnas nuevas. El RMSE baja alrededor de un 1 %: no tan
drástico como con `age`, pero está bien.

## Todas las categóricas

Mirando de nuevo `dtypes`: `make`, `engine_fuel_type`, `transmission_type`,
`driven_wheels`, `market_category`, `vehicle_size`, `vehicle_style`.

> El modelo (`model`) **no** se incluye: tiene demasiados valores.

Primero, un diccionario con los 5 valores más comunes de cada una:

```python
categorical_variables = [
    'make', 'engine_fuel_type', 'transmission_type', 'driven_wheels',
    'market_category', 'vehicle_size', 'vehicle_style'
]

categories = {}

for c in categorical_variables:
    categories[c] = list(df[c].value_counts().head().index)
```

Después, dos loops anidados en `prepare_X` (sobre los pares clave–valor con
`items()`):

```python
    for c, values in categories.items():
        for v in values:
            df['%s_%s' % (c, v)] = (df[c] == v).astype('int')
            features.append('%s_%s' % (c, v))
```

El primer `%s` se reemplaza por `c` (la variable) y el segundo por `v` (el valor).

## ¡Se rompió!

| Modelo | RMSE (validación) |
|---|---|
| Base + `age` | ≈ 0.51 |
| + todas las categóricas | ≈ 41 |

Las predicciones son números enormes (negativos, en notación científica) y
algunos pesos `w` también son gigantes.

> Queríamos mejorar el modelo agregando variables y lo empeoramos. En la próxima
> lección se ve por qué pasa y cómo arreglarlo.

---

**Anterior:** [2.11 — Feature engineering](11-feature-engineering.md) ·
**Siguiente:** [2.13 — Regularización](13-regularizacion.md)
