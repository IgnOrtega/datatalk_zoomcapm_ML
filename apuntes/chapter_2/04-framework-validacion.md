# 2.4 — Framework de validación

Como vimos en [1.5 — Selección de modelos](../chapter_1/05-seleccion-de-modelos.md),
para validar hay que dividir el dataset en **tres partes**:

```
┌───────────────────────┬───────────┬───────────┐
│      train 60 %       │  val 20 % │ test 20 % │
└───────────────────────┴───────────┴───────────┘
```

- **train**: para entrenar.
- **validation**: para comprobar si el modelo funciona bien.
- **test**: se deja para el final; se usa muy de vez en cuando, solo para
  confirmar que el modelo anda bien.

De cada parte se obtiene una matriz de features `X` y un target `y`:
`X_train, y_train`, `X_val, y_val`, `X_test, y_test`.

En esta clase se hace todo **a mano**, con pandas y NumPy, sin librerías extra.

## Calcular los tamaños

El dataset tiene casi 12 000 filas → el 20 % son unas 2 400. Los tamaños tienen
que ser **enteros**, así que se convierten con `int`:

```python
n = len(df)

n_val = int(n * 0.2)
n_test = int(n * 0.2)
n_train = int(n * 0.6)

n_val + n_test + n_train == n    # False: por el redondeo no suman n
```

Por el redondeo, la suma queda **por debajo de `n`** y se perderían algunas filas.
La solución: train se queda con todo lo que sobra.

```python
n_train = n - n_val - n_test

n_val, n_test, n_train    # (2382, 2382, 7150)
```

## Slicing con `iloc`

`iloc` acepta rangos (el límite superior es **exclusivo**):

```python
df.iloc[:10]      # filas 0 a 9
df.iloc[10:20]    # filas 10 a 19
df.iloc[10:]      # desde la 10 hasta el final
```

Primer intento (secuencial):

```python
df_train = df.iloc[:n_train]
df_val = df.iloc[n_train:n_train + n_val]
df_test = df.iloc[n_train + n_val:]
```

> En clase primero se arma con validation al principio y después se reordena a
> train → val → test, que es más lógico.

### Problema: los datos están ordenados

Si se corta en orden, por ejemplo todos los BMW pueden caer en un solo conjunto
y no aparecer en train. Por eso hay que **mezclar** (shuffle).

> En general siempre conviene mezclar los datos, por si existe algún orden
> accidental que haya que romper.

## Mezclar con un índice aleatorio

`iloc` también acepta una secuencia arbitraria de posiciones y devuelve las filas
en ese orden. Entonces: generar `0 … n-1`, mezclarlo y usarlo para indexar.

```python
idx = np.arange(n)
np.random.shuffle(idx)

df.iloc[idx[:10]]      # las primeras 10 filas en orden mezclado
```

### Reproducibilidad: semilla

Sin semilla, cada ejecución da una mezcla distinta. Con una semilla fija, el
resultado es el mismo (con la misma versión de NumPy):

```python
np.random.seed(2)

idx = np.arange(n)
np.random.shuffle(idx)

df_train = df.iloc[idx[:n_train]]
df_val = df.iloc[idx[n_train:n_train + n_val]]
df_test = df.iloc[idx[n_train + n_val:]]

len(df_train), len(df_val), len(df_test)
```

## Reiniciar el índice

Después de mezclar, cada DataFrame conserva el índice original (desordenado). No
lo necesitamos, así que se resetea y se descarta:

```python
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
```

## Separar el target

Se aplica la transformación logarítmica al precio y se toma directamente el
**array de NumPy** con `.values` (no hacen falta el índice ni el resto de la
Series):

```python
y_train = np.log1p(df_train.msrp.values)
y_val = np.log1p(df_val.msrp.values)
y_test = np.log1p(df_test.msrp.values)
```

## Borrar el target de los DataFrames

```python
del df_train['msrp']
del df_val['msrp']
del df_test['msrp']
```

> Consejo del instructor: siempre borrar el target después de separarlo. Si queda
> en el DataFrame, se puede usar por accidente como feature para predecir el
> precio: el modelo sale «perfecto» y uno pierde mucho tiempo buscando qué anda
> mal. Dice que le pasó muchas veces.

Con esto los datos quedan listos para entrenar. Próximo paso: **regresión lineal**.

---

**Anterior:** [2.3 — Análisis exploratorio](03-analisis-exploratorio.md) ·
**Siguiente:** [2.5 — Regresión lineal](05-regresion-lineal.md)
