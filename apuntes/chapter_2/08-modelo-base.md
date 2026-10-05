# 2.8 — Modelo base (baseline)

Se usa `train_linear_regression` de la lección anterior para construir un
primer modelo de precios: el **baseline**.

## Elegir las features

Para el baseline se usan **solo las columnas numéricas** (se revisan con
`df.dtypes`):

| Columna            | Significado                     |
|--------------------|---------------------------------|
| `engine_hp`        | caballos de fuerza del motor    |
| `engine_cylinders` | número de cilindros             |
| `highway_mpg`      | millas por galón en carretera   |
| `city_mpg`         | millas por galón en ciudad      |
| `popularity`       | popularidad (menciones en Twitter) |

```python
base = ['engine_hp', 'engine_cylinders', 'highway_mpg',
        'city_mpg', 'popularity']

X_train = df_train[base].values     # .values → array de NumPy
```

## Valores faltantes

Al entrenar aparecen `NaN` en el resultado: hay **missing values** en
`engine_hp` y `engine_cylinders`.

```python
df_train[base].isnull().sum()
```

Lo más simple es rellenarlos con cero:

```python
X_train = df_train[base].fillna(0).values

w0, w = train_linear_regression(X_train, y_train)
```

### ¿Qué significa rellenar con 0?

Con dos features, si `xᵢ₁` falta y se reemplaza por 0:

```
g(xᵢ) = w₀ + xᵢ₁·w₁ + xᵢ₂·w₂
      = w₀ + 0·w₁   + xᵢ₂·w₂
      = w₀ + xᵢ₂·w₂
```

En la práctica, el modelo **ignora** esa feature para ese auto.

> Desde el sentido común, 0 no tiene mucho sentido: no existen autos con
> 0 caballos de fuerza ni motores con 0 cilindros. Pero en machine learning,
> en la práctica, **a veces 0 funciona bien**. Rellenar con la media (como en
> el homework) es posible, solo complica un poco el proceso.

## Predecir sobre train

Por ahora se aplica el modelo al mismo dataset de entrenamiento:

```python
y_pred = w0 + X_train.dot(w)     # sin olvidar el bias
```

## Comparar visualmente

Histograma de predicciones vs. target con Seaborn:

```python
sns.histplot(y_pred, color='red', alpha=0.5, bins=50)
sns.histplot(y_train, color='blue', alpha=0.5, bins=50)
```

- `color` → distinguir las dos distribuciones.
- `bins=50` → menos barras.
- `alpha` → transparencia, para ver ambas superpuestas.

**Lo que se observa:** la distribución de las predicciones está corrida;
incluso los **picos no coinciden**. En muchos casos el modelo predice
**valores más bajos** que los reales.

> Mirar un gráfico sugiere que el modelo no es ideal, pero hace falta una
> forma **objetiva** de decir si un modelo es bueno, y de confirmar que una
> mejora es realmente una mejora. Para eso: **RMSE**, en la siguiente
> lección.

---

**Anterior:** [2.7 — Entrenamiento: ecuación normal](07-ecuacion-normal.md) ·
**Siguiente:** [2.9 — RMSE](09-rmse.md)
