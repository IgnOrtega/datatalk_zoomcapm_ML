# 2.11 — Feature engineering

Objetivo: agregar más features al modelo.

## De año a edad

La columna `year` es de las más importantes para predecir el precio: un auto
viejo suele ser más barato, uno nuevo más caro.

En vez de usar el año tal cual, se calcula la **edad** del auto. Para eso hay que
saber cuándo se recolectaron los datos: en **2017**.

```python
df_train['age'] = 2017 - df_train.year
```

Hay autos de 0 años, otros de 9, etc.

## Agregar la feature en `prepare_X`

```python
def prepare_X(df):
    df = df.copy()               # ¡no modificar el DataFrame original!
    df['age'] = 2017 - df.year

    features = base + ['age']    # features base + la nueva

    df_num = df[features]
    df_num = df_num.fillna(0)
    X = df_num.values
    return X
```

### ¿Por qué `df.copy()`?

Sin la copia, la línea `df['age'] = ...` **agrega la columna al DataFrame que le
pasamos** (por ejemplo `df_train`).

> Como usuario de la función, no quiero que cambie mis datos: podría hacer algo
> que después no se puede deshacer. Mejor trabajar sobre una copia.

Con `df.copy()`, la función modifica la copia, y al salir esa copia se descarta:
lo único que importa es la `X` que devuelve. El `X_train` resultante tiene ahora
**6 columnas**, la última es `age`.

## Resultado

```python
X_train = prepare_X(df_train)
w0, w = train_linear_regression(X_train, y_train)

X_val = prepare_X(df_val)
y_pred = w0 + X_val.dot(w)
rmse(y_val, y_pred)
```

| Modelo | RMSE (validación) |
|---|---|
| Base (5 numéricas) | 0.76 |
| Base + `age` | 0.51 |

Una mejora grande. Se confirma graficando predicciones vs. valores reales (ahora
con `y_val`, porque se predice sobre validación): la forma de la distribución se
parece mucho más, aunque todavía hay zonas donde falla.

> Hay mucho margen de mejora. Lo siguiente: **variables categóricas** (marca,
> modelo, etc.).

---

**Anterior:** [2.10 — RMSE en validación](10-rmse-validacion.md) ·
**Siguiente:** [2.12 — Variables categóricas](12-variables-categoricas.md)
