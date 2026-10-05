# 2.10 — RMSE en el conjunto de validación

## El problema

Hasta ahora entrenamos el modelo base (5 variables numéricas) con el conjunto de
**entrenamiento** y calculamos el RMSE **sobre ese mismo conjunto**.

Pero el esquema de validación es otro: el dataset se divide en tres partes
(train / validation / test). Lo correcto es:

1. Entrenar con **train**.
2. Aplicar el modelo a **validation**.
3. Medir el RMSE sobre **validation**.

## La función `prepare_X`

La línea que armaba la matriz de features hacía muchas cosas a la vez. Se separa
en una función que sirve para **cualquier** dataset:

```python
base = ['engine_hp', 'engine_cylinders', 'highway_mpg',
        'city_mpg', 'popularity']

def prepare_X(df):
    df_num = df[base]          # 1. seleccionar las columnas numéricas
    df_num = df_num.fillna(0)  # 2. rellenar los valores faltantes
    X = df_num.values          # 3. extraer la matriz (array de NumPy)
    return X
```

> De una línea pasamos a cinco, pero es más fácil entender qué está pasando.

La idea clave: **preparar los datos exactamente igual** sea train, validation o
test.

## Entrenar y validar

```python
# parte de entrenamiento: solo se toca df_train
X_train = prepare_X(df_train)
w0, w = train_linear_regression(X_train, y_train)

# parte de validación: misma preparación, mismo modelo
X_val = prepare_X(df_val)
y_pred = w0 + X_val.dot(w)

rmse(y_val, y_pred)
```

| Parte | Qué hace |
|---|---|
| Entrenamiento | `prepare_X(df_train)` → entrenar → obtener `w0`, `w` |
| Validación | `prepare_X(df_val)` → aplicar los pesos aprendidos → `rmse(y_val, y_pred)` |

Con esto ya tenemos una forma de evaluar el modelo sobre datos que no vio. El
siguiente paso es **mejorarlo**.

---

**Anterior:** [2.9 — RMSE](09-rmse.md) ·
**Siguiente:** [2.11 — Feature engineering](11-feature-engineering.md)
