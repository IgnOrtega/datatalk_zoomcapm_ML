# 2.15 — Usar el modelo

Dos pasos: entrenar el **modelo final** y **usarlo** para predecir el precio de
un auto.

## Entrenar el modelo final

Hasta ahora se entrenaba en *train* y se evaluaba en *validation*. Para el
modelo final:

1. Se entrena con **train + validation** juntos («full train»).
2. Se hace la evaluación final en **test**.

> El RMSE en test **no debería ser muy distinto** del de validación.

### Juntar los DataFrames

`pd.concat` recibe una lista de DataFrames y los concatena:

```python
df_full_train = pd.concat([df_train, df_val])
df_full_train = df_full_train.reset_index(drop=True)
```

Sin `reset_index` el índice queda con los valores originales de validación; al
resetearlo queda secuencial.

### Feature matrix y target

```python
X_full_train = prepare_X(df_full_train)

y_full_train = np.concatenate([y_train, y_val])
```

`np.concatenate` hace lo mismo para arrays de NumPy (que no tienen índice, así
que no hay nada que resetear).

### Entrenar y evaluar en test

```python
w0, w = train_linear_regression_reg(X_full_train, y_full_train, r=0.001)

X_test = prepare_X(df_test)
y_pred = w0 + X_test.dot(w)
score = rmse(y_test, y_pred)
score
```

El RMSE en test es prácticamente igual al de validación (hasta el tercer
decimal).

> Muy buena señal: el modelo **generaliza bien** y no obtuvo ese score por pura
> casualidad.

## Usar el modelo para un auto nuevo

La idea: tenemos un auto, extraemos sus features (el *feature vector*), las
pasamos al modelo y este predice el precio.

Se toma un auto del test (un Toyota Sienna, con algunos valores faltantes) y se
**finge que es nuevo** — es válido porque el modelo nunca lo vio en el
entrenamiento.

### El auto como diccionario

En la vida real no llega un DataFrame sino algo como un **diccionario**: por
ejemplo, una web o app donde el usuario ingresa los datos del auto, que envía un
*request* al modelo y recibe el precio de vuelta.

```python
car = df_test.iloc[20].to_dict()
car
```

### Pasarlo por prepare_X

`prepare_X` espera un **DataFrame**, así que se crea uno con **una sola fila** a
partir de una lista con un solo diccionario:

```python
df_small = pd.DataFrame([car])
X_small = prepare_X(df_small)
```

### Predecir

```python
y_pred = w0 + X_small.dot(w)
y_pred = y_pred[0]        # un solo auto → un solo número
y_pred                    # ≈ 10.6
```

El resultado está en **escala logarítmica**. Para volver al precio se deshace
el `log1p` con `np.expm1`:

```python
np.expm1(y_pred)          # precio predicho

np.expm1(y_test[20])      # precio real
```

La predicción quedó a unos **5.000 dólares** del precio real: una predicción
relativamente buena.

## Resumen del flujo

```
df_train + df_val ──concat──▶ df_full_train ──prepare_X──▶ X_full_train
                                                    │
                          train_linear_regression_reg(r=0.001)
                                                    │
                                                 w0, w
                                                    │
auto (dict) ──pd.DataFrame([car])──▶ prepare_X ──▶ ŷ = w0 + X·w ──np.expm1──▶ precio
```

---

**Anterior:** [2.14 — Ajuste del modelo](14-ajuste-modelo.md) ·
**Siguiente:** [2.16 — Resumen](16-resumen.md)
