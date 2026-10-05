# 2.14 — Ajuste del modelo

Vimos que `r` afecta la calidad del modelo. Ahora se busca **el mejor valor** de
`r` usando el **dataset de validación** (*tuning* de un parámetro).

## Probar varios valores de r

Se recorre una lista de valores, desde `0` y aumentando gradualmente, y para
cada uno se entrena, se evalúa en validación y se imprime el resultado:

```python
for r in [0.0, 0.00001, 0.0001, 0.001, 0.1, 1, 10]:
    X_train = prepare_X(df_train)
    w0, w = train_linear_regression_reg(X_train, y_train, r=r)

    X_val = prepare_X(df_val)
    y_pred = w0 + X_val.dot(w)
    score = rmse(y_val, y_pred)

    print(r, w0, score)
```

Se imprime el parámetro, el **bias** (`w0`) y el **score** (RMSE).

## Qué se observa

| `r` | Bias `w0` | RMSE |
|---|---|---|
| `0` | Enorme | Enorme |
| Valores pequeños | Razonable | Mejora mucho y casi no cambia entre ellos |
| Valores grandes (`1`, `10`) | Cada vez más chico | Empeora |

- Con un poco de regularización ya mejora muchísimo.
- Cuanta más regularización, **más pequeño el bias**.
- Al subir demasiado `r` el modelo empieza a degradarse.

> Se elige un valor pequeño en el que el modelo **todavía no empieza a
> degradarse**. Entre los valores pequeños la diferencia es mínima, así que da
> lo mismo cuál de ellos se elija. En el notebook del curso se usa `r = 0.001`.

## Entrenar con el r elegido

```python
r = 0.001
X_train = prepare_X(df_train)
w0, w = train_linear_regression_reg(X_train, y_train, r=r)

X_val = prepare_X(df_val)
y_pred = w0 + X_val.dot(w)
rmse(y_val, y_pred)
```

Funciona bien en validación. Falta comprobarlo en el **dataset de test**, que es
lo que se hace en la próxima clase.

---

**Anterior:** [2.13 — Regularización](13-regularizacion.md) ·
**Siguiente:** [2.15 — Usar el modelo](15-usar-el-modelo.md)
