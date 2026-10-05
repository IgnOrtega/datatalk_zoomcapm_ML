# 2.9 — RMSE (Root Mean Squared Error)

En la lección anterior entrenamos el modelo base (solo variables numéricas) y
graficamos predicciones vs. valores reales: se veía que estaban algo
desfasados, pero **no teníamos un número** para medir qué tan malo es el modelo.
El RMSE es una forma de evaluar modelos de regresión.

## La fórmula

```
RMSE = √( (1/m) · Σ (g(xᵢ) − yᵢ)² )      con i = 1 … m
```

| Parte | Qué es |
|---|---|
| `g(xᵢ)` | la predicción para la observación i |
| `yᵢ` | el valor real (el precio) de la observación i |
| `(g(xᵢ) − yᵢ)²` | el **error al cuadrado** (squared error) |
| `(1/m) · Σ …` | el promedio sobre las m observaciones → **MSE** |
| `√ …` | la raíz cuadrada → **RMSE** |

## Ejemplo paso a paso

| | Obs. 1 | Obs. 2 | Obs. 3 | Obs. 4 |
|---|---|---|---|---|
| Predicción (`y_pred`) | 10 | 9 | 11 | 10 |
| Real (`y_train`) | 9 | 9 | 10.5 | 11.5 |
| Diferencia | 1 | 0 | 0.5 | −1.5 |
| Diferencia² | 1 | 0 | 0.25 | 2.25 |

```
MSE  = (1 + 0 + 0.25 + 2.25) / 4 = 0.875
RMSE = √0.875 ≈ 0.93
```

## Implementación

```python
def rmse(y, y_pred):
    error = y - y_pred
    se = error ** 2      # squared error
    mse = se.mean()      # mean squared error
    return np.sqrt(mse)  # root mean squared error
```

No hace falta sumar y dividir por la cantidad de elementos: en NumPy alcanza con
el método `.mean()`.

Versión simplificada (sin la variable intermedia `error`):

```python
def rmse(y, y_pred):
    se = (y - y_pred) ** 2
    mse = se.mean()
    return np.sqrt(mse)
```

Se usa con las predicciones que ya teníamos:

```python
rmse(y_train, y_pred)
```

> Cuanto **más bajo** el RMSE, mejor el modelo. En la próxima lección se aplica
> sobre el conjunto de validación.

---

**Anterior:** [2.8 — Modelo base](08-modelo-base.md) ·
**Siguiente:** [2.10 — RMSE en validación](10-rmse-validacion.md)
