# 2.13 — Regularización

Al agregar las variables categóricas (2.12) el RMSE se disparó y los pesos
salieron enormes. En esta clase se ve **por qué pasa** y cómo arreglarlo.

## El problema: la inversa de XᵀX

Recordemos la ecuación normal:

```
w = (XᵀX)⁻¹ Xᵀ y
```

La parte delicada es invertir la **matriz de Gram** `XᵀX`. **Esa inversa no
siempre existe.** Ocurre típicamente cuando la matriz de features tiene
**columnas duplicadas**: en álgebra lineal se dice que una columna es
**combinación lineal** de las otras.

```python
X = [
    [4, 4, 4],
    [3, 5, 5],
    [5, 1, 1],
    [5, 4, 4],
    [7, 5, 5],
    [4, 5, 5],
]
X = np.array(X)

XTX = X.T.dot(X)
np.linalg.inv(XTX)     # LinAlgError: Singular matrix
```

La segunda y la tercera columna son iguales → la matriz es **singular** y NumPy
se niega a invertirla.

## Por qué en nuestro dataset no dio error

Los datos reales tienen **ruido**. Basta con que un valor sea `5.0000001` en vez
de `5` para que las columnas ya no sean *exactamente* iguales:

```python
X = [
    [4, 4, 4],
    [3, 5, 5],
    [5, 1, 1],
    [5, 4, 4],
    [7, 5, 5],
    [4, 5, 5.0000001],
]
```

Ahora la matriz es **numéricamente invertible**: NumPy encuentra una «inversa»
aunque en rigor no debería existir, y lo hace con **números gigantes**. Al
calcular `w`, la feature única sale bien, pero las dos columnas casi duplicadas
reciben pesos enormes. Eso es lo que vimos en 2.12.

## La solución: sumar un número a la diagonal

Si se suma un número pequeño a la **diagonal** de `XᵀX`, la columna 3 deja de
ser (casi) un duplicado de la columna 2 y la inversa se vuelve estable:

```python
XTX = [
    [1, 2, 2],
    [2, 1, 1.0000001],
    [2, 1.0000001, 1],
]
XTX = XTX + 0.01 * np.eye(3)   # suma 0.01 solo en la diagonal

np.linalg.inv(XTX)              # números mucho más pequeños
```

El truco: `np.eye(3)` es la identidad (unos en la diagonal). Sumarla agregaría
`1` a la diagonal; multiplicándola por un número pequeño se agrega solo ese
número.

> **Regularización** = *controlar* los pesos para que no crezcan demasiado.
> Cuanto más grande el número sumado a la diagonal, más pequeños los valores de
> la inversa y más «bajo control» quedan los pesos.

## Implementación

Se copia `train_linear_regression` y se le agrega un parámetro `r` (de
*regularization*):

```python
def train_linear_regression_reg(X, y, r=0.001):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX = XTX + r * np.eye(XTX.shape[0])

    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)

    return w_full[0], w_full[1:]
```

Y se reemplaza en el bloque de entrenamiento/validación:

```python
X_train = prepare_X(df_train)
w0, w = train_linear_regression_reg(X_train, y_train, r=0.01)

X_val = prepare_X(df_val)
y_pred = w0 + X_val.dot(w)
rmse(y_val, y_pred)
```

El resultado no solo es muchísimo mejor que la versión sin regularizar: también
mejora en ~0.5 respecto del mejor modelo anterior (el de 2.11), una mejora
considerable.

## `r` es un parámetro

| Valor de `r` | Efecto |
|---|---|
| `0` | Vuelve a la regresión lineal normal (sin regularización) |
| Pequeño | Controla los pesos y estabiliza la inversa |
| Demasiado grande | Al modelo le cuesta aprender → empeora |

> Hay que encontrar el mejor valor de `r`: eso se hace en la próxima clase.

---

**Anterior:** [2.12 — Variables categóricas](12-variables-categoricas.md) ·
**Siguiente:** [2.14 — Ajuste del modelo](14-ajuste-modelo.md)
