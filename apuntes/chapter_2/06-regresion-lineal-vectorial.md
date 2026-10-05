# 2.6 — Regresión lineal en forma vectorial

Objetivo: pasar de la fórmula para **un auto** a la forma con la **matriz
completa** `X`.

## La sumatoria es un producto punto

La fórmula para un auto:

```
g(xᵢ) = w₀ + Σⱼ wⱼ·xᵢⱼ
```

La sumatoria no es otra cosa que un **producto punto** (vector-vector) entre
el vector de features y el vector de pesos:

```
g(xᵢ) = w₀ + xᵢᵀw
```

### Implementación

Se reutiliza la multiplicación vector-vector del módulo 1:

```python
def dot(xi, w):
    n = len(xi)

    res = 0.0
    for j in range(n):
        res = res + xi[j] * w[j]

    return res

def linear_regression(xi):
    return w0 + dot(xi, w)
```

> Con NumPy no hace falta escribir `dot`: basta con `xi.dot(w)`.

## Meter el bias dentro del producto punto

El `w₀` queda "colgando" solo, sin ningún `x`. Truco: imaginar una **feature
ficticia** `xᵢ₀` que **siempre vale 1**.

```
w  = (w₀, w₁, w₂, ..., wₙ)        → n+1 elementos
xᵢ = (1, xᵢ₁, xᵢ₂, ..., xᵢₙ)        → n+1 elementos
```

Al hacer el producto punto, `w₀ · 1 = w₀`, y el resto es igual que antes:

```
g(xᵢ) = xᵢᵀw      (con el 1 al inicio de xᵢ y w₀ al inicio de w)
```

```python
w_new = [w0] + w          # con listas, + concatena: antepone w0

def linear_regression(xi):
    xi = [1] + xi         # antepone el 1
    return dot(xi, w_new)
```

El resultado es el mismo que con la versión anterior.

## Generalizar a todos los autos

La matriz `X` ahora tiene una **columna de unos** al principio:

```
      ┌ 1  x₁₁  x₁₂  ...  x₁ₙ ┐          ┌ w₀ ┐
      │ 1  x₂₁  x₂₂  ...  x₂ₙ │          │ w₁ │
X  =  │ ...                   │     w =  │ ...│
      └ 1  xₘ₁  xₘ₂  ...  xₘₙ ┘          └ wₙ ┘
```

- `X` es de `m × (n+1)`: **m filas** (autos) y **n+1 columnas**.
- Para cada fila se hace el producto punto con `w` → una predicción por auto.

```
ŷ = (x₁ᵀw, x₂ᵀw, ..., xₘᵀw)  =  X·w
```

> Esto es exactamente una **multiplicación matriz-vector**. Aplicar la
> regresión lineal = multiplicar `X` por `w`.

### Implementación

```python
# valores de ejemplo inventados en clase (con el 1 al inicio)
x1  = [1, 148, 24, 1385]
x2  = [1, 132, 25, 2031]
x10 = [1, 453, 11, 86]

X = [x1, x2, x10]       # lista de listas
X = np.array(X)         # → matriz (array 2D de NumPy)

def linear_regression(X):
    return X.dot(w_new)

linear_regression(X)    # una predicción por cada auto
```

## Resumen del recorrido

| Paso | Forma |
|------|-------|
| 1 | `for` sobre las features (sumatoria) |
| 2 | Producto punto `w₀ + xᵢᵀw` |
| 3 | Feature ficticia = 1 → `xᵢᵀw` |
| 4 | Todos los autos a la vez → `X·w` |

Queda la pregunta: **¿de dónde salen los pesos `w`?** → siguiente lección.

---

**Anterior:** [2.5 — Regresión lineal](05-regresion-lineal.md) ·
**Siguiente:** [2.7 — Entrenamiento: ecuación normal](07-ecuacion-normal.md)
