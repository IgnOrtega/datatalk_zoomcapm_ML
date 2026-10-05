# 2.5 — Regresión lineal

La regresión lineal es un modelo para problemas de **regresión**: predice un
**número** (en nuestro caso, el precio de un auto).

Recordando la fórmula de la introducción:

```
g(X) ≈ y
```

- `g` → el modelo (aquí, regresión lineal)
- `X` → matriz de features
- `y` → target (el precio)

Antes de trabajar con la matriz completa, la clase mira el caso simplificado:
**una sola observación** (un auto).

## Una observación = un vector

Un auto es una fila de la matriz de features, o sea un vector de `n` elementos:

```
xᵢ = (xᵢ₁, xᵢ₂, ..., xᵢₙ)
```

Queremos una función `g(xᵢ)` que tome esas características y devuelva algo
**cercano al precio** `yᵢ`.

### Ejemplo de la clase

Se toma la fila 10 del **dataset de entrenamiento** (solo train, nada de
validación ni test): un Rolls-Royce Phantom Drophead Coupe de 2015. Para
simplificar se usan solo 3 features:

| Feature              | Valor |
|----------------------|-------|
| `engine_hp`          | 453   |
| `city_mpg`           | 11    |
| `popularity`         | 86    |

```python
xi = [453, 11, 86]

def g(xi):
    # hacer algo
    return 10000   # por ahora, una predicción inventada
```

## La fórmula

```
g(xᵢ) = w₀ + w₁·xᵢ₁ + w₂·xᵢ₂ + w₃·xᵢ₃
```

- `w₀` → **bias term** (término de sesgo): la predicción que haríamos **sin
  saber nada del auto**.
- `w₁, w₂, w₃` → **pesos** (weights): cada feature se multiplica por su peso.

Escrita de forma compacta con una sumatoria (se usa `j` porque `i` ya indica
el auto):

```
g(xᵢ) = w₀ + Σⱼ wⱼ·xᵢⱼ      (j de 1 a n)
```

## Implementación

En Python los índices van de `0` a `n−1`, no de 1 a n.

```python
w0 = 0
w = [1, 1, 1]

def linear_regression(xi):
    n = len(xi)          # debe coincidir con len(w)

    pred = w0
    for j in range(n):
        pred = pred + w[j] * xi[j]

    return pred
```

El `for` **es** la sumatoria: cada feature se multiplica por su peso y se va
acumulando en la predicción.

Con pesos inventados (de dónde salen se ve más adelante):

```python
w0 = 7.17
w = [0.01, 0.04, 0.002]

linear_regression(xi)    # → 12.31
```

```
7.17 + 453·0.01 + 11·0.04 + 86·0.002 = 12.31
```

## Interpretación de los pesos

| Término      | Peso    | Lectura                                                                 |
|--------------|---------|-------------------------------------------------------------------------|
| bias `w₀`    | 7.17    | Predicción para un "auto promedio" del que no sabemos nada              |
| `engine_hp`  | 0.01    | Más caballos de fuerza → más caro (100 hp suman 1)                      |
| `city_mpg`   | 0.04    | Cada milla por galón extra sube un poco el precio                       |
| `popularity` | 0.002   | Peso muy bajo: hacen falta muchísimas menciones en Twitter para que pese |

> El profesor comenta que el signo de `city_mpg` puede tener sentido según
> cómo se mire (autos que consumen más suelen ser más "fancy"), y que la
> popularidad casi no afecta salvo en casos extremos (¿Tesla?).

## Deshacer el logaritmo

El valor `12.31` **no es el precio**: el target se transformó con
`log(y + 1)`. Para volver a la escala original hay que aplicar la
exponencial y restar 1:

```python
np.exp(12.31) - 1      # → ~222.000 dólares
```

Igual que existe un atajo para el logaritmo (`np.log1p`), existe uno para su
inversa:

```python
np.expm1(12.31)        # equivale a np.exp(x) - 1
```

> `np.log1p` y `np.expm1` se deshacen mutuamente.

Hasta aquí: la fórmula aplicada a **un solo vector** de tamaño 3. En la
siguiente lección se generaliza a muchos ejemplos.

---

**Anterior:** [2.4 — Framework de validación](04-framework-validacion.md) ·
**Siguiente:** [2.6 — Regresión lineal en forma vectorial](06-regresion-lineal-vectorial.md)
