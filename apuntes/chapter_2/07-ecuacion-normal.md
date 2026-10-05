# 2.7 — Entrenamiento: ecuación normal

Ya sabemos **aplicar** el modelo (`X·w`). Ahora: ¿cómo se **obtienen** los
pesos `w`?

## El problema

Queremos que las predicciones se parezcan al target:

```
g(X) = X·w ≈ y
```

Idealmente `X·w = y`: un sistema de ecuaciones a resolver para `w`.

### Si X tuviera inversa

```
X⁻¹·X·w = X⁻¹·y
    I·w = X⁻¹·y
      w = X⁻¹·y
```

Pero `X` es **rectangular** (`m × (n+1)`: muchas filas, pocas columnas), así
que **no tiene inversa**. El sistema no tiene solución exacta.

## La solución aproximada

Se multiplican ambos lados por `Xᵀ`:

```
XᵀX·w = Xᵀy
```

`XᵀX` es la **matriz de Gram**. Siempre es **cuadrada**, de
`(n+1) × (n+1)`, así que su inversa **puede** existir (normalmente existe; no
siempre — se retoma más adelante).

Multiplicando por `(XᵀX)⁻¹`:

```
(XᵀX)⁻¹·XᵀX·w = (XᵀX)⁻¹·Xᵀy
            I·w = (XᵀX)⁻¹·Xᵀy

              w = (XᵀX)⁻¹·Xᵀ·y        ← ecuación normal
```

> Este `w` **no es** la solución del sistema (no existe), pero es la **más
> cercana posible**. La demostración requiere bastante matemática; el
> profesor recomienda el libro *The Elements of Statistical Learning* para
> quien quiera ver la derivación. Lo de la clase es a nivel intuitivo.

## Paso a paso en NumPy

Se inventa una matriz `X` con **más filas que columnas** (con pocas filas la
inversa probablemente no existiría) y un `y` cualquiera.

```python
XTX = X.T.dot(X)                 # matriz de Gram
XTX_inv = np.linalg.inv(XTX)     # su inversa

XTX.dot(XTX_inv)                 # ≈ matriz identidad
```

> La comprobación no da exactamente la identidad: aparecen números
> diminutos fuera de la diagonal. Es por la **precisión finita** de la
> máquina; se pueden tratar como cero.

```python
w = XTX_inv.dot(X.T).dot(y)
```

## No olvidar el bias

Si `X` no tiene la columna de unos, el modelo se entrena **sin bias term**.
A veces eso tiene sentido, pero queremos el bias porque es la **línea base**:
cuánto costaría un auto del que no sabemos nada.

Para agregar la columna de unos:

```python
ones = np.ones(X.shape[0])       # un 1 por cada fila
X = np.column_stack([ones, X])   # apila como columnas: los unos quedan primero
```

Luego se repite la ecuación normal y se separa el resultado:

```python
w_full = XTX_inv.dot(X.T).dot(y)

w0 = w_full[0]       # bias
w  = w_full[1:]      # resto de los pesos
```

### Pesos negativos

En el ejemplo algunos pesos salen **negativos**: significa que la feature
**resta** al precio en vez de sumar. Ejemplo de la clase: si la feature fuera
la **edad** del auto, cada año extra bajaría el precio.

## La función final

```python
def train_linear_regression(X, y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)

    return w_full[0], w_full[1:]
```

> Los unos se agregan **dentro** de la función: como usuarios no hay que
> preocuparse de añadirlos a la matriz de features.

```python
w0, w = train_linear_regression(X, y)
```

---

**Anterior:** [2.6 — Regresión lineal en forma vectorial](06-regresion-lineal-vectorial.md) ·
**Siguiente:** [2.8 — Modelo base](08-modelo-base.md)
