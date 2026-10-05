# 1.8 — Repaso de álgebra lineal

Temas: operaciones con vectores, los tres tipos de multiplicación y la matriz
inversa.

> **Nota de notación:** en álgebra lineal los vectores se escriben como
> **columnas**. En NumPy se ven como filas. Es solo convención de escritura, no
> cambia nada en el código.

## Operaciones simples con vectores

**Multiplicar por un número**: se multiplica cada elemento.

**Sumar dos vectores**: se suman elemento a elemento.

Ambas son exactamente lo que hace NumPy:

```python
u + v      # suma element-wise
2 * v      # multiplicación por escalar
```

## 1. Producto vector-vector (dot product)

También llamado producto interno o escalar. **Ojo: no es la multiplicación
element-wise de NumPy** — el resultado es **un número**, no un array.

Se multiplican los elementos uno a uno y después **se suman los resultados**:

$$u^T v = \sum_{i=1}^{n} u_i v_i$$

Ejemplo: `u = [2, 4, 5, 6]`, `v = [1, 0, 0, 2]`
→ `2·1 + 4·0 + 5·0 + 6·2 = 2 + 12 = 14`

### La notación uᵀv

La **transpuesta** (`ᵀ`) convierte columnas en filas. Como para multiplicar
necesitamos un vector fila por un vector columna, el producto punto se escribe
`uᵀv`. Es solo notación.

### Implementación

```python
def vector_vector_multiplication(u, v):
    assert u.shape[0] == v.shape[0]      # deben tener el mismo tamaño

    n = u.shape[0]
    result = 0.0
    for i in range(n):
        result = result + u[i] * v[i]
    return result
```

El `for` **es** la sumatoria de la fórmula; el cuerpo es lo que va adentro del
símbolo Σ. La fórmula va de 1 a n, pero los arrays de NumPy se indexan de 0 a n−1.

En NumPy ya existe:

```python
u.dot(v)     # → 14
```

## 2. Producto matriz-vector

Para cada **fila** de la matriz `U` se hace un producto punto con el vector `v`.

```
U (k filas × n columnas)  ×  v (n elementos)  =  vector de k elementos
```

La dimensionalidad tiene que coincidir: el **número de columnas de U** debe ser
igual al **número de elementos de v**.

```python
def matrix_vector_multiplication(U, v):
    assert U.shape[1] == v.shape[0]

    num_rows = U.shape[0]
    result = np.zeros(num_rows)
    for i in range(num_rows):
        result[i] = vector_vector_multiplication(U[i], v)
    return result
```

En NumPy: `U.dot(v)`. NumPy se da cuenta solo de que, al invocarlo sobre un array
2D, tiene que hacer producto matriz-vector.

## 3. Producto matriz-matriz

Se descompone la matriz `V` en sus **columnas** y se hace un producto
matriz-vector con cada una. Cada resultado es una columna del resultado final.

```
U × V  →  columna i del resultado = U · (columna i de V)
```

Dimensiones del resultado: **filas de U × columnas de V**.

```python
def matrix_matrix_multiplication(U, V):
    assert U.shape[1] == V.shape[0]

    num_rows = U.shape[0]
    num_cols = V.shape[1]
    result = np.zeros((num_rows, num_cols))

    for i in range(num_cols):
        vi = V[:, i]                                     # columna i de V
        Uvi = matrix_vector_multiplication(U, vi)
        result[:, i] = Uvi                               # se asigna como columna
    return result
```

En NumPy: `U.dot(V)`.

> **La idea que unifica la clase:**
> matriz-matriz se expresa como varias multiplicaciones matriz-vector, y
> matriz-vector se expresa como varias multiplicaciones vector-vector.
> Escritas en código, las fórmulas dejan de dar miedo.

## Matriz identidad

Matriz **cuadrada** con unos en la diagonal y ceros en todo el resto.

```python
I = np.eye(3)
```

Es el «número 1» de las matrices: `U · I = U` e `I · U = U`, sin importar de qué
lado se multiplique (con las dimensiones correspondientes).

## Matriz inversa

La inversa de `A`, escrita `A⁻¹`, es la matriz tal que:

$$A^{-1} A = I$$

**Solo las matrices cuadradas tienen inversa** (mismo número de filas que de
columnas).

```python
Vs = V[[0, 1, 2]]          # tomar un bloque cuadrado
Vs_inv = np.linalg.inv(Vs)  # linalg = linear algebra

Vs_inv.dot(Vs)              # → matriz identidad
```

> **Por qué importa:** la inversa es la pieza clave de la **regresión lineal**,
> que se ve en la sesión siguiente.

---

**Anterior:** [1.7 — NumPy](07-numpy.md) ·
**Siguiente:** [1.9 — Pandas](09-pandas.md)
