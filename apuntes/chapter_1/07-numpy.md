# 1.7 — Introducción a NumPy

```python
import numpy as np   # el alias np es la convención en data science
```

## Crear arrays

```python
np.zeros(10)          # array de 10 ceros
np.ones(10)           # array de 10 unos
np.full(10, 2.5)      # array de 10 elementos, todos 2.5

a = np.array([1, 2, 3, 5, 7, 12])   # desde una lista de Python
```

### Rangos y secuencias

```python
np.arange(10)          # [0, 1, ..., 9]  — el límite superior es EXCLUSIVO
np.arange(3, 10)       # [3, 4, ..., 9]

np.linspace(0, 1, 11)  # 11 números repartidos entre 0 y 1 (inclusive ambos)
```

`arange` es el equivalente del `range` de Python, pero devuelve un array de NumPy
en vez de un iterador.

> Truco de la clase: con `linspace(0, 1, 11)` los pasos quedan redondos (0.0, 0.1,
> 0.2…); con 10 no.

## Acceder y modificar elementos

```python
a[2]        # tercer elemento (los índices empiezan en 0)
a[2] = 10   # reemplazar
```

## Arrays multidimensionales

```python
np.zeros((5, 2))    # 5 filas, 2 columnas

n = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
])
```

El primer número de la tupla son **filas**, el segundo **columnas**.

```python
n[0, 1]              # fila 0, columna 1  → 2
n[0, 1] = 20         # reemplazar un elemento

n[0]                 # fila completa
n[2] = [1, 1, 1]     # reemplazar una fila entera (la dimensión debe coincidir)

n[:, 1]              # columna 1 completa — los ":" significan "todas las filas"
n[:, 2] = [0, 1, 2]  # reemplazar una columna
```

Para acceder a una columna **no se puede dejar el primer índice vacío**: hay que
poner `:`.

## Números aleatorios

```python
np.random.rand(5, 2)        # uniforme estándar, entre 0 y 1
np.random.randn(5, 2)       # distribución normal estándar
np.random.randint(0, 100, size=(5, 2))   # enteros; el límite superior es exclusivo
```

### Semilla (seed) — reproducibilidad

```python
np.random.seed(2)
np.random.rand(5, 2)   # siempre da lo mismo
```

Los números no son realmente aleatorios, son **pseudoaleatorios**: se generan a
partir de una semilla. Fijándola, la secuencia es la misma en tu máquina y en la
mía.

> Puede variar levemente según la versión de NumPy o el sistema operativo, pero
> en general coincide.

## Operaciones element-wise

La gran ventaja sobre las listas de Python: **no hace falta escribir un for**.

```python
a = np.arange(5)     # [0, 1, 2, 3, 4]

a + 1                # suma 1 a cada elemento
a * 2                # multiplica cada elemento por 2
a / 100
a ** 2

b = (a / 2) * 10 + 10      # se pueden encadenar
```

Así es como se generan enteros entre 0 y 100: `np.random.rand(5, 2) * 100`.

### Entre dos arrays

```python
a + b      # suma elemento a elemento
a * b
a / b
```

### Comparaciones (también element-wise)

```python
a >= 2         # → array de booleanos [False, False, True, True, True]
a > b          # comparar dos arrays elemento a elemento
```

### Filtrado con máscara booleana

```python
a[a > b]       # devuelve solo los elementos donde la condición es True
```

Por dentro: `a > b` produce un array de booleanos, y usarlo como índice devuelve
los elementos que cumplen la condición. Esto reaparece en Pandas.

## Operaciones de resumen (summarizing)

A diferencia de las anteriores, devuelven **un solo número** en vez de un array:

```python
a.min()     # mínimo
a.max()     # máximo
a.sum()     # suma
a.mean()    # promedio
a.std()     # desviación estándar
```

Funcionan igual sobre arrays bidimensionales (`n.sum()`, `n.min()`, …).

> La clase menciona que hay mucho más: por ejemplo obtener el mínimo de **cada
> fila** de un array 2D, u ordenar. Vale mirar la documentación.

---

**Anterior:** [1.6 — Entorno](06-entorno-codespaces.md) ·
**Siguiente:** [1.8 — Álgebra lineal](08-algebra-lineal.md)
