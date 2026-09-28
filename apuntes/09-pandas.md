# 1.9 — Introducción a Pandas

Pandas es la librería de Python para manipular **datos tabulares**. Por debajo usa
NumPy.

```python
import pandas as pd
```

La abstracción principal es el **DataFrame** (una tabla). Cada columna de un
DataFrame es una **Series**.

El dataset de la clase es un subconjunto del de precios de autos que se usa en la
sesión 2: marca, modelo, año, motor, caballos de fuerza, cilindros, tipo de
transmisión, estilo de vehículo y precio (MSRP).

## Crear un DataFrame

### Desde una lista de listas

```python
data = [
    ["Nissan", "Stanza", 1991, 138.0, 4, "MANUAL", "sedan", 2000],
    # ...
]
columns = ["Make", "Model", "Year", "Engine HP", "Engine Cylinders",
           "Transmission Type", "Vehicle_Style", "MSRP"]

df = pd.DataFrame(data, columns=columns)
```

Sin el parámetro `columns`, Pandas nombra las columnas `0, 1, 2, 3…` porque no
sabe qué significan.

### Desde una lista de diccionarios

```python
data = [
    {"Make": "Nissan", "Model": "Stanza", "Year": 1991, ...},
    # ...
]
df = pd.DataFrame(data)
```

Acá no hace falta pasar `columns`: Pandas usa las **claves** de los diccionarios
como nombres de columna.

## Lo primero al abrir un dataset

```python
df.head()      # primeras 5 filas
df.head(2)     # primeras 2
```

> Es lo primero que hace el instructor después de cargar datos desde un CSV o una
> consulta SQL: mirar las primeras filas.

## Acceder a columnas (Series)

```python
df.Make                 # notación de punto
df["Make"]              # notación de corchetes — equivalente
```

La notación de punto **no funciona** si el nombre tiene espacios o guiones
(`"Engine HP"`). Para esos casos hay que usar corchetes.

### Varias columnas a la vez

```python
df[["Make", "Model", "MSRP"]]    # doble corchete: una lista adentro de los []
```

Devuelve un DataFrame nuevo con solo esas columnas.

### Agregar, reemplazar y borrar columnas

```python
df["id"] = [1, 2, 3, 4, 5]     # agregar (o reemplazar si ya existe)
del df["id"]                    # borrar — igual que con un diccionario
```

## El índice

Los números al costado de las filas son el **índice**.

```python
df.index        # RangeIndex de 0 a 5 (el límite superior es exclusivo)
df.Make.index   # todas las Series comparten el índice del DataFrame
```

### loc vs iloc

| | Qué usa | Ejemplo |
|---|---|---|
| `df.loc[1]` | El índice **por etiqueta** | `df.loc[[1, 2]]` |
| `df.iloc[1]` | El índice **posicional** (0 a n−1) | `df.iloc[[1, 2]]` |

Si se reemplaza el índice por otra cosa:

```python
df.index = ["a", "b", "c", "d", "e"]

df.loc[1]        # ERROR: no existe ese índice
df.loc[["b", "c"]]   # así sí
df.iloc[[1, 2]]      # el posicional sigue funcionando siempre
```

### Volver al índice normal

```python
df.reset_index()                # guarda el índice viejo en una columna "index"
df.reset_index(drop=True)       # lo descarta

df = df.reset_index(drop=True)  # hay que reasignar: NO modifica el DataFrame original
```

> **Patrón general de Pandas:** casi todas las operaciones **devuelven un objeto
> nuevo** en vez de modificar el original. Si querés conservar el cambio, hay que
> reasignar.

## Operaciones element-wise

Igual que en NumPy, pero sobre Series:

```python
df["Engine HP"] / 100
df["Engine HP"] * 2
```

`NaN` indica un **valor faltante**; las operaciones lo dejan como está.

La diferencia con NumPy es que una Series tiene índice y nombre; por dentro es lo
mismo.

### Comparaciones y filtrado

```python
df["Year"] >= 2015          # devuelve una Series booleana (True/False)

df[df["Year"] >= 2015]      # FILTRA: devuelve solo las filas donde es True
```

Son dos partes: la condición de adentro produce la máscara booleana, y los
corchetes de afuera se quedan con las filas verdaderas.

```python
df[df["Make"] == "Nissan"]

# combinar condiciones con & (and) y | (or)
df[(df["Make"] == "Nissan") & (df["Year"] >= 2015)]
```

## Operaciones con strings

Algo que NumPy no tiene: NumPy es sobre todo para números, en Pandas hay strings
todo el tiempo.

El accesor `.str` aplica un método de string a **todos** los elementos de la
Series:

```python
df["Vehicle_Style"].str.lower()                 # a minúsculas
df["Vehicle_Style"].str.replace(" ", "_")       # espacios → guiones bajos

# encadenables
df["Vehicle_Style"].str.replace(" ", "_").str.lower()

# recordar reasignar para que el cambio quede
df["Vehicle_Style"] = df["Vehicle_Style"].str.replace(" ", "_").str.lower()
```

Es un paso de preprocesamiento típico: normalizar mayúsculas e inconsistencias de
espaciado.

## Operaciones de resumen

```python
df["MSRP"].mean()      # también min(), max(), sum(), std()

df["MSRP"].describe()  # count, mean, std, min, percentiles (25/50/75), max
df.describe()          # lo mismo para TODAS las columnas numéricas
df.describe().round(2) # más compacto de leer
```

`describe()` sobre el DataFrame ignora automáticamente las columnas de texto: no
tiene sentido el promedio de «Make».

### Variables categóricas

```python
df["Make"].nunique()   # cuántos valores únicos hay
df.nunique()           # valores únicos por cada columna
```

Útil para ver de un vistazo, por ejemplo, que solo hay dos tipos de transmisión.

## Valores faltantes

Importan mucho: para machine learning **no queremos NaN**.

```python
df.isnull()          # DataFrame de booleanos: True donde falta el valor
df.isnull().sum()    # ← lo realmente útil: cuántos faltantes por columna
```

La suma se aplica por columna, así que da la cuenta de faltantes de cada una.

## Group by

El equivalente de este SQL:

```sql
SELECT transmission_type, AVG(MSRP)
FROM cars
GROUP BY transmission_type;
```

es:

```python
df.groupby("Transmission Type")["MSRP"].mean()
df.groupby("Transmission Type")["MSRP"].min()
df.groupby("Transmission Type")["MSRP"].max()
```

Primero se agrupa por una columna, después se calcula el agregado dentro de cada
grupo.

## Conversiones

```python
df["MSRP"].values              # el array de NumPy que está por debajo
df.to_dict(orient="records")   # de vuelta a lista de diccionarios
```

`to_dict(orient="records")` es útil para guardar a un archivo o pasarle los datos
a otro sistema.

---

**Anterior:** [1.8 — Álgebra lineal](08-algebra-lineal.md) ·
**Siguiente:** [1.10 — Resumen](10-resumen.md)
