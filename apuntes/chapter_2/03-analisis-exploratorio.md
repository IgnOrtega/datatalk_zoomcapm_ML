# 2.3 — Análisis exploratorio

Objetivo del **EDA** (*exploratory data analysis*): entender cómo se ven los
datos, qué valores hay, y aprender más sobre los datos y el problema.

## Recorrer las columnas

Para cada columna, mostrar algunos valores, los primeros valores únicos y cuántos
valores únicos hay:

```python
for col in df.columns:
    print(col)
    print(df[col].unique()[:5])   # primeros 5 valores únicos
    print(df[col].nunique())      # cantidad de valores únicos
    print()
```

> Con `head()` se ven solo las primeras filas (todas BMW), poco informativo. Más
> útil es mirar los **valores únicos**.

| Columna | Qué es | Observación |
|---------|--------|-------------|
| `make` | fabricante | 48 valores (bmw, audi, fiat, mercedes-benz, chrysler…) |
| `model` | modelo | más granular: muchos más valores |
| `year` | año | numérica, 28 valores |
| `engine_fuel_type` | tipo de combustible | 10 tipos |
| `engine_hp` | potencia del motor | |
| `engine_cylinders` | cantidad de cilindros | |
| `transmission_type`, `driven_wheels`, … | otras características | |
| `highway_mpg` | millas por galón en carretera | |
| `city_mpg` | millas por galón en ciudad | |
| `popularity` | menciones en Twitter | los autores del dataset las contaron |
| `msrp` | **precio** | el target; muchísimos valores distintos |

## Distribución del precio

Para ver el panorama general hay que **graficar**; mirar números sueltos no
alcanza.

```python
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline      # para que los gráficos se vean en el notebook

sns.histplot(df.msrp, bins=50)
```

- **Matplotlib**: librería de gráficos de bajo nivel.
- **Seaborn**: construida sobre Matplotlib, hace las cosas más fáciles.
- `bins` = cantidad de barras del histograma.

> En el eje, `1e6` es notación científica: 10⁶ = un millón.

### Long tail

Casi todos los autos son relativamente baratos y hay muy pocos carísimos (1, 1.5,
2 millones). Esa forma se llama **distribución de cola larga** (*long tail*).

Para ver mejor la zona donde están la mayoría, se hace zoom:

```python
sns.histplot(df.msrp[df.msrp < 100000], bins=50)
```

Observaciones:

- Hay un **pico raro en 1000**: probablemente el precio mínimo que permite la
  plataforma.
- Hay muchos autos (unos 700) alrededor de 25 000, y después la cantidad va
  bajando.

> Las colas largas son muy comunes en precios: la mayoría de las cosas son
> accesibles para el público general y unas pocas son muy caras, porque poca gente
> puede pagarlas.

## Por qué la cola larga es un problema

Una distribución así **no es buena para ML**: la cola confunde al modelo. Hay que
eliminar ese efecto, y lo habitual es aplicar el **logaritmo** al precio: los
valores grandes quedan mucho más comprimidos.

```python
np.log([1, 10, 1000, 100000])   # crece mucho más despacio que los valores
```

### `log1p`: logaritmo de (x + 1)

`log(0)` no existe. En este dataset no hay precios 0 (el mínimo es 1000), pero es
práctica común **sumar 1** antes de aplicar el logaritmo: así `log(0 + 1) = 0` y no
hay problemas.

NumPy tiene un atajo que hace las dos cosas (`1p` = *plus one*):

```python
np.log1p([0, 1, 10, 1000, 100000])
# es lo mismo que
np.log([0 + 1, 1 + 1, 10 + 1, 1000 + 1, 100000 + 1])
```

Aplicado al precio:

```python
price_logs = np.log1p(df.msrp)

sns.histplot(price_logs, bins=50)
```

Resultado: **la cola desaparece**. Los precios muy altos colapsan hacia una zona
compacta y la forma se parece a una **campana**: la **distribución normal**.

> Si el target se parece a una normal (un centro claro y que baja hacia ambos
> lados), los modelos predicen mucho mejor que con una cola larga. El pico en 1000
> sigue ahí, pero igual se parece bastante más a una normal.

## Valores faltantes

En `number_of_doors`, por ejemplo, aparece `NaN` (*not a number*): en pandas
indica que el valor **falta**, no se registró.

```python
df.isnull()         # True/False celda por celda (poco útil así)
df.isnull().sum()   # cantidad de valores faltantes por columna
```

Hay varios: para algunos autos no se conoce el tipo de combustible, la categoría
de mercado (`market_category`), los caballos de fuerza o los cilindros.

> Hay que tenerlo en cuenta: **antes de entrenar habrá que hacer algo con los
> valores faltantes**.

## Resumen

1. Recorrimos las columnas y sus valores para entender los datos.
2. El precio tiene **cola larga** → la eliminamos con `np.log1p`.
3. Hay **valores faltantes** que habrá que tratar antes de entrenar.

Próximo paso: armar el **framework de validación**.

---

**Anterior:** [2.2 — Preparación de datos](02-preparacion-datos.md) ·
**Siguiente:** [2.4 — Framework de validación](04-framework-validacion.md)
