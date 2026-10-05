# 1.10 — Resumen de la sesión 1

Repaso rápido de todo el módulo.

## 1.1 — Introducción al ML

Ejemplo del precio de autos. **Features** = todas las características que conocemos
del auto; **target** = lo que queremos predecir (el precio). Features y target
entran a un algoritmo de ML y sale un **modelo**, que después se usa con autos
cuyo precio no conocemos.

## 1.2 — Reglas vs machine learning

En un sistema de reglas los humanos encuentran los patrones a mano y los traducen
a código. Se vuelve inmanejable con el tiempo. Con ML los modelos extraen los
patrones solos: miran los datos de entrenamiento y usan estadística y matemática
para descubrir qué sirve para decidir.

## 1.3 — Aprendizaje supervisado

Ambos ejemplos son aprendizaje supervisado porque tenemos el **target** (`y`): la
información que queremos predecir y que ya conocemos. El modelo `g` extrae
patrones de la matriz de features `X` y produce algo lo más cercano posible a `y`.

Tipos según el target: regresión (número), clasificación (categoría; multiclase o
binaria) y ranking.

## 1.4 — CRISP-DM

El modelado (`g(X) = y`) es **solo una parte** del proceso completo. Alrededor
están el entendimiento del negocio, el de los datos, la preparación (la `X` hay
que armarla), y el despliegue — porque sin despliegue **ni el mejor modelo sirve
de nada**.

## 1.5 — Selección de modelos

Se parte el dataset en **tres**: se usa validation para encontrar el mejor modelo
y test para asegurarse de no haber elegido uno que quedó bien **por pura
casualidad**.

## 1.6 — Entorno

No es tanto una clase como la preparación del ambiente: NumPy, Pandas,
scikit-learn, Jupyter. La opción más fácil es Anaconda; también se puede usar
GitHub Codespaces o un servidor en AWS u otro cloud.

## 1.7 — NumPy

Librería de Python para manipular datos numéricos (arrays), con las operaciones
útiles para data science y machine learning.

## 1.8 — Álgebra lineal

Los tres productos: **vector-vector**, **matriz-vector** (`U` mayúscula por `v`
minúscula) y **matriz-matriz**.

La relación entre ellos es lo importante: el producto matriz-matriz se expresa
como un conjunto de productos matriz-vector, y el matriz-vector como un conjunto
de productos vector-vector. Implementados en código, las fórmulas dejan de
asustar.

## 1.9 — Pandas

Librería para procesar datos tabulares. La abstracción principal es el
**DataFrame**.

---

## Qué viene después

Esta sesión fue **abstracta**: qué se puede hacer con machine learning, con NumPy,
con Pandas.

La siguiente es **práctica**: un proyecto completo prediciendo el precio de un
auto (regresión lineal).

---

**Anterior:** [1.9 — Pandas](09-pandas.md) · **Índice:** [README](README.md)
