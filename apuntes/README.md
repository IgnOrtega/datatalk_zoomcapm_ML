# Apuntes — ML Zoomcamp, Módulo 1: Introducción al Machine Learning

Apuntes tomados de las transcripciones de las 10 clases del módulo 1.

## Índice

| # | Clase | Idea central |
|---|-------|--------------|
| [1.1](01-introduccion-ml.md) | Introducción al ML | Features + target → modelo → predicciones |
| [1.2](02-ml-vs-reglas.md) | ML vs sistemas de reglas | Las reglas no escalan; el ML aprende los patrones solo |
| [1.3](03-aprendizaje-supervisado.md) | Aprendizaje supervisado | `g(X) ≈ y`; regresión, clasificación, ranking |
| [1.4](04-crisp-dm.md) | CRISP-DM | Las 6 fases de un proyecto de ML; iterar siempre |
| [1.5](05-seleccion-de-modelos.md) | Selección de modelos | Split train/validation/test y el problema de comparaciones múltiples |
| [1.6](06-entorno-codespaces.md) | Entorno (Codespaces) | Ambiente de trabajo y entrega de tareas |
| [1.7](07-numpy.md) | NumPy | Arrays, operaciones element-wise, agregaciones |
| [1.8](08-algebra-lineal.md) | Álgebra lineal | Productos vector-vector, matriz-vector, matriz-matriz, inversa |
| [1.9](09-pandas.md) | Pandas | DataFrames, Series, filtrado, group by, valores faltantes |
| [1.10](10-resumen.md) | Resumen de la sesión 1 | Repaso de todo lo anterior |

## Hilo conductor del módulo

El módulo arma, de lo conceptual a lo práctico, la base para el módulo 2 (regresión lineal aplicada a predecir el precio de un auto):

1. **Qué es el ML** (1.1–1.3): extraer patrones de datos en vez de escribir reglas a mano.
2. **Cómo se organiza un proyecto** (1.4–1.5): CRISP-DM para el proceso completo, y el proceso de selección de modelos para la parte de modelado.
3. **Herramientas** (1.6–1.9): entorno, NumPy, álgebra lineal y Pandas.

## Conceptos que atraviesan todo el módulo

- **Features (X)**: todo lo que sabemos del objeto. Matriz: filas = observaciones, columnas = características.
- **Target (y)**: lo que queremos predecir. Vector.
- **Modelo (g)**: la función que se entrena para que `g(X)` se parezca lo más posible a `y`.
- **Entrenar (fit)**: el proceso de obtener `g` a partir de `X` e `y`.
- El modelo acierta **en promedio**, no caso por caso.

---

*Nota: las transcripciones son automáticas y traen errores de reconocimiento
(«crisp them» por CRISP-DM, «psyit learn» por scikit-learn, «palm/sperm» por spam,
etc.). En estos apuntes los términos van corregidos.*
