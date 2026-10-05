# 2.1 — Proyecto: predicción del precio de autos

## El problema

Retomamos el escenario de la introducción: un sitio de **clasificados online**
para vender autos. El usuario que publica su auto tiene que poner un precio y no
siempre sabe cuánto pedir. Queremos un modelo que le **sugiera el mejor precio**.

## El dataset

Usamos un dataset de precios de autos de **Kaggle**. Cada fila es un auto con sus
características (features):

- marca (make), modelo, año
- motor, tipo de combustible, transmisión, etc.
- y el precio: la columna **MSRP**

> **MSRP** = *Manufacturer Suggested Retail Price*, el precio de venta sugerido por
> el fabricante. Es nuestro **target**: lo que queremos predecir.

Idea general: a partir de las features del auto, predecir su precio.

## Plan del proyecto

| Paso | Qué se hace |
|------|-------------|
| 1 | Obtener los datos y hacer **análisis exploratorio** (EDA) |
| 2 | Preparar el dataset |
| 3 | Entrenar una **regresión lineal** para predecir el precio |
| 4 | Ver cómo funciona por dentro: **implementarla nosotros mismos** |
| 5 | Evaluar el modelo con **RMSE** (*root mean squared error*) |
| 6 | **Feature engineering**: crear nuevas features para el modelo |
| 7 | Resolver problemas de **estabilidad numérica** con **regularización** |
| 8 | Usar el modelo |

## Dónde está el código

Todo está en el repo del libro, `mlbookcamp-code`, en la carpeta de *chapter 2,
car price*. Hay dos archivos:

- el **notebook** con todo el código de la sesión;
- el **CSV** con el dataset que usaremos para entrenar.

> En clase se hace live coding y el notebook queda disponible.

El siguiente paso no es todavía el EDA: primero hay que **cargar y preparar** el CSV.

---

**Anterior:** [1.10 — Resumen de la sesión 1](../chapter_1/10-resumen.md) ·
**Siguiente:** [2.2 — Preparación de datos](02-preparacion-datos.md)
