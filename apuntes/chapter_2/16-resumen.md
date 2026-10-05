# 2.16 — Resumen de la sesión 2

Repaso de todo el proyecto de predicción del precio de autos.

## El proyecto (2.1)

Dataset con precios y características de autos: marca, modelo, motor, tipo de
combustible, transmisión, etc. El target es **MSRP** (*manufacturer's suggested
retail price*, precio sugerido por el fabricante).

## Preparación de datos (2.2)

Se limpió el dataset para que quedara **uniforme**: nombres de columnas y
valores con espacios, mayúsculas y minúsculas mezcladas → todo en minúscula y
con `_`.

## Análisis exploratorio (2.3)

- El precio tenía una distribución de **cola larga** (*long tail*). Se eliminó
  aplicando la **transformación logarítmica**, porque los modelos de ML suelen
  tener problemas con colas largas.
- Había **valores faltantes**, con los que no se puede entrenar un modelo: hay
  que hacer algo con ellos.

## Framework de validación (2.4)

Split en **train / validation / test**.

## Regresión lineal (2.5–2.6)

Primero para un solo ejemplo, como fórmula con un `for`; después en forma
vectorial con **producto punto**, y luego como producto **matriz-vector**. La
salida de la regresión lineal son los pesos: el **bias** `w0` y el **vector de
pesos** `w`.

## Ecuación normal (2.7)

Cómo obtener los pesos. El ML **no es magia**: es una fórmula, la **ecuación
normal**, implementada en NumPy:

```
w = (XᵀX)⁻¹ Xᵀ y
```

## Modelo base (2.8)

Primer modelo con solo **cinco features numéricas** básicas. No anduvo muy bien
según el gráfico, pero un gráfico no permite medir el desempeño **de forma
objetiva**.

## RMSE (2.9–2.10)

**Root mean squared error**: métrica para evaluar modelos de regresión. Se armó
el framework de validación con la función `prepare_X`, que prepara la feature
matrix **igual** para train y validation. Bastaba con redefinirla y copiar la
misma celda a lo largo de la sesión → **experimentar mucho más rápido**.

## Feature engineering (2.11)

Crear features nuevas a partir de las existentes. La feature **`age`** mejoró
el modelo drásticamente: la distribución de las predicciones se parece mucho más
a la de los valores reales (no igual, pero mucho mejor).

## Variables categóricas (2.12)

Cada variable categórica se representa con un conjunto de **columnas binarias**.
Esta codificación se llama **one-hot encoding** (se profundiza en la sesión de
clasificación).

> El RMSE se volvió enorme de repente. Gracias al dataset de validación el
> problema se detectó fácilmente.

## Regularización (2.13)

La causa era **inestabilidad numérica**. Solución: sumar un número pequeño a la
**diagonal de `XᵀX`** antes de invertirla:

```python
XTX = XTX + r * np.eye(XTX.shape[0])
```

Con las features categóricas y regularización, el modelo mejoró bastante
respecto de la versión anterior.

## Ajuste del modelo (2.14)

Se probaron distintos valores del parámetro de regularización. Se eligió
`r = 0.001`: quizás no es el mejor, pero está al mismo nivel que los otros.

## Usar el modelo (2.15)

- Se juntaron train y validation en un **full train** y se entrenó el modelo
  final (otra vez con `prepare_X`, muy cómodo).
- Se aplicó a un auto del test **fingiendo no conocer su precio**: la predicción
  no estuvo lejos del precio real.

| Paso | Herramienta clave |
|---|---|
| Limpieza | `str.lower()`, `str.replace(' ', '_')` |
| Target | `np.log1p` / `np.expm1` |
| Split | train / val / test |
| Entrenamiento | ecuación normal + regularización |
| Evaluación | RMSE |
| Features | `prepare_X`, `age`, one-hot |
| Modelo final | `pd.concat`, `np.concatenate` |

---

## Qué viene después

- Una sección **solo de texto** (sin video) con otras cosas para probar y
  aprender más sobre el tema.
- La **tarea** (*homework*), para aplicar lo aprendido por cuenta propia.
- La próxima sesión es **clasificación**: en lugar de implementar todo a mano
  como aquí, se usará **scikit-learn**.

> Ya vimos cómo implementar las cosas nosotros mismos; ahora estamos listos para
> usar la librería.

---

**Anterior:** [2.15 — Usar el modelo](15-usar-el-modelo.md)
