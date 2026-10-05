# Apuntes — ML Zoomcamp, Módulo 2: Regresión (predicción del precio de autos)

Apuntes tomados de las transcripciones de las 16 clases del módulo 2.
PDF: [ML-Zoomcamp-Modulo-2-Apuntes.pdf](ML-Zoomcamp-Modulo-2-Apuntes.pdf)

Informe complementario: [Variables categóricas de alta cardinalidad](Categoricas-Alta-Cardinalidad.md) ([PDF](Categoricas-Alta-Cardinalidad.pdf))

## Índice

| # | Clase | Idea central |
|---|-------|--------------|
| [2.1](01-proyecto-precio-autos.md) | Proyecto: precio de autos | Dataset de Kaggle; el target es MSRP |
| [2.2](02-preparacion-datos.md) | Preparación de datos | Nombres y valores uniformes: minúsculas y `_` |
| [2.3](03-analisis-exploratorio.md) | Análisis exploratorio | Cola larga en el precio → `np.log1p`; valores faltantes |
| [2.4](04-framework-validacion.md) | Framework de validación | Split 60/20/20 aleatorio en train/val/test |
| [2.5](05-regresion-lineal.md) | Regresión lineal | `g(xᵢ) = w0 + Σ wⱼ·xᵢⱼ` para un solo auto |
| [2.6](06-regresion-lineal-vectorial.md) | Forma vectorial | Producto punto y luego `X·w` para todo el dataset |
| [2.7](07-ecuacion-normal.md) | Ecuación normal | `w = (XᵀX)⁻¹ Xᵀ y` implementada en NumPy |
| [2.8](08-modelo-base.md) | Modelo base | 5 features numéricas, `fillna(0)`, primer modelo |
| [2.9](09-rmse.md) | RMSE | Métrica objetiva para regresión |
| [2.10](10-rmse-validacion.md) | RMSE en validación | `prepare_X` para tratar train y val igual |
| [2.11](11-feature-engineering.md) | Feature engineering | La feature `age` mejora mucho el modelo |
| [2.12](12-variables-categoricas.md) | Variables categóricas | One-hot encoding… y el RMSE explota |
| [2.13](13-regularizacion.md) | Regularización | Sumar `r` a la diagonal de `XᵀX` |
| [2.14](14-ajuste-modelo.md) | Ajuste del modelo | Probar valores de `r` en validación |
| [2.15](15-usar-el-modelo.md) | Usar el modelo | Entrenar con train+val, evaluar en test, predecir un auto |
| [2.16](16-resumen.md) | Resumen de la sesión 2 | Repaso de todo el proyecto |

## Hilo conductor del módulo

Un proyecto completo de punta a punta siguiendo CRISP-DM y el proceso de
selección de modelos del módulo 1:

1. **Datos** (2.1–2.4): entender, limpiar y explorar el dataset; separar train/validation/test.
2. **Modelo** (2.5–2.7): regresión lineal implementada desde cero con NumPy.
3. **Evaluar e iterar** (2.8–2.14): baseline → RMSE → nuevas features → categóricas → regularización → ajuste.
4. **Usar** (2.15): modelo final y predicción para un auto nuevo.

---

*Nota: las transcripciones son automáticas. Fueron reparadas (formato, puntuación
y términos mal reconocidos como «zoom cam» → Zoomcamp), pero algunos números
dictados pueden no ser exactos: p. ej. el valor de regularización aparece como
0.01 en 2.13–2.15 y como 0.001 en 2.16; los apuntes usan 0.001, como el notebook
del curso.*
