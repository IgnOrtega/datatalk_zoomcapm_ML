# 1.1 — Introducción al Machine Learning

## El ejemplo: predecir el precio de un auto

Un sitio de avisos clasificados de autos. Cuando alguien publica su auto, llega al
campo **precio** y se traba:

- Precio muy alto → nadie compra.
- Precio muy bajo → deja plata sobre la mesa.

La solución manual es mirar avisos similares y estimar. La pregunta del curso es:
¿cómo ayudamos al usuario a encontrar el precio justo? Con machine learning.

## De dónde sale el conocimiento

Ya tenemos datos de autos publicados: precio, año de fabricación, marca,
kilometraje, modelo, número de puertas, etc.

Un **experto** de una concesionaria puede estimar el precio mirando esas mismas
características. ¿Cómo aprendió? Mirando muchísimos casos anteriores y extrayendo
patrones: los autos viejos valen menos, más kilometraje baja el precio, una BMW
cuesta más que una Volkswagen.

> La idea clave: si un experto puede aprender esos patrones de los datos,
> un modelo también.

## Vocabulario

| Término | Definición | En el ejemplo |
|---------|-----------|---------------|
| **Features** | Todo lo que sabemos del objeto | Año, marca, kilometraje, modelo, puertas |
| **Target** | Lo que queremos predecir | El precio |
| **Modelo** | Lo que encapsula los patrones aprendidos | — |

## El flujo completo

**Entrenamiento:**

```
Features + Target  →  [algoritmo de ML]  →  Modelo
```

**Uso (predicción):**

```
Features (sin target)  →  [Modelo]  →  Predicción
```

Para autos nuevos no sabemos el precio: le pasamos al modelo solo las features y
nos devuelve el precio estimado.

## Precisión: el promedio, no el caso

El modelo **no acierta siempre el precio exacto** de un auto específico. Puede
predecir de más o de menos. Pero en promedio, para un auto de ese año, esa marca y
ese kilometraje, la predicción es razonablemente correcta.

Esto no es un defecto a corregir: es la naturaleza del ML y hay que tenerlo
presente al comunicar resultados.

## Cómo se cierra el círculo en el producto

1. El usuario llena el formulario con los datos de su auto.
2. Extraemos las features de ese formulario.
3. El modelo devuelve un precio.
4. Ponemos ese precio en el campo como sugerencia.
5. El usuario lo ajusta si quiere (subirlo o bajarlo).

El usuario queda contento porque se ahorró toda la investigación manual.

## Definición resumida

> **Machine learning es el proceso de extraer patrones de los datos.**

Los datos son de dos tipos: **features** (información sobre el objeto) y **target**
(lo que queremos predecir). La salida del proceso es un **modelo**, que después se
usa con features nuevas para producir predicciones.

---

**Siguiente:** [1.2 — ML vs sistemas basados en reglas](02-ml-vs-reglas.md)
