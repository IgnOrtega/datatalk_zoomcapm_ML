# 1.4 — CRISP-DM: cómo se organiza un proyecto de ML

**CRISP-DM** = *CRoss-Industry Standard Process for Data Mining*. Metodología
creada por IBM en los años 90. Es vieja, pero resistió el paso del tiempo: se
sigue usando hoy casi sin modificaciones.

Son **6 fases**, y el punto central es que **se itera**: casi cualquier fase puede
mandarte de vuelta a una anterior.

```
  ┌──────────────────────────────────────────────┐
  │                                              ▼
Business ──► Data ──► Data ──► Modeling ──► Evaluation ──► Deployment
Understanding  Understanding  Preparation        │              │
      ▲             ▲              ▲             │              │
      │             └──────────────┴─────────────┘              │
      └─────────────────────────────────────────────────────────┘
```

---

## 1. Business understanding — entender el problema

Objetivo: identificar el problema, entender si es importante y **definir cómo se
mide el éxito**.

La pregunta obligatoria de esta fase:

> **¿De verdad necesitamos machine learning?**

Muchos problemas no lo necesitan. «Si tenés un martillo, todo parece un clavo»:
a veces basta con un sistema de reglas o una heurística, sin invertir tiempo y
recursos en ML.

**En el ejemplo del spam:**

- ¿Cuántos usuarios se quejan? ¿Es un problema real o es un solo usuario? Esto
  también sirve para justificar la inversión de tiempo.
- ¿Es ML la herramienta correcta, o alcanza con reglas?
- Definir la métrica: no basta con «queremos reducir el spam». Hay que decir
  **cuánto**: «reducir los mensajes de spam en un 50 %». Sin un número no hay
  forma de decir después si el proyecto fue exitoso.

## 2. Data understanding — qué datos tenemos

Sin datos no hay machine learning. Hay que averiguar qué hay disponible y si
alcanza.

Preguntas:

- **¿El dato es confiable?** El botón «marcar como spam», ¿funciona bien?
  ¿Se registra siempre el clic?
- **¿El dato es lo bastante bueno?** Si los usuarios marcan como spam mensajes que
  no lo son, el modelo va a aprender ese comportamiento y a reproducir el error.
  Hay que analizar los datos a mano.
- **¿Hay suficiente volumen?** Con 10 registros no se hace nada. Una salida
  legítima de esta fase es: «todavía no estamos listos, primero hay que juntar
  2000 o 3000 registros».

Lo que se aprende acá puede cambiar el entendimiento del problema → **volver a la
fase 1** y revisarla.

## 3. Data preparation — preparar los datos

Transformar los datos crudos en algo que se le pueda dar a un algoritmo.

- **Limpieza**: sacar ruido, corregir los casos mal etiquetados.
- **Pipelines**: una secuencia de pasos que toma datos crudos, aplica
  transformaciones y produce datos limpios. Es código que hay que escribir.
- **Formato tabular**: remitente, destinatario, asunto, cuerpo… y la columna con
  el target.
- **Extracción de features**: por ejemplo, la columna binaria «¿contiene la
  palabra deposit?».

El resultado es la **X** y la **y** de la clase anterior.

## 4. Modeling — entrenar modelos

Acá pasa el machine learning propiamente dicho.

> El objetivo es **probar distintos modelos y elegir el mejor**.

Regresión logística, árboles de decisión, redes neuronales y muchos más — se ven
a lo largo del curso. Cómo elegir el mejor es el tema de la clase 1.5.

Muy seguido acá se descubre que las features no alcanzan, o que hay problemas en
los datos → **volver a la fase 3**.

## 5. Evaluation — ¿alcanzamos el objetivo?

Se vuelve a mirar la meta definida en la fase 1: dijimos «reducir el spam un 50 %».
¿Lo logramos? ¿Redujimos un 30 %? ¿Es suficiente?

Las salidas posibles incluyen una que suele olvidarse: **cancelar el proyecto**.
Si la meta resultó inalcanzable, se hace una retrospectiva y se decide si vale la
pena seguir invirtiendo tiempo o no.

> **Nota del instructor:** hoy evaluación y despliegue van casi siempre juntos.
> En los 90 se evaluaba primero y se desplegaba después; ahora la forma de evaluar
> es desplegando.

Esto es la **evaluación online**: se prueba con usuarios reales, pero no con todos
— por ejemplo con el 5 %. Si funciona bien, se extiende al resto.

## 6. Deployment — poner el modelo en producción

Si en la fase de modelado el foco era el machine learning, acá el foco es la
**ingeniería**: que el servicio sea monitoreable, mantenible, escalable y
confiable. Todas las buenas prácticas de ingeniería entran en juego en esta fase.

## Iterar

No se termina en el despliegue. Se vuelve al *business understanding* con lo
aprendido: ¿se puede mejorar? ¿Hace falta? Tal vez alcanza por ahora, y seis meses
después aparecen problemas y arranca otra vuelta.

> **Consejo central de la clase: empezar simple.**
> Hacer una iteración rápida y completa con algo sencillo, desplegar, aprender.
> Después volver y hacerlo un poco más complejo. Dos o tres iteraciones rápidas
> en vez de una larga: se pierde menos tiempo y se demuestra valor temprano.

---

**Anterior:** [1.3 — Aprendizaje supervisado](03-aprendizaje-supervisado.md) ·
**Siguiente:** [1.5 — Selección de modelos](05-seleccion-de-modelos.md)
