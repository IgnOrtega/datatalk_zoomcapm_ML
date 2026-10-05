# 1.2 — Machine Learning vs sistemas basados en reglas

El ejemplo de esta clase es un **detector de spam** para un sistema de correo:
queremos mandar los mensajes no deseados a la carpeta de spam.

## El camino de las reglas

Analizamos los mensajes marcados como spam y encontramos patrones:

- Todo lo que viene de `promotions@online.com` es spam.
- Si el asunto contiene «tax review» y el dominio es `online.com`, es spam.

Codificamos esas reglas en Python: unos cuantos `if`, se despliega y funciona.

### Por qué se rompe

1. Aparece otro tipo de spam (mensajes que piden un depósito de 10 dólares).
2. Agregamos la regla: si el cuerpo contiene «deposit» → spam.
3. Un usuario legítimo escribe sobre un depósito real y su mensaje cae en spam.
4. Hay que agregar más reglas para distinguir los dos casos.
5. Y otra vez, y otra vez.

El spam cambia constantemente, así que las reglas hay que mantenerlas al día para
siempre. El código crece, se vuelve imposible de mantener y cada cambio rompe algo
en otro lado. Ahí es cuando toca preguntarse si hay otra herramienta.

## El camino del ML

Cuatro pasos: **obtener datos → extraer features → entrenar el modelo → usarlo**.

### 1. Datos

El botón «marcar como spam» ya genera las etiquetas: los usuarios nos dicen qué es
spam y qué no. Eso es exactamente el target que necesitamos.

### 2. Features

Features **binarias** (solo valen 0 o 1), por ejemplo:

1. ¿El asunto tiene más de 10 caracteres?
2. ¿El cuerpo tiene más de 10 caracteres?
3. ¿El remitente es `promotions@online.com`?
4. ¿El remitente es *(otro remitente sospechoso)*?
5. ¿El dominio del remitente es `test.com`?
6. ¿La descripción contiene la palabra «deposit»?

> **Punto importante:** estas features **salen de las reglas que ya teníamos**.
> Por eso conviene empezar con un sistema de reglas y después reusar esas reglas
> como features del modelo. No hay que saltar directo al ML.

Cada email se codifica como un vector: `[1, 1, 0, 0, 1, 1]` con target `1` (spam).
Se repite para todos los emails y queda el dataset.

### 3. Entrenar

Features + target entran al algoritmo; el proceso de entrenar también se llama
**fit**. Sale el modelo.

### 4. Usar el modelo

El modelo devuelve **probabilidades**, no etiquetas: 0.8, 0.6, 0.1, 0.01, 0.7, 0.4.
Se interpretan como «qué tan seguro estoy de que esto es spam».

Después hay que tomar la **decisión**, con un umbral (típicamente 0.5):

| Probabilidad | Decisión | Destino |
|---|---|---|
| 0.8 | spam | carpeta spam |
| 0.6 | spam | carpeta spam |
| 0.1 | no spam | bandeja de entrada |
| 0.01 | no spam | bandeja de entrada |
| 0.7 | spam | carpeta spam |
| 0.4 | no spam | bandeja de entrada |

La probabilidad y la decisión son cosas distintas: el modelo da la primera, el
umbral produce la segunda.

## La diferencia de fondo

Esta es la comparación clave del módulo — **qué es entrada y qué es salida**:

| | Software tradicional | Machine learning |
|---|---|---|
| **Entrada** | Datos + **código** (reglas) | Datos + **resultado** (etiquetas) |
| **Salida** | Resultado (spam / no spam) | **Modelo** |

En software tradicional el resultado está *hard-codeado* en las reglas. En ML, el
resultado es una **entrada** al algoritmo: le mostramos qué es spam y qué no, y la
salida es un modelo que después produce resultados para casos que no vimos.

---

**Anterior:** [1.1 — Introducción](01-introduccion-ml.md) ·
**Siguiente:** [1.3 — Aprendizaje supervisado](03-aprendizaje-supervisado.md)
