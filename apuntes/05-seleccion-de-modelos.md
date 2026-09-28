# 1.5 — El proceso de selección de modelos

Esta clase profundiza la fase de **modeling** de CRISP-DM: cómo elegir el mejor
modelo entre varios candidatos.

## El problema: evaluar como si fuera el futuro

Cuando usamos un modelo en producción pasa esto:

- En **julio** entrenamos el modelo `g` con los datos que tenemos hasta ahí.
- En **agosto** lo aplicamos a mensajes que nunca vio.

Al evaluar queremos imitar esa situación: medir qué tan bien anda el modelo con
datos que **no vio durante el entrenamiento**. No podemos viajar al futuro, pero
sí podemos simularlo.

## Solución: esconder parte de los datos

Tomamos el dataset completo, apartamos un ~20 % y hacemos de cuenta que no existe:

- **80 % → training**: lo usamos para entrenar (es «julio»).
- **20 % → validation**: no se usa para entrenar (es «agosto»).

```
X_train, y_train  ──►  entrenar  ──►  g
X_val             ──►  g  ──►  ŷ    (comparar con y_val)
```

## Cómo se mide

El modelo devuelve probabilidades; con el umbral de 0.5 se convierten en
predicciones y se comparan con el valor real:

| ŷ (prob.) | predicción | real | ¿correcto? |
|-----------|-----------|------|------------|
| 0.8 | 1 | 1 | ✔ |
| 0.7 | 1 | 0 | ✘ |
| 0.6 | 1 | 1 | ✔ |
| 0.1 | 0 | 0 | ✔ |
| 0.9 | 1 | 1 | ✔ |
| 0.6 | 1 | 0 | ✘ |

4 de 6 correctas → **66 % de accuracy**.

Se repite con varios modelos:

| Modelo | Accuracy |
|--------|----------|
| Regresión logística | 66 % |
| Árbol de decisión | 60 % |
| Random forest | 67 % |
| Red neuronal | **80 %** |

Parecería que ya está: elegimos la red neuronal. **Pero hay un problema.**

## El problema de las comparaciones múltiples

Ejemplo ilustrativo: en vez de modelos, usemos **monedas**. El modelo es «tiro la
moneda: cara = spam, cruz = no spam». Probamos con cinco monedas distintas sobre
las mismas 5 observaciones de validación:

| «Modelo» | Aciertos |
|----------|----------|
| Euro | 20 % |
| Dólar | 40 % |
| Zloty polaco | 20 % |
| Rublo | 20 % |
| Grivna ucraniana | **100 %** |

La grivna «gana» con 100 % de accuracy. Pero sabemos que es puro azar: esa moneda
simplemente produjo por casualidad la misma secuencia que el target.

> **Lo mismo puede pasar con modelos reales.** Los métodos de ML son
> probabilísticos: uno de ellos puede tener suerte con ese subconjunto particular
> de datos, sin ninguna razón de fondo.

En estadística esto se llama **problema de comparaciones múltiples**: cuando se
hace la misma comparación muchas veces contra el mismo conjunto de validación,
alguno va a salir bien por azar. Si tomáramos otro 20 % de los datos, el resultado
sería completamente distinto.

## Solución: tres particiones

En vez de apartar un conjunto, apartamos **dos**:

```
┌───────────────────────┬───────────┬───────────┐
│      train 60 %       │  val 20 % │ test 20 % │
└───────────────────────┴───────────┴───────────┘
```

> El 60/20/20 no está escrito en piedra: pueden ser otras proporciones.

El **test set** se guarda y se olvida. Se hace toda la selección de modelos contra
validation y, recién cuando ya elegimos el mejor, se lo aplica una sola vez al test
set para confirmar que no tuvo suerte.

En el ejemplo: la red neuronal dio 80 % en validation y da 79 % en test. Son
valores razonablemente cercanos → el modelo se comporta bien de verdad.

## El proceso, paso a paso

1. **Dividir** el dataset en train / validation / test.
2. **Entrenar** el modelo con train.
3. **Validar** contra validation y registrar la métrica.
4. **Repetir** 2 y 3 para todos los modelos candidatos.
5. **Seleccionar** el mejor.
6. **Testear** el elegido contra el test set para confirmar.

> Montar bien este proceso es una de las cosas más importantes del machine
> learning.

## Refinamiento: no desperdiciar el validation set

Tal como está, el 20 % de validación termina «gastado»: solo sirvió para comparar.
Se puede aprovechar mejor:

1. Hacer la selección de modelos normalmente.
2. Una vez elegido el mejor, **unir train + validation** (80 % de los datos).
3. **Reentrenar** el modelo con ese conjunto más grande → un `g` mejor, porque usa
   más datos.
4. Recién ahí evaluarlo contra el test set.

---

**Anterior:** [1.4 — CRISP-DM](04-crisp-dm.md) ·
**Siguiente:** [1.6 — Entorno de trabajo](06-entorno-codespaces.md)
