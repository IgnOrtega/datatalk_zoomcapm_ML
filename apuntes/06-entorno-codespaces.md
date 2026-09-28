# 1.6 — Entorno de trabajo (GitHub Codespaces, edición 2024)

Clase práctica: cómo armar el ambiente de trabajo y entregar las tareas. La gracia
de Codespaces es que **casi no requiere configuración**: es un entorno remoto que
ya viene con casi todo.

> El resumen de la sesión (1.10) menciona la alternativa clásica: instalar
> **Anaconda** localmente, que trae NumPy, Pandas, scikit-learn y Jupyter de una.
> También se puede levantar un servidor en AWS u otro cloud.

## Crear el entorno

1. Crear un repositorio nuevo en GitHub (por ejemplo `ml-zoomcamp-homework`).
   Público, con README y `.gitignore` de Python.
2. Botón **Code** → pestaña **Codespaces** → *Create codespace on main*.
3. Se levanta un VS Code en el navegador. Se puede seguir ahí, o abrirlo en el
   VS Code de escritorio con *Open in VS Code Desktop*.

> Para conectarse desde el VS Code de escritorio hace falta la **extensión GitHub
> Codespaces**. Normalmente la ofrece sola al abrir; si no, se busca en la pestaña
> de extensiones.

Por dentro es una máquina Linux (Ubuntu) remota, pero se trabaja como si fuera
local: terminal con `Ctrl + ñ` / `` Ctrl + ` `` (o *View → Terminal*), sistema de
archivos normal, git funcionando.

## Instalar las librerías

```bash
pip install jupyter numpy pandas scikit-learn seaborn
```

Con eso alcanza para los primeros cuatro o cinco módulos. Más adelante hacen falta
`xgboost` y `tensorflow`, que se instalan igual.

## Levantar Jupyter

```bash
jupyter notebook
```

Codespaces **detecta automáticamente** el puerto 8888 y lo redirige a la máquina
local (pestaña **Ports**). Se abre el enlace y se pega el token —o la URL
completa— que muestra la terminal.

## Tip de la clase: acortar el prompt

El prompt de Codespaces es larguísimo y no deja ver lo que uno escribe:

```bash
PS1="> "
```

## Flujo de trabajo con git

```bash
git status
git commit -am "readme update"
git push
```

Ejemplo del final de la clase: crear una carpeta `01-intro`, adentro un notebook,
leer el CSV de la tarea con pandas…

```python
import pandas as pd
df = pd.read_csv("<archivo de la tarea>.csv")
```

…completar, guardar, y después:

```bash
git add homework.ipynb
git commit -m "homework"
git push
```

## Entregar la tarea

Se va al sitio del curso (*Machine Learning Zoomcamp → homework 1*), se completan
las respuestas y se pega la URL del repositorio como enlace de la tarea.

---

**Anterior:** [1.5 — Selección de modelos](05-seleccion-de-modelos.md) ·
**Siguiente:** [1.7 — NumPy](07-numpy.md)
