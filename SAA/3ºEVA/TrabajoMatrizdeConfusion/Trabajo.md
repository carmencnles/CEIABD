# Trabajo de investigación sobre la matriz de confusión

## 1 - Introducción

### Contexto del problema

Cuando se habla de incendios forestales, no se suele tener en cuenta que no todos son iguales. Puede ser un pequeño fuego controlado que no afecte al ecosistema o uno que arrase con varias hectareas. Por ello, lo que necesitan los equipos de emergencia, no es solamente si hay un incendio o no, mas bien como de fuerte es ese incendio.   

Para este dataset usamos datos proporcionados por satelites de la NASA y datos del clima (como el calor que hace o que tan fuerte sopla el viento).
El objetivo es ver si con esa información metereologica podemos predecir que tan fuerte va a ser el indencio (Baja, Moderada, Alta o Extrema).   

El corazón del analisis no es solamente ver en que acierta el modelo, si no, en que se equivoca, utilizando la matriz de confusión revisaremos y observaremos como se equivocan entre las diferentes clases.

### Objetivos del trabajo

El objetivo de este trabajo es entender que tan bien funciona nuestro "adivino" de incendios y donde están sus puntos debiles.
Haremos:
- Entrenar los dos modelos: crear un sistema que aprenda a clasificar la intensidad del fuego según el clima.
- Observar la matriz de confusión: revisar posibles confusiones con clases cercanas como Extrema y Alta.
- Revisar errores: observar si hay errores como que prediga que es "bajo" cuando en realidad es alto, es un error grave que afecta en la realidad.

## 2 - Descripción del dataset

### Origen y características
El dataset le he obtenido de Kaggle: 

Wildfire Risk Dataset 2024-2025 | 7 Regions
[link text](https://www.kaggle.com/datasets/alitaqishah/wildfire-risk-dataset-2024-2025-7-regions)

Es un dataset que recoge datos satelitales de focos de incendios forestales, complementado con datos meteorologicos diarios que abarcan siete regiones propensas (Asia Meridional, Sudeste Asiático, Mediterráneo,
África Subsahariana, Sudamérica, Norteamérica, Australia) de incendios a nivel mundial.
Tenemos 15.500 eventos de detección de incendios en 35 países.  

El periodo del dataset: enero de 2024 – diciembre de 2025
> Detecciones de incendios: NASA FIRMS (VIIRS NOAA-20)     
> Enriquecimiento de datos meteorológicos: Open-Meteo Historical API   


### Distribución de clases

El tema que voy a escoger:  Análisis de la matriz de confusión en clasificación multiclase.   

Para entender el rendimiento del modelo, primero debemos observar cómo se reparten los 15.500 eventos entre las cuatro categorías de fire_intensity. En problemas de incendios forestales, lo habitual es encontrar un desbalance de clases:

- Low / Moderate: Suelen ser las clases mayoritarias (fuegos pequeños, quemas agrícolas o controladas).

- High / Extreme: Suelen ser las clases minoritarias (eventos catastróficos), pero son precisamente las que más nos interesa predecir correctamente.

## 3 - Preparación de los datos

### Limpieza y transformaciones

### Justificación de decisiones (evitar data leakage)

Se ha decidido evitar la codificación ordinal (0, 1, 2, 3) de la variable intensidad para no introducir un sesgo jerárquico artificial en el aprendizaje del modelo. Aunque la intensidad tiene un orden lógico, tratar las etiquetas como categorías nominales permite que el modelo identifique los patrones climáticos de cada nivel de manera independiente.    
Esto garantiza que el análisis de la matriz de confusión refleje la capacidad real de discriminación del algoritmo sin interferencias por la magnitud asignada a la etiqueta.

## 4 - Modelos de clasificación

### Modelos implementados

### Justificación de su elección


## 5 - Matriz de confusión y métricas

### Matrices obtenidas por modelo

### Análisis de falsos positivos y falsos negativos


## 6 - Evaluación y comparación de resultados

### Comparación entre modelos

### Discusión alineada con el tema elegido

## 7 - Conclusiones y limitaciones


## 8 - Líneas de mejora y trabajo futuro


## 9 - Referencias


## 10 - Anexo - Uso de herramientas de Inteligencia Artificial