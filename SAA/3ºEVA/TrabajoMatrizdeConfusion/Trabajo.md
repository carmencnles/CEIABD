# **Trabajo de investigación sobre la matriz de confusión**

### Incendios Forestales

**Autor:** Maria del Carmen Canales    

**Fecha:** 12/05/2026

## Índice
1. [Introducción](#1---introducción)
   - [Contexto del problema](#contexto-del-problema)
   - [Objetivos del trabajo](#objetivos-del-trabajo)
2. [Descripción del dataset](#2---descripción-del-dataset)
   - [Origen y características](#origen-y-características)
   - [Distribución de clases](#distribución-de-clases)
3. [Preparación de los datos](#3---preparación-de-los-datos)
   - [Limpieza y transformaciones](#limpieza-y-transformaciones)
   - [Justificación de decisiones (evitar data leakage)](#justificación-de-decisiones-evitar-data-leakage)
4. [Modelos de clasificación](#4---modelos-de-clasificación)
   - [Modelos implementados](#modelos-implementados)
   - [Justificación de su elección](#justificación-de-su-elección)
5. [Matriz de confusión y métricas](#5---matriz-de-confusión-y-métricas)
   - [Matrices obtenidas por modelo](#matrices-obtenidas-por-modelo)
   - [Análisis de falsos positivos y falsos negativos](#análisis-de-falsos-positivos-y-falsos-negativos)
6. [Evaluación y comparación de resultados](#6---evaluación-y-comparación-de-resultados)
   - [Comparación entre modelos](#comparación-entre-modelos)
   - [Discusión alineada con el tema elegido](#discusión-alineada-con-el-tema-elegido)
7. [Conclusiones y limitaciones](#7---conclusiones-y-limitaciones)
8. [Líneas de mejora y trabajo futuro](#8---líneas-de-mejora-y-trabajo-futuro)
9. [Referencias](#9---referencias)
10. [Anexo - Uso de herramientas de Inteligencia Artificial](#10---anexo---uso-de-herramientas-de-inteligencia-artificial)


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

En este trabajo se han seleccionado 2 algoritmos de aprendizaje supervisado distintos para abordar el problema de clasificación multiclase de la intensidad de los incendios:

- Support Vector Machine (SVM) con Kernel Lineal: Un modelo que busca encontrar el hiperplano óptimo que separa las clases. Se utilizó la implementación SVC de Scikit-learn.

- Random Forest Classifier: Un modelo de ensamble basado en una multitud de árboles de decisión. Se configuró con 1000 estimadores y una profundidad máxima de 20 para capturar patrones no lineales y evitar el sobreajuste.

### Justificación de su elección

La elección de este binomio permite realizar una comparativa técnica profunda basada en el comportamiento de la matriz de confusión:

- Contraste Metodológico: El SVM Lineal nos sirve como base de referencia (baseline). Al ser un modelo lineal, nos permite comprobar si la intensidad de un incendio se puede predecir simplemente por el aumento gradual de temperatura o descenso de humedad. Si este modelo funciona bien, significa que la relación es directa y predecible.

- Gestión de Datos Complejos: El Random Forest se ha elegido por su capacidad para manejar datos que no tienen una relación lineal clara (por ejemplo, el viento solo es peligroso si la humedad es muy baja). Es un modelo muy robusto frente al ruido y permite analizar la importancia de cada variable.

- Solución al Desequilibrio de Clases: Ambos modelos permiten el uso del parámetro class_weight='balanced'. Esta es la decisión técnica más importante del trabajo, ya que sin este ajuste, ambos modelos ignoraban las clases minoritarias (High y Extreme) para centrarse en la clase mayoritaria (Moderate).

- Análisis de Fronteras: Comparar la matriz de confusión de un SVM (que divide el espacio en regiones planas) frente a un Random Forest (que crea divisiones rectangulares) ayuda a entender mejor si las categorías de intensidad están bien definidas o si se solapan en los datos meteorológicos.


## 5 - Matriz de confusión y métricas

### Matrices obtenidas por modelo

En esta sección se presentan las matrices de confusión que comparan el desempeño del SVM Lineal y el Random Forest, analizando cómo influye el balanceo de clases en la interpretación de los resultados.

- Modelo sin balancear (SVM Lineal/Random Forest base): Las primeras pruebas arrojaron un Accuracy engañosamente alto (aprox. 0.42). Sin embargo, al observar la matriz de confusión, se identificó un sesgo de clase mayoritaria. El modelo predecía casi exclusivamente la categoría "Moderate", ignorando por completo los incendios "Extreme" y "Low". Estadísticamente acertaba mucho porque la mayoría de los datos son "Moderate", pero su utilidad práctica era nula.

- Modelo con balanceo (class_weight='balanced'): Al aplicar el balanceo, el Accuracy descendió (a valores entre 0.16 y 0.36), pero la matriz de confusión mostró una distribución real. Los modelos empezaron a "arriesgarse" a clasificar incendios en todas las categorías, llenando la diagonal principal de la matriz.

### Análisis de falsos positivos y falsos negativos

- Errores Adyacentes (Leves): Se observa que la mayoría de los fallos ocurren entre niveles contiguos (ej. clasificar un incendio High como Moderate). Esto indica que el modelo captura la tendencia de la intensidad, pero la frontera climática entre estos niveles es borrosa.

- Falsos Negativos Críticos (Graves): El error más peligroso detectado en la matriz es clasificar incendios reales Extreme como Low o Moderate. Estos falsos negativos supondrían un fallo en los sistemas de emergencia. Se observa que el Random Forest mitiga mejor estos errores que el SVM Lineal gracias a su capacidad de detectar interacciones complejas en los datos de la NASA.

- Falsos Positivos de Alerta: Clasificar un incendio Low como Extreme. Aunque es un error, en la gestión de incendios se considera preferible (falsa alarma) antes que omitir un incendio catastrófico. El uso de class_weight='balanced' aumentó intencionadamente estos casos para asegurar que el modelo no ignorara las clases críticas.

## 6 - Evaluación y comparación de resultados

### Comparación entre modelos

### Discusión alineada con el tema elegido

## 7 - Conclusiones y limitaciones


## 8 - Líneas de mejora y trabajo futuro


## 9 - Referencias


## 10 - Anexo - Uso de herramientas de Inteligencia Artificial

Gemini:
- ¿Debería elimninar la columna confidence de este dataset? ¿Que significa?
- 