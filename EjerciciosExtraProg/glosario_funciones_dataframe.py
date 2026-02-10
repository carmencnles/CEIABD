"""
GLOSARIO DE FUNCIONES PARA ANÁLISIS Y LIMPIEZA DE DATAFRAMES
=============================================================
Funciones nativas de pandas, numpy y otras librerías útiles
"""

import pandas as pd
import numpy as np
from scipy import stats

# Cargar datos de ejemplo
df = pd.read_csv('datos.csv')
# df = pd.read_excel('datos.xlsx')
# df = pd.read_json('datos.json')
# df = pd.read_sql(query, connection)


# ============================================================================
# 1. INSPECCIÓN BÁSICA
# ============================================================================

# Primeras/últimas filas
df.head()
df.head(10)
df.tail()
df.tail(20)

# Muestra aleatoria
df.sample(5)
df.sample(n=10)
df.sample(frac=0.1)  # 10% aleatorio

# Información general
df.info()
df.info(memory_usage='deep')  # memoria detallada
df.describe()  # estadísticas de columnas numéricas
df.describe(include='all')  # todas las columnas
df.describe(include=['object'])  # solo categóricas
df.describe(include=[np.number])  # solo numéricas
df.describe(percentiles=[.1, .25, .5, .75, .9])  # percentiles personalizados

# Dimensiones y estructura
df.shape  # (filas, columnas)
df.size  # total elementos
len(df)  # número de filas
df.ndim  # número de dimensiones
df.columns  # nombres de columnas
df.columns.tolist()
df.index
df.dtypes  # tipos de datos
df.dtypes.value_counts()  # contar tipos

# Memoria
df.memory_usage()
df.memory_usage(deep=True)
df.memory_usage(deep=True).sum()


# ============================================================================
# 2. VALORES NULOS / FALTANTES
# ============================================================================

# Detectar nulos
df.isnull()  # matriz booleana
df.isna()  # alias de isnull()
df.notnull()
df.notna()

# Contar nulos
df.isnull().sum()  # por columna
df.isnull().sum(axis=1)  # por fila
df.isnull().sum().sum()  # total
df.isnull().any()  # columnas con al menos un nulo
df.isnull().all()  # columnas todo nulos

# Porcentaje de nulos
(df.isnull().sum() / len(df)) * 100
df.isnull().mean() * 100

# Filtrar filas/columnas con nulos
df[df.isnull().any(axis=1)]  # filas con algún nulo
df[df.isnull().all(axis=1)]  # filas todo nulos
df.loc[:, df.isnull().any()]  # columnas con algún nulo

# Eliminar nulos
df.dropna()  # elimina filas con algún nulo
df.dropna(how='all')  # elimina solo si todos son nulos
df.dropna(subset=['columna1', 'columna2'])  # solo en columnas específicas
df.dropna(thresh=5)  # mantener solo filas con al menos 5 no-nulos
df.dropna(axis=1)  # eliminar columnas con nulos
df.dropna(inplace=True)  # modificar el dataframe original

# Rellenar nulos
df.fillna(0)  # con un valor
df.fillna({'col1': 0, 'col2': 'desconocido'})  # diferentes valores por columna
df.fillna(method='ffill')  # forward fill (valor anterior)
df.fillna(method='bfill')  # backward fill (valor posterior)
df.fillna(df.mean())  # con la media
df.fillna(df.median())  # con la mediana
df.fillna(df.mode().iloc[0])  # con la moda
df['columna'].fillna(df['columna'].mean())  # rellenar columna específica
df.interpolate()  # interpolación lineal
df.interpolate(method='polynomial', order=2)  # interpolación polinomial

# Reemplazar valores específicos que podrían ser nulos
df.replace(['?', '-', 'N/A', 'NA', 'null'], np.nan)
df.replace('', np.nan)
df.replace(to_replace=r'^\s*$', value=np.nan, regex=True)  # espacios vacíos


# ============================================================================
# 3. DUPLICADOS
# ============================================================================

# Detectar duplicados
df.duplicated()  # filas duplicadas (booleano)
df.duplicated(keep='first')  # marca duplicados excepto primera ocurrencia
df.duplicated(keep='last')  # marca duplicados excepto última ocurrencia
df.duplicated(keep=False)  # marca todas las ocurrencias
df.duplicated(subset=['col1', 'col2'])  # duplicados en columnas específicas

# Contar duplicados
df.duplicated().sum()
df.duplicated(keep=False).sum()

# Ver duplicados
df[df.duplicated()]
df[df.duplicated(keep=False)]
df[df.duplicated(subset=['columna'])]

# Eliminar duplicados
df.drop_duplicates()
df.drop_duplicates(keep='first')
df.drop_duplicates(keep='last')
df.drop_duplicates(keep=False)  # elimina todas las ocurrencias
df.drop_duplicates(subset=['col1'])
df.drop_duplicates(subset=['col1', 'col2'], keep='last')
df.drop_duplicates(inplace=True)


# ============================================================================
# 4. TIPOS DE DATOS
# ============================================================================

# Verificar tipos
df.dtypes
df['columna'].dtype
pd.api.types.is_numeric_dtype(df['columna'])
pd.api.types.is_string_dtype(df['columna'])
pd.api.types.is_categorical_dtype(df['columna'])
pd.api.types.is_datetime64_any_dtype(df['columna'])
pd.api.types.is_bool_dtype(df['columna'])
pd.api.types.is_integer_dtype(df['columna'])
pd.api.types.is_float_dtype(df['columna'])

# Seleccionar por tipo
df.select_dtypes(include=[np.number])  # solo numéricas
df.select_dtypes(include=['int64', 'float64'])
df.select_dtypes(include=['object'])  # solo strings/categóricas
df.select_dtypes(exclude=[np.number])  # excluir numéricas
df.select_dtypes(include=['datetime64'])

# Convertir tipos
df['columna'].astype(int)
df['columna'].astype(float)
df['columna'].astype(str)
df['columna'].astype('category')
df['columna'].astype(bool)

pd.to_numeric(df['columna'])  # convertir a numérico
pd.to_numeric(df['columna'], errors='coerce')  # no válidos -> NaN
pd.to_numeric(df['columna'], errors='ignore')  # mantener originales si falla
pd.to_numeric(df['columna'], downcast='integer')  # optimizar memoria

pd.to_datetime(df['columna'])
pd.to_datetime(df['columna'], format='%Y-%m-%d')
pd.to_datetime(df['columna'], errors='coerce')
pd.to_datetime(df['columna'], dayfirst=True)

pd.to_timedelta(df['columna'])

# Convertir categóricas
df['columna'].astype('category')
pd.Categorical(df['columna'])
df['columna'] = pd.Categorical(df['columna'], categories=['A', 'B', 'C'], ordered=True)


# ============================================================================
# 5. VALORES ÚNICOS
# ============================================================================

# Contar únicos
df['columna'].nunique()  # número de valores únicos
df['columna'].nunique(dropna=False)  # incluir NaN en el conteo
df.nunique()  # por todas las columnas

# Ver valores únicos
df['columna'].unique()  # array de valores únicos
df['columna'].unique().tolist()
df['columna'].value_counts()  # frecuencia de valores
df['columna'].value_counts(normalize=True)  # frecuencias relativas
df['columna'].value_counts(dropna=False)  # incluir NaN
df['columna'].value_counts(sort=False)
df['columna'].value_counts(ascending=True)
df['columna'].value_counts(bins=5)  # para datos numéricos

# Modo (valor más frecuente)
df['columna'].mode()
df['columna'].mode()[0]  # primera moda


# ============================================================================
# 6. ESTADÍSTICAS DESCRIPTIVAS - NUMÉRICAS
# ============================================================================

# Medidas centrales
df['columna'].mean()  # media
df['columna'].median()  # mediana
df['columna'].mode()  # moda
df.mean()  # media de todas las numéricas
df.median()

# Medidas de dispersión
df['columna'].std()  # desviación estándar
df['columna'].var()  # varianza
df['columna'].sem()  # error estándar de la media
df['columna'].mad()  # desviación absoluta media
df.std()

# Valores extremos
df['columna'].min()
df['columna'].max()
df['columna'].idxmin()  # índice del mínimo
df['columna'].idxmax()  # índice del máximo
df.min()
df.max()

# Cuantiles y percentiles
df['columna'].quantile(0.25)  # Q1
df['columna'].quantile(0.5)  # mediana
df['columna'].quantile(0.75)  # Q3
df['columna'].quantile([0.1, 0.25, 0.5, 0.75, 0.9])

# Rango intercuartílico
df['columna'].quantile(0.75) - df['columna'].quantile(0.25)

# Suma y producto
df['columna'].sum()
df['columna'].prod()
df['columna'].cumsum()  # suma acumulada
df['columna'].cumprod()  # producto acumulado

# Asimetría y curtosis
df['columna'].skew()  # asimetría
df['columna'].kurtosis()  # curtosis (exceso)
df['columna'].kurt()  # alias


# ============================================================================
# 7. OUTLIERS / VALORES ATÍPICOS
# ============================================================================

# Método IQR (Rango Intercuartílico)
Q1 = df['columna'].quantile(0.25)
Q3 = df['columna'].quantile(0.75)
IQR = Q3 - Q1
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR
outliers = df[(df['columna'] < limite_inferior) | (df['columna'] > limite_superior)]
df_sin_outliers = df[(df['columna'] >= limite_inferior) & (df['columna'] <= limite_superior)]

# Método Z-score
from scipy import stats
z_scores = np.abs(stats.zscore(df['columna'].dropna()))
outliers = df[z_scores > 3]
df_sin_outliers = df[z_scores <= 3]

# Método percentiles
limite_inf = df['columna'].quantile(0.01)
limite_sup = df['columna'].quantile(0.99)
df_sin_outliers = df[(df['columna'] >= limite_inf) & (df['columna'] <= limite_sup)]

# Winsorización (limitar extremos)
from scipy.stats.mstats import winsorize
df['columna_winsorized'] = winsorize(df['columna'], limits=[0.05, 0.05])

# Capping (imputar límites)
df['columna'].clip(lower=limite_inferior, upper=limite_superior)


# ============================================================================
# 8. CORRELACIONES
# ============================================================================

# Matriz de correlación
df.corr()  # Pearson por defecto
df.corr(method='pearson')
df.corr(method='spearman')
df.corr(method='kendall')

# Correlación entre dos columnas
df['col1'].corr(df['col2'])

# Covarianza
df.cov()
df['col1'].cov(df['col2'])

# Correlaciones con una columna específica
df.corrwith(df['target'])

# Encontrar correlaciones altas
corr_matrix = df.corr().abs()
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.8)]


# ============================================================================
# 9. OPERACIONES CON STRINGS (COLUMNAS OBJECT)
# ============================================================================

# Limpieza básica
df['columna'].str.strip()  # eliminar espacios inicio/fin
df['columna'].str.lstrip()  # solo inicio
df['columna'].str.rstrip()  # solo fin
df['columna'].str.lower()  # minúsculas
df['columna'].str.upper()  # mayúsculas
df['columna'].str.title()  # primera letra mayúscula
df['columna'].str.capitalize()  # solo primera palabra capitalizada

# Reemplazar
df['columna'].str.replace('viejo', 'nuevo')
df['columna'].str.replace(r'\d+', '', regex=True)  # eliminar números
df['columna'].str.replace(r'\s+', ' ', regex=True)  # espacios múltiples a uno

# Buscar y filtrar
df['columna'].str.contains('texto')
df['columna'].str.contains('texto', case=False)  # ignorar mayúsculas
df['columna'].str.contains('texto', na=False)  # tratar NaN como False
df['columna'].str.startswith('pre')
df['columna'].str.endswith('suf')
df['columna'].str.match(r'^[A-Z]')  # regex desde inicio
df['columna'].str.findall(r'\d+')  # encontrar todos los números

# Extraer
df['columna'].str.extract(r'(\d+)')  # primer grupo
df['columna'].str.extractall(r'(\d+)')  # todos los grupos
df['columna'].str.split(' ')  # dividir en lista
df['columna'].str.split(' ', expand=True)  # dividir en columnas
df['columna'].str.slice(0, 5)  # substring

# Longitud y conteo
df['columna'].str.len()
df['columna'].str.count('a')

# Validaciones
df['columna'].str.isdigit()
df['columna'].str.isalpha()
df['columna'].str.isalnum()
df['columna'].str.isnumeric()
df['columna'].str.isdecimal()
df['columna'].str.isspace()
df['columna'].str.islower()
df['columna'].str.isupper()


# ============================================================================
# 10. OPERACIONES CON FECHAS
# ============================================================================

# Extraer componentes
df['fecha'].dt.year
df['fecha'].dt.month
df['fecha'].dt.day
df['fecha'].dt.hour
df['fecha'].dt.minute
df['fecha'].dt.second
df['fecha'].dt.dayofweek  # 0=lunes, 6=domingo
df['fecha'].dt.dayofyear
df['fecha'].dt.week
df['fecha'].dt.quarter
df['fecha'].dt.weekday  # alias de dayofweek
df['fecha'].dt.day_name()  # nombre del día
df['fecha'].dt.month_name()  # nombre del mes

# Operaciones
df['fecha'].dt.date  # solo fecha
df['fecha'].dt.time  # solo hora
df['fecha'].dt.normalize()  # poner hora a 00:00:00
df['fecha'] + pd.Timedelta(days=1)  # sumar días
df['fecha'] - pd.Timedelta(hours=2)  # restar horas
df['fecha2'] - df['fecha1']  # diferencia entre fechas

# Validaciones
df['fecha'].dt.is_month_start
df['fecha'].dt.is_month_end
df['fecha'].dt.is_quarter_start
df['fecha'].dt.is_quarter_end
df['fecha'].dt.is_year_start
df['fecha'].dt.is_year_end
df['fecha'].dt.is_leap_year

# Formatear
df['fecha'].dt.strftime('%Y-%m-%d')
df['fecha'].dt.strftime('%d/%m/%Y %H:%M')

# Zona horaria
df['fecha'].dt.tz_localize('UTC')
df['fecha'].dt.tz_convert('Europe/Madrid')

# Redondear
df['fecha'].dt.floor('D')  # inicio del día
df['fecha'].dt.ceil('D')  # fin del día
df['fecha'].dt.round('H')  # redondear a hora


# ============================================================================
# 11. FILTRADO Y SELECCIÓN
# ============================================================================

# Por condición
df[df['columna'] > 10]
df[df['columna'] == 'valor']
df[df['columna'].isin(['A', 'B', 'C'])]
df[~df['columna'].isin(['A', 'B'])]  # NOT IN
df[(df['col1'] > 10) & (df['col2'] < 20)]
df[(df['col1'] > 10) | (df['col2'] < 20)]

# Por posición
df.iloc[0]  # primera fila
df.iloc[0:5]  # primeras 5 filas
df.iloc[:, 0]  # primera columna
df.iloc[0:5, 0:3]  # bloque

# Por etiqueta
df.loc[0]  # fila con índice 0
df.loc[0:5]  # filas 0 a 5 (inclusivo)
df.loc[:, 'columna']
df.loc[df['columna'] > 10, ['col1', 'col2']]

# Query (más legible)
df.query('columna > 10')
df.query('columna > 10 and col2 < 20')
df.query('columna in ["A", "B", "C"]')
df.query('columna == @variable')  # usar variable externa

# Filtrar con función
df[df['columna'].apply(lambda x: len(x) > 5)]
df.filter(items=['col1', 'col2'])  # seleccionar columnas
df.filter(like='total')  # columnas que contienen 'total'
df.filter(regex='^col')  # columnas que empiezan con 'col'


# ============================================================================
# 12. AGRUPACIÓN Y AGREGACIÓN
# ============================================================================

# GroupBy básico
df.groupby('columna').mean()
df.groupby('columna').sum()
df.groupby('columna').count()
df.groupby('columna').size()
df.groupby(['col1', 'col2']).mean()

# Múltiples agregaciones
df.groupby('columna').agg(['mean', 'sum', 'count'])
df.groupby('columna').agg({'col1': 'mean', 'col2': 'sum'})
df.groupby('columna').agg(
    promedio=('col1', 'mean'),
    total=('col2', 'sum'),
    conteo=('col3', 'count')
)

# Funciones personalizadas
df.groupby('columna').agg(lambda x: x.max() - x.min())

# Transformar (mantiene tamaño original)
df.groupby('columna')['valor'].transform('mean')
df['media_grupo'] = df.groupby('columna')['valor'].transform('mean')

# Filtrar grupos
df.groupby('columna').filter(lambda x: len(x) > 5)
df.groupby('columna').filter(lambda x: x['valor'].mean() > 10)


# ============================================================================
# 13. PIVOTING Y RESHAPING
# ============================================================================

# Pivot table
df.pivot_table(values='valor', index='fila', columns='columna')
df.pivot_table(values='valor', index='fila', columns='columna', aggfunc='mean')
df.pivot_table(values='valor', index='fila', columns='columna', aggfunc=['mean', 'sum'])
df.pivot_table(values='valor', index='fila', columns='columna', fill_value=0)

# Pivot
df.pivot(index='fila', columns='columna', values='valor')

# Melt (wide to long)
pd.melt(df, id_vars=['id'], value_vars=['col1', 'col2'])
pd.melt(df, id_vars=['id'], var_name='variable', value_name='valor')

# Stack/Unstack
df.stack()  # columnas a filas
df.unstack()  # filas a columnas
df.unstack(level=0)


# ============================================================================
# 14. COMBINACIÓN DE DATAFRAMES
# ============================================================================

# Concatenar
pd.concat([df1, df2])  # verticalmente
pd.concat([df1, df2], axis=1)  # horizontalmente
pd.concat([df1, df2], ignore_index=True)  # resetear índice
pd.concat([df1, df2], keys=['df1', 'df2'])  # con identificador

# Merge (SQL-like joins)
pd.merge(df1, df2, on='columna')  # inner join
pd.merge(df1, df2, left_on='col1', right_on='col2')
pd.merge(df1, df2, on='columna', how='left')  # left join
pd.merge(df1, df2, on='columna', how='right')  # right join
pd.merge(df1, df2, on='columna', how='outer')  # full outer join
pd.merge(df1, df2, on=['col1', 'col2'])  # múltiples columnas
pd.merge(df1, df2, left_index=True, right_index=True)  # join por índice

# Join
df1.join(df2)  # join por índice
df1.join(df2, how='left')
df1.join(df2, on='columna')


# ============================================================================
# 15. ORDENAMIENTO
# ============================================================================

# Ordenar por valores
df.sort_values('columna')
df.sort_values('columna', ascending=False)
df.sort_values(['col1', 'col2'])
df.sort_values(['col1', 'col2'], ascending=[True, False])
df.sort_values('columna', na_position='first')
df.sort_values('columna', inplace=True)

# Ordenar por índice
df.sort_index()
df.sort_index(ascending=False)
df.sort_index(axis=1)  # ordenar columnas

# Ranking
df['columna'].rank()
df['columna'].rank(method='min')  # empates toman el menor valor
df['columna'].rank(method='max')  # empates toman el mayor valor
df['columna'].rank(method='dense')  # sin gaps
df['columna'].rank(ascending=False)
df['columna'].rank(pct=True)  # percentiles


# ============================================================================
# 16. COLUMNAS - OPERACIONES
# ============================================================================

# Renombrar
df.rename(columns={'viejo': 'nuevo'})
df.rename(columns={'col1': 'nueva1', 'col2': 'nueva2'})
df.rename(columns=str.lower)  # todas a minúsculas
df.rename(columns=lambda x: x.strip())
df.columns = ['nuevo1', 'nuevo2', 'nuevo3']  # reemplazar todos

# Eliminar columnas
df.drop('columna', axis=1)
df.drop(['col1', 'col2'], axis=1)
df.drop(columns=['col1', 'col2'])
del df['columna']

# Seleccionar columnas
df[['col1', 'col2']]
df.filter(items=['col1', 'col2'])
df.loc[:, ['col1', 'col2']]

# Reordenar columnas
df[['col2', 'col1', 'col3']]
cols = df.columns.tolist()
df = df[cols[-1:] + cols[:-1]]  # última columna al principio


# ============================================================================
# 17. FILAS - OPERACIONES
# ============================================================================

# Eliminar filas
df.drop(0)  # por índice
df.drop([0, 1, 2])
df.drop(index=[0, 1, 2])

# Reset índice
df.reset_index()
df.reset_index(drop=True)  # sin crear columna del índice viejo

# Set índice
df.set_index('columna')
df.set_index(['col1', 'col2'])  # multiíndice

# Reindexar
df.reindex([3, 1, 0, 2])
df.reindex(range(100), fill_value=0)


# ============================================================================
# 18. APLICAR FUNCIONES
# ============================================================================

# Apply
df['columna'].apply(lambda x: x * 2)
df['columna'].apply(str.lower)
df.apply(lambda x: x.max() - x.min())  # por columna
df.apply(lambda x: x.max() - x.min(), axis=1)  # por fila

# Applymap (a cada elemento) - DEPRECADO en pandas 2.1+
df.applymap(lambda x: x * 2)
# Nuevo método:
df.map(lambda x: x * 2)

# Map (para Series)
df['columna'].map({'A': 1, 'B': 2, 'C': 3})
df['columna'].map(lambda x: x.upper())

# Replace con funciones
df.replace({'A': 1, 'B': 2})


# ============================================================================
# 19. BINNING Y DISCRETIZACIÓN
# ============================================================================

# Cut (bins de igual ancho)
pd.cut(df['columna'], bins=5)
pd.cut(df['columna'], bins=[0, 10, 20, 30, 40])
pd.cut(df['columna'], bins=5, labels=['Muy Bajo', 'Bajo', 'Medio', 'Alto', 'Muy Alto'])

# Qcut (bins de igual frecuencia)
pd.qcut(df['columna'], q=4)  # cuartiles
pd.qcut(df['columna'], q=10)  # deciles
pd.qcut(df['columna'], q=[0, 0.25, 0.5, 0.75, 1])


# ============================================================================
# 20. VALIDACIÓN DE DATOS
# ============================================================================

# Verificar valores en rango
df['columna'].between(10, 20)
df[df['columna'].between(10, 20)]

# Verificar valores válidos
df['columna'].isin(['A', 'B', 'C'])
df[~df['columna'].isin(['A', 'B', 'C'])]  # valores NO válidos

# Todos/Alguno
df['columna'].all()  # todos True
df['columna'].any()  # alguno True
(df['columna'] > 0).all()
(df['columna'] > 0).any()

# Encontrar valores no válidos
df[pd.to_numeric(df['columna'], errors='coerce').isna()]  # no numéricos


# ============================================================================
# 21. SAMPLING Y PARTICIONAMIENTO
# ============================================================================

# Muestra aleatoria
df.sample(n=100)
df.sample(frac=0.1)  # 10%
df.sample(n=100, random_state=42)  # reproducible
df.sample(n=100, replace=True)  # con reemplazo

# Train/test split
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.2, random_state=42)

# Muestreo estratificado
df.groupby('categoria', group_keys=False).apply(lambda x: x.sample(frac=0.1))


# ============================================================================
# 22. EXPORTAR DATOS
# ============================================================================

# CSV
df.to_csv('output.csv')
df.to_csv('output.csv', index=False)
df.to_csv('output.csv', sep=';')
df.to_csv('output.csv', encoding='utf-8')

# Excel
df.to_excel('output.xlsx', index=False)
df.to_excel('output.xlsx', sheet_name='Datos')

# JSON
df.to_json('output.json')
df.to_json('output.json', orient='records')

# SQL
df.to_sql('tabla', connection, if_exists='replace')

# Pickle (formato pandas)
df.to_pickle('output.pkl')

# Clipboard
df.to_clipboard()


# ============================================================================
# 23. OPTIMIZACIÓN DE MEMORIA
# ============================================================================

# Downcast numéricos
df['columna'] = pd.to_numeric(df['columna'], downcast='integer')
df['columna'] = pd.to_numeric(df['columna'], downcast='float')

# Convertir a categorías (ahorra mucha memoria)
df['columna'] = df['columna'].astype('category')

# Usar tipos eficientes
df['columna'] = df['columna'].astype('int8')  # si valores pequeños
df['columna'] = df['columna'].astype('int16')
df['columna'] = df['columna'].astype('int32')

# Sparse (para datos con muchos ceros)
df['columna'] = df['columna'].astype(pd.SparseDtype("float", 0))


# ============================================================================
# 24. FUNCIONES AVANZADAS DE LIMPIEZA
# ============================================================================

# Eliminar espacios en nombres de columnas
df.columns = df.columns.str.strip()
df.columns = df.columns.str.replace(' ', '_')

# Estandarizar nombres de columnas
df.columns = df.columns.str.lower()
df.columns = df.columns.str.replace('[^a-z0-9_]', '', regex=True)

# Detectar y eliminar columnas constantes
df.loc[:, df.nunique() > 1]

# Detectar y eliminar columnas con alta cardinalidad
df.loc[:, df.nunique() < len(df) * 0.95]

# Detectar filas con demasiados nulos
umbral = 0.5
df[df.isnull().mean(axis=1) < umbral]

# Imputación avanzada
from sklearn.impute import SimpleImputer, KNNImputer
imputer = SimpleImputer(strategy='mean')
df[['col1', 'col2']] = imputer.fit_transform(df[['col1', 'col2']])

imputer = KNNImputer(n_neighbors=5)
df[['col1', 'col2']] = imputer.fit_transform(df[['col1', 'col2']])

# Encoding de categóricas
df['columna_encoded'] = pd.factorize(df['columna'])[0]
df_encoded = pd.get_dummies(df, columns=['columna'])
df_encoded = pd.get_dummies(df, columns=['columna'], drop_first=True)

# Label encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['columna_encoded'] = le.fit_transform(df['columna'])

# One-hot encoding
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)
encoded = encoder.fit_transform(df[['columna']])


# ============================================================================
# 25. ANÁLISIS DE CALIDAD - FUNCIONES AUXILIARES
# ============================================================================

# Perfil completo de calidad
def data_quality_report(df):
    report = pd.DataFrame({
        'dtype': df.dtypes,
        'non_null_count': df.count(),
        'null_count': df.isnull().sum(),
        'null_percentage': (df.isnull().sum() / len(df) * 100).round(2),
        'unique_count': df.nunique(),
        'unique_percentage': (df.nunique() / len(df) * 100).round(2)
    })
    return report

# Resumen de duplicados por columna
def duplicates_summary(df):
    dup_summary = {}
    for col in df.columns:
        dup_count = df[col].duplicated().sum()
        if dup_count > 0:
            dup_summary[col] = {
                'count': dup_count,
                'percentage': (dup_count / len(df) * 100).round(2)
            }
    return pd.DataFrame(dup_summary).T

# Matriz de nulos
df.isnull().sum().to_frame('null_count')

# Cross-tabulation para analizar relaciones
pd.crosstab(df['col1'], df['col2'])
pd.crosstab(df['col1'], df['col2'], normalize='index')
pd.crosstab(df['col1'], df['col2'], margins=True)
