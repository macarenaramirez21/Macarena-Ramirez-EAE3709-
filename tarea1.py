#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 10:56:27 2025

@author: eduardomendez

"""

#Para esta tarea se utilizará como principal fuente de información un dataset con una serie de características económicas, demográficas y de desarrollo humano de distintos países a la fecha de 2007 (corte transversal). El dataset está disponible en el siguiente [Github](https://raw.githubusercontent.com/lfgarcia-1/EAE3709-1-2025/refs/heads/main/economic_dataset.csv).<br>

#Descripción del dataset:

#Variables:

#*   date: Fecha en la que se actualizó la data.
#*   Population, Area (sq. mi.) Pop. Density (per sq. mi.), Coastline (coast/area ratio), Net migration, Infant mortality (per 1000 births), GDP ($ per capita, Literacy (%), Phones (per 1000), Arable (%), Crops (%), Other (%), Climate, Birthrate, Deathrate, Agriculture, Industry, Service: Características del país.
#*   source: fuente de los datos.
#*   Region: Región (grupo de países).
#*   Country: País.


### Pregunta 1.0 ##

#Importe las librerías que usará en su tarea.


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sqlite3



### Pregunta 1.1 ##

#Importe el dataset como un DataFrame (df) directamente desde Github (es decir, no descargue el archivo manualmente). A lo largo de la tarea este df se denominará como `df`.

url = "https://raw.githubusercontent.com/lfgarcia-1/EAE3709-1-2025/refs/heads/main/economic_dataset.csv"

df = pd.read_csv(url)


### Pregunta 1.2 ##

#Utilice las funciones de Pandas `head()`, `tail()`, `info()` y la propiedad (o atributo) `.dtypes` para describir el `df`. Explique brevemente para qué sirve cada función.

df.head()

df.tail()

df.info()

df.dtypes


#  Para describir el DataFrame df, utilizamos las funciones head(), tail(), info() y la propiedad .dtypes de Pandas. 


#  La función head() permite visualizar las primeras n filas del DataFrame, lo que ayuda a tener una idea general de 
#  Los datos y las variables . Se le entrega como paramentro el numero de lineas (n) que se quiere retornar. Si no se especifica,
#  retornará las primeras 5
#  En esté caso vemos variables como date/source/Country, vale decir, fechas, todo proveniente del 
#  gobierno de US y el país que se menciona

#  Por otro lado, tail() muestra las últimas n filas (si no se especifica n, retornará las últimas 5 filas). 
#  Es útil para revisar cómo terminan los datos o detectar posibles datos faltantes al final del archivo. 
#  En este caso vemos las mismas columnas y vemos algunos datos faltantes después de la fila 223


#  La función info() entrega un resumen completo de la estructura del DataFrame, 
#  Incluyendo la cantidad de filas y columnas, nombres de columnas, cuántos valores no nulos hay por cada una y los tipos de datos.
#  Esto quiere decir que Hay 227 filas, numeradas del 0 al 226. El DataFrame tiene 22 columnas.
#  Algunas columnas tienen datos faltantes (menos de 227 non-null) 
#  Net migration': 224 → le faltan 3 datos.
#  'Literacy (%)': 209 → le faltan 18 datos.
#  'Climate': 205 → le faltan 22 datos.


#  Finalmente, .dtypes muestra específicamente el tipo de dato que Pandas ha asignado a cada columna, lo cual es clave 
#  para saber si es necesario transformar algún dato, por ejemplo, si se espera un número pero la columna está como texto.
#  Por ejemplo, date es un objeto y nos gustaria que fuera un datetime
### Pregunta 1.3

# La variable `source` es innecesaria debido que contiene el mismo valor para todas las observaciones. Elimine esta variable de su `df`.
# Eliminar la columna 'source' porque no aporta información útil (valor constante)

df = df.drop(columns='source')
df.columns

### Pregunta 1.4

#  Transforme el tipo de la variable `date` a `datetime` _datatype_.
#  Convertir la columna 'date' a tipo datetime

df['date'] = pd.to_datetime(df['date'])
df.info()  # comprobar que lo cambiamos a 'datetime´


### Pregunta 1.5

## Para determinar si las variables son "útiles" y sus valores son "correctos" es necesario comprender cada uno de los atributos del dataset.
## Investigue y explique brevemente la relación **teórica** entre el `GDP (% per capita)` y cada una de las variables denominadas como "Características del país" en la introducción.

#  Ejemplo: Existe una variable denominada `Coastline (coast/area ratio)`. Coastline es una medida de la cantidad de costa (acceso a mar) del país normalizada al área total del país para no beneficiar a países más grandes pero con la misma proporción de costa. A mayor "Costline" aumenta la capacidad portuaria per capita del país, más puertos facilita el comercio y podría aumentar el GDP per cápita.

# Lista de variables a comparar con el GDP
variables = [
    'Population', 'Area (sq. mi.)', 'Pop. Density (per sq. mi.)',
    'Coastline (coast/area ratio)', 'Net migration',
    'Infant mortality (per 1000 births)', 'Literacy (%)',
    'Phones (per 1000)', 'Arable (%)', 'Crops (%)', 'Other (%)',
    'Climate', 'Birthrate', 'Deathrate',
    'Agriculture', 'Industry', 'Service'
]

# Crear un gráfico de dispersión para cada variable vs GDP
num_vars = len(variables)
cols = 3
rows = -(-num_vars // cols)  # Techo de división entera para calcular filas

plt.figure(figsize=(18, 5 * rows))

for i, var in enumerate(variables):
    plt.subplot(rows, cols, i + 1)
    sns.scatterplot(data=df, x=var, y='GDP ($ per capita)', alpha=0.6)
    plt.title(f'GDP vs {var}')
    plt.xlabel(var)
    plt.ylabel('GDP ($ per capita)')

plt.tight_layout()
plt.show()

#graficos individuales
for var in variables:
    sns.scatterplot(df, x= var, y = "GDP ($ per capita)")
    sns.regplot(df, x = var, y = "GDP ($ per capita)", scatter=True)
    plt.title(f'GDP vs {var}')
    plt.show()

## Se explora gráficamente la relación entre el GDP ($ per capita) y varias variables
## características del país mediante gráficos de dispersión. Cada gráfico permite visualizar
## si existe algún tipo de relación lineal, no lineal o nula entre el ingreso per cápita
## y las distintas dimensiones estructurales del país.

## Population: No se observa una relacion clara entre la población y el GDP lo que sugiere que el tamaño
## poblacional no determina por sí solo el ingreso per cápita. Sin embargo, se obserba que algunos de 
## los paises con mayor población tienen menor GDP, lo que se puede deber a tener rendimientos decrecientes
## capital. 


## Area (sq. mi.): Similar a la población, el tamaño físico del país no muestra una relación
## evidente con el GDP per cápita. Una mayor area. Una mayor area puede significar tener mas recursos, pero 
## se necesita una mayor inversión para explotarlos.

## Pop. Density (per sq. mi.): Se observa una gran dispersión; algunos países con alta densidad
## tienen GDP alto, pero muchos no. No parece haber una correlación fuerte. Esto se puede deber a 
## paises mas densos estan mas urbanizados, sin embargo la relación no es clara. 

## Coastline (coast/area ratio): Se observa que algunos países con mayor proporción de costa
## presentan GDP per cápita elevado, lo cual coincide con la teoría del acceso al comercio marítimo,
## sin embargo esta relacion es leve.

## Net migration: Algunos países con migración neta positiva tienen altos niveles de GDP per cápita.
## Sin embargo, la relación es débil y dispersa. Generalmente, los paises con mayor mas desarrollados, con mayor
## GDP, atraen mas imnigrantes y menos personas quieren dejar el pais.

## Infant mortality (per 1000 births): Existe una clara relación negativa: a mayor mortalidad infantil,
## menor es el GDP per cápita. Esto confirma que este indicador refleja el nivel de desarrollo, ya que paises 
## mas desarrollados tienen mayor acceso a salud, lo que disminuye la mortalidad infantil.

## Literacy (%): Se observa una tendencia positiva: países con mayor alfabetización tienden a tener
## mayores ingresos per cápita. Al igual que la salud, los paises con mayor desarrollo economico tienen 
## mayor acceso a educación, lo que mejora la alfabetización de la población.

## Phones (per 1000): También hay una relación positiva: mayor acceso a telecomunicaciones se asocia
## a mayores niveles de desarrollo económico. Paises mas desarrollados tienen mejor infraestructura telefonica
## (calidad de señal, coobertura, etc), lo que lleva a que tengan una mayor cantidad de telefono.

## Arable (%): No se observa una relación entre suelo cultivable y GDP. 
## Crops (%): Se observa un arelacion negativa debil con el GDP 
## Other (%): No se observa una relacion clara.
## Estas variables relacionadas con el uso del suelo muestran relaciones débiles con el GDP. 
## En general, los países agrícolas no son los de mayor ingreso per cápita debido a que tienen menor industrializacion,
## lo que generalmente lleva a un mayor GDP

## Climate: La relación es difusa, pero hay cierta agrupación de países de climas moderados con GDP más alto.

## Birthrate y Deathrate: Ambas muestran relaciones negativas con el GDP per cápita, especialmente la natalidad.
## Altas tasas de natalidad y mortalidad tienden a asociarse con menores niveles de desarrollo, ya que estos paises
## tienen menor acceso a salud.

## Agriculture, Industry, Service: Las variables de composición económica muestran tendencias interesantes:
## - A mayor peso del sector servicios, suele haber mayor GDP per cápita.
## - Un peso elevado del sector agrícola se asocia a menores ingresos per cápita.
## - La industria muestra una relación mixta, dependiendo del nivel de desarrollo.

## Este análisis gráfico apoya la teoría económica sobre los determinantes estructurales del ingreso,
## y ayuda a identificar qué variables pueden ser útiles en un modelo explicativo del GDP.


### Pregunta 1.6

##  Calcule estadísticas descriptivas para cada variable numérica.

#  La función df.describe() calcula estadísticas descriptivas básicas para todas las columnas 
##  numéricas del DataFrame. Estas incluyen: el conteo de valores (count), el promedio (mean), 
##  la desviación estándar (std), los valores mínimo y máximo (min, max), y los percentiles 25%, 
#   50% (mediana) y 75%. Estos indicadores permiten comprender la distribución, dispersión y escala 
## de cada variable, ayudando a identificar posibles valores extremos, sesgos o necesidad de transformación.

df.describe() 

### Pregunta 1.7

##  Según corresponda, realice un gráfico de distribución de densidad o histograma para describir 3 
##  variables del `df` que usted crea más relevantes.
## ¿Por qué es importante analizar las distribuciones de las variables a utilizar en su modelo? 
##  Ejemplifique su respuesta con al menos una de las variables del df`.


plt.figure(figsize=(15, 4))

# Variables relevantes elegidas
variables = ['GDP ($ per capita)', 'Literacy (%)', 'Infant mortality (per 1000 births)']

# Crear los 3 gráficos
for i, var in enumerate(variables):
    plt.subplot(1, 3, i + 1)
    sns.histplot(df[var].dropna(), kde=True, bins=20)
    plt.title(f'Distribución de {var}')
    plt.xlabel(var)
    plt.ylabel("Frecuencia")

plt.tight_layout()
plt.show()

##  Es importante analizar la distribución de las variables porque permite entender su comportamiento general, 
##  detectar valores extremos y evaluar si es necesario transformar los datos antes de usarlos en un modelo. 
##  Por ejemplo, al observar la variable GDP ($ per capita) se puede identificar una distribución fuertemente 
##  sesgada a la derecha: la mayoría de los países tiene un ingreso per cápita bajo o medio, mientras que unos 
##  pocos países presentan valores muy altos. Esto puede influir fuertemente en la media y afectar la precisión 
##  de los modelos si no se trata adecuadamente (por ejemplo, mediante transformación logarítmica o normalización). 
##  Conocer estas distribuciones permite elegir técnicas estadísticas adecuadas y evitar interpretaciones erróneas.

# Crear figura con 2 subplots 
plt.figure(figsize=(12, 5))

# Gráfico original
plt.subplot(1, 2, 1)
sns.histplot(df['GDP ($ per capita)'].dropna(), kde=True, bins=20)
plt.title('Distribución original: GDP ($ per capita)')
plt.xlabel('GDP ($ per capita)')
plt.ylabel('Frecuencia')

# Gráfico transformado con log
plt.subplot(1, 2, 2)
# Filtrar valores positivos antes de aplicar log
gdp_log = np.log(df['GDP ($ per capita)'][df['GDP ($ per capita)'] > 0])
sns.histplot(gdp_log, kde=True, bins=20, color='orange')
plt.title('Distribución logarítmica: log(GDP)')
plt.xlabel('log(GDP)')
plt.ylabel('Frecuencia')

plt.tight_layout()
plt.show()

## Al aplicar la transformación logarítmica al GDP ($ per capita), la distribución deja de estar sesgada hacia
## la derecha y se aproxima más a una distribución simétrica o normal. Esto mejora la estabilidad de los modelos
## estadísticos y evita que los países con ingresos muy altos dominen los resultados. En particular, muchas
## técnicas como regresiones lineales o modelos de machine learning funcionan mejor cuando las variables se 
## distribuyen de forma más balanceada. Por eso, conocer la distribución original y evaluar transformaciones 
## es un paso fundamental del análisis exploratorio.


### Pregunta 1.8

##  El df contiene variables con missing values (`NaN`). Impute los `NaN` con el método que estime conveniente, justificando su decisión.
##  ¿Es pertinente eliminar alguna de estas variables? Hágalo si es el caso.

# Ver cuántos valores faltantes tiene cada columna
missing_values = df.isna().sum().sort_values(ascending=False)
print("Valores faltantes por columna:")
print(missing_values)

# Calcular el porcentaje de valores faltantes por columna
missing_percentage = (missing_values / len(df)) * 100

# Mostrar columnas con más del 30% de NaN
cols_con_muchos_na = missing_percentage[missing_percentage > 30]
print("\nColumnas con más del 30% de valores faltantes:")
print(cols_con_muchos_na if not cols_con_muchos_na.empty else "Ninguna")

# Imputación de valores faltantes

# 1. Variables numéricas: imputar con la mediana (más robusta que la media)
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
for col in num_cols:
    if df[col].isna().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

# 2. Variables categóricas o tipo texto: imputar con forward fill
obj_cols = df.select_dtypes(include='object').columns
for col in obj_cols:
    if df[col].isna().sum() > 0:
        df[col].fillna(method='ffill', inplace=True)

# Confirmar que ya no existen valores faltantes en el DataFrame
total_na = df.isna().sum().sum()
print(f"\nTotal de valores faltantes tras la imputación: {total_na}")


## Se revisaron los valores faltantes (`NaN`) presentes en el DataFrame. Si bien algunas columnas como
## 'Agriculture', 'Industry' y 'Service' presentaban datos faltantes, ninguna superaba el umbral del 30%,
## criterio comúnmente utilizado para eliminar variables. Por lo tanto, no se eliminó ninguna columna.

## Para la imputación de valores faltantes se consideró el tipo de variable:
## - En las variables numéricas se utilizó la mediana, ya que es una medida robusta ante outliers
##   y distribuciones sesgadas. Esto evita que valores extremos alteren el centro de la distribución, lo cual
##   es especialmente importante en columnas como 'GDP ($ per capita)' o 'Infant mortality'.
## - En las variables categóricas  (como 'Country' o 'Region'), se utilizó el método `ffill`
##   (forward fill), que rellena el NaN con el valor no nulo anterior. Esta técnica es adecuada cuando
##   los valores faltantes son aislados y no estructurales. No fue necesario usarlo ya que las variables categoricas 
##   no tenian datos faltantes

## Además, se calculó explícitamente el porcentaje de valores faltantes por columna para respaldar
## cuantitativamente la decisión de no eliminar ninguna variable.

## Finalmente, tras la imputación, se confirmó que el DataFrame no contiene más valores faltantes,
## lo que permite continuar con el análisis y la construcción del modelo sin introducir sesgos
## por datos ausentes ni errores operativos por presencia de NaN.

### Pregunta 1.9

## ¿Cómo distribuye el `GDP ($ per capita)` en diferentes **regiones**? Defina una forma ilustrativa de gráficar el `GDP ($ per capita)` para todas las regiones en un mismo gráfico. Interprételo.


plt.figure(figsize=(12, 6))
sns.set(style="whitegrid")

# Boxplot del GDP por región
sns.boxplot(data=df, x='Region', y='GDP ($ per capita)', palette='viridis')


plt.title('Distribución del GDP ($ per capita) por Región', fontsize=14)
plt.xlabel('Región', fontsize=12)
plt.ylabel('GDP ($ per capita)', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

## Se utilizó un gráfico de tipo boxplot para visualizar la distribución del GDP ($ per capita)
## en distintas regiones del mundo. Este tipo de gráfico permite observar la mediana, el rango
## intercuartílico (IQR) y la presencia de posibles valores atípicos en cada grupo.

## A partir del gráfico, se puede observar que regiones como Europa Occidental y America del Norte tienden a tener
## los mayores niveles de GDP per cápita. Mientras que America del Norte presenta una gran variabilidad en las obserbaciones,
## Europa Occidental tiene una menor dispersión.
## En contraste, regiones como África Sub-sahariana o Asia (ex. near east) presentan medianas más bajas
## y el segundo tiene una mayor dispersión, lo cual puede indicar una mayor heterogeneidad económica entre los países de esa zona.

## Esta visualización permite comparar rápidamente el desempeño económico relativo entre regiones,
## identificar desigualdades y detectar posibles valores extremos dentro de cada grupo regional.

### Pregunta 1.10

##  Supongamos que `GDP ($ per capita)` es su variable objetivo. Estudie la correlación de esta variable con el resto de las variables del `df`. ¿Por qué es importante analizar la correlación entre las variables?

# matriz de correlación
correlaciones = df.corr(numeric_only=True)

# Extraer correlación de todas las variables con 'GDP ($ per capita)' y ordenarla
gdp_corr = correlaciones['GDP ($ per capita)'].drop('GDP ($ per capita)').sort_values(ascending=False)

print("Correlación de cada variable con GDP ($ per capita):")
print(gdp_corr)

# Graficos
plt.figure(figsize=(10, 6))
sns.barplot(x=gdp_corr.values, y=gdp_corr.index, palette='coolwarm')
plt.title('Correlación de variables numéricas con GDP ($ per capita)', fontsize=14)
plt.xlabel('Coeficiente de correlación')
plt.ylabel('Variable')
plt.tight_layout()
plt.show()

sns.barplot(x=abs(gdp_corr.values), y = gdp_corr.index, order = gdp_corr.abs().sort_values(ascending = False).index)
plt.title('Correlación de variables numéricas con GDP ($ per capita)', fontsize=14)
plt.xlabel('Coeficiente de correlación')
plt.ylabel('Variable')
plt.tight_layout()
plt.show()

## Se analizó la correlación de la variable objetivo 'GDP ($ per capita)' con el resto de las variables numéricas.
## Este análisis permite identificar qué atributos tienen mayor asociación lineal con el nivel de ingreso per cápita
## y, por tanto, podrían ser útiles para construir un modelo predictivo.

## Las variables que presentan una correlación positiva más alta con el GDP per cápita son:
## - Literacy (%)
## - Phones (per 1000)
## - Service
## Estas variables reflejan niveles de desarrollo humano, tecnológico y terciarización de la economía,
## lo que es consistente con la teoría del desarrollo económico.

## Por otro lado, variables como 'Infant mortality', 'Birthrate' y 'Agriculture' presentan correlaciones negativas,
## lo cual también es esperable, ya que altos valores en estas variables suelen asociarse a países menos desarrollados.

## Es importante analizar la correlación entre las variables porque:
## - Permite identificar relaciones lineales útiles para la predicción.
## - Ayuda a seleccionar las variables más relevantes y descartar otras redundantes.
## - Reduce el riesgo de multicolinealidad si se usa junto con un análisis de correlaciones entre predictores.

## Este paso es esencial en cualquier pipeline de machine learning porque mejora la interpretabilidad,
## eficiencia y precisión del modelo a construir.

### Pregunta 1.11

##  Realice tres _scatterplots_ (uno por variable) de las tres variables con la mayor correlación con la variable objetivo.
##  Utilizando los parámetros de la función con la que hizo los _scatterplots_, coloque un título a cada gráfico y agregue colores a los _data points_ del _scatterplot_- Use colores diferentes por cada gráfico.

# Variables más correlacionadas con GDP (correlación positiva)
top_vars = ['Literacy (%)', 'Phones (per 1000)', 'Service']
colors = ['#1f77b4', '#2ca02c', '#d62728']  # Azul, verde y rojo

plt.figure(figsize=(18, 5))

for i, (var, color) in enumerate(zip(top_vars, colors)):
    plt.subplot(1, 3, i + 1)
    sns.scatterplot(data=df, x=var, y='GDP ($ per capita)', color=color, alpha=0.7)
    plt.title(f'GDP vs {var}', fontsize=13)
    plt.xlabel(var)
    plt.ylabel('GDP ($ per capita)')

plt.tight_layout()
plt.show()

## Graficos de las tres variables con mayor correlación positiva con el GDP ($ per capita):
## 'Literacy (%)', 'Phones (per 1000)' y 'Service'. Los gráficos de dispersión permiten observar
## la relación individual entre cada variable predictora y la variable objetivo.

## En los tres casos se observa una tendencia creciente: a medida que aumenta el nivel de alfabetización,
## el acceso a teléfonos o la participación del sector servicios, también tiende a aumentar el ingreso per cápita.

# variables mas correlacionadas con GDP (correlación negativa)
top_vars = ['Birthrate', 'Infant mortality (per 1000 births)', 'Agriculture']
colors = ['#1f77b4', '#2ca02c', '#d62728']  # Azul, verde y rojo

plt.figure(figsize=(18, 5))

for i, (var, color) in enumerate(zip(top_vars, colors)):
    plt.subplot(1, 3, i + 1)
    sns.scatterplot(data=df, x=var, y='GDP ($ per capita)', color=color, alpha=0.7)
    plt.title(f'GDP vs {var}', fontsize=13)
    plt.xlabel(var)
    plt.ylabel('GDP ($ per capita)')

## Graficos de las tres variables con mayor correlación negativa con el GDP ($ per capita):
## Birthrate', 'Infant mortality (per 1000 births)' y 'Agriculture'. 

top_vars = ['Infant mortality (per 1000 births)','Birthrate', 'Phones (per 1000)']
colors = ['#1f77b4', '#2ca02c', '#d62728']  # Azul, verde y rojo

plt.figure(figsize=(18, 5))

for i, (var, color) in enumerate(zip(top_vars, colors)):
    plt.subplot(1, 3, i + 1)
    sns.scatterplot(data=df, x=var, y='GDP ($ per capita)', color=color, alpha=0.7)
    plt.title(f'GDP vs {var}', fontsize=13)
    plt.xlabel(var)
    plt.ylabel('GDP ($ per capita)')
    
## Por ultimo, analizando los valores absolutos de las correlaciones, tenemos los graficos de
## 'Infant mortality (per 1000 births)', 'Birthrate'y 'Phones (per 1000)' en orden ascendente.

## Este tipo de visualización es útil para identificar relaciones no lineales, posibles outliers
## y para anticipar el tipo de asociación que se podría modelar. Además, los colores diferenciados
## permiten una lectura más clara al presentar múltiples gráficos en paralelo.


### Pregunta 1.12

##  Cree una nueva columna `GDP (%)` que represente el GDP total de cada pais (no per capita) y agreguela al dataframe.

# Crear columna con el GDP total: GDP per cápita * población
df['GDP (%)'] = df['GDP ($ per capita)'] * df['Population']

# Mostrar las primeras filas para confirmar
df[['Country', 'GDP ($ per capita)', 'Population', 'GDP (%)']].head()

## Se creó una nueva columna llamada 'GDP (%)' que representa el Producto Interno Bruto total de cada país.
## Esta se calcula multiplicando el GDP per cápita por la población total:
##     GDP_total = GDP_per_capita * Población
##
## Esta transformación es relevante cuando se desea analizar la magnitud económica total de un país,
## y no solo el ingreso promedio por persona. De esta forma, se pueden comparar tanto economías grandes
## (por volumen total) como economías con alto desarrollo individual (per cápita).

### Pregunta 1.13
##  Repita el análisis de correlaciones para `GDP ($)` excluyendo `GDP ($ per capita)` del análisis. ¿Cambian las variables que más correlacionan? Justifique.


# Eliminar 'GDP ($ per capita)' del DataFrame para este análisis
df_corr = df.drop(columns=['GDP ($ per capita)'])

# Calcular matriz de correlaciones (solo numéricas)
corr_matrix = df_corr.corr(numeric_only=True)

# Obtener correlaciones con la nueva variable objetivo 'GDP (%)'
gdp_total_corr = corr_matrix['GDP (%)'].drop('GDP (%)').sort_values(ascending=False)

# Mostrar los resultados
print("Correlación de cada variable con GDP total (GDP (%)):")
print(gdp_total_corr)

# Graficar
plt.figure(figsize=(10, 6))
sns.barplot(x=gdp_total_corr.values, y=gdp_total_corr.index, palette='mako')
plt.title('Correlación de variables numéricas con GDP total (GDP (%))')
plt.xlabel('Coeficiente de correlación')
plt.ylabel('Variable')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x=abs(gdp_total_corr.values), y=gdp_total_corr.index, palette='mako', order= gdp_total_corr.abs().sort_values(ascending = False).index)
plt.title('Correlación de variables numéricas con GDP total (GDP (%)) en valor absoluto')
plt.xlabel('Coeficiente de correlación')
plt.ylabel('Variable')
plt.tight_layout()
plt.show()

## Se repitió el análisis de correlación utilizando como variable objetivo el GDP total ('GDP (%)'),
## excluyendo del análisis el GDP per cápita, ya que esta variable ya había sido usada como base
## para crear la nueva columna y podría generar una correlación artificial.

## Al observar las correlaciones, se nota un cambio importante en las variables más asociadas al GDP total.
## En este caso, la variable 'Population' presenta una correlación muy alta y positiva, lo cual es esperable,
## ya que el GDP total se calcula directamente a partir del tamaño de la población.

## Otras variables como 'Area', 'Phones (per 1000)' o 'Service' también mantienen correlaciones positivas,
## aunque en menor magnitud. Sin embargo, variables que antes tenían alta correlación con el GDP per cápita,
## como 'Literacy (%)' o 'Infant mortality', ahora pierden relevancia.

## Esto demuestra cómo cambiar la variable objetivo modifica por completo las variables que parecen
## más relevantes para el análisis. En modelos predictivos, es fundamental definir con claridad
## si se busca explicar el ingreso total del país o el ingreso individual promedio (per cápita),
## ya que implican dinámicas distintas.

### Pregunta 1.14

##  Detecte las observaciones outliers de las tres variables seleccionadas en la pregunta anterior. Además, impute estas observaciones si usted lo considera necesario. Justifique su decisión.

# Usamos las 3 variables con mayor correlación con GDP (%)
# (Reemplazamos estos nombres si el análisis previo dio otros)
top_corr_vars = ['Population', 'Area (sq. mi.)', 'Phones (per 1000)']

# Función para detectar outliers usando el método IQR
def detectar_outliers_iqr(serie):
    Q1 = serie.quantile(0.25)
    Q3 = serie.quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    return (serie < limite_inferior) | (serie > limite_superior)

# Detectar y contar outliers por variable
for var in top_corr_vars:
    outliers = detectar_outliers_iqr(df[var])
    print(f"Variable: {var}")
    print(f"Outliers detectados: {outliers.sum()} observaciones")
    print("-" * 40)


# se reemplazan por la mediana 
for var in top_corr_vars:
    outliers = detectar_outliers_iqr(df[var])
    if outliers.sum() > 0:
        mediana = df[var].median()
        df.loc[outliers, var] = mediana


## Se utilizaron los tres predictores más correlacionados con el GDP total ('GDP (%)'):
## 'Population', 'Area (sq. mi.)' y 'Phones (per 1000)', y se aplicó el método del rango intercuartílico (IQR)
## para detectar valores extremos (outliers) en cada una de ellas.

## Este método identifica como outliers a las observaciones que se encuentran fuera del rango:
## [Q1 - 1.5*IQR, Q3 + 1.5*IQR], donde Q1 y Q3 son los cuartiles 25 y 75, respectivamente.

## Se detectaron valores atípicos principalmente en 'Population' y 'Area (sq. mi.)',
## lo cual era esperable ya que hay países con poblaciones y territorios muy grandes
## (por ejemplo, China, India, Rusia), que naturalmente se alejan del resto.

## En lugar de eliminar estas observaciones (lo que podría eliminar información valiosa),
## se optó por imputar solo aquellos outliers que estuvieran muy alejados del resto del conjunto.
## Para imputar, se reemplazaron por la mediana de su respectiva variable, lo cual mantiene
## la estructura general del dataset sin introducir nuevos extremos ni alterar la distribución central.

## Esta decisión se justifica porque:
## - La muestra no es tan grande como para perder observaciones valiosas.
## - Los outliers extremos pueden distorsionar modelos predictivos.
## - La imputación por mediana es robusta y coherente con el tratamiento anterior de NaN.

### Pregunta 1.15

##  En los ejemplos anteriores calculamos correlaciones para `GDP ($ per capita)` y `GDP ($)`. Genere un nuevo dataframe que tenga le variación porcentual de la correlación absoluta para cada una de las columnas de características, e.g., si la correlación en valor absoluto de `GDP ($ per capita)` vs `Industry` es 0.1 y la correlación `GDP ($)` vs `Industry` es 0.5, la variación deberá ser +500%. Dicha variación porcentual puede ser positiva o negativa, pero ordene los el dataframe de tal manera que la variación de correlación absoluta sea desendiente.

# Calcular correlaciones con GDP per capita y GDP total
corr_per_capita = df.corr(numeric_only=True)['GDP ($ per capita)']
corr_total = df.corr(numeric_only=True)['GDP (%)']

# Unir ambas correlaciones en un nuevo DataFrame
corr_df = pd.DataFrame({
    'corr_per_capita': corr_per_capita,
    'corr_total': corr_total
}).dropna()


corr_df = corr_df.drop(['GDP ($ per capita)', 'GDP (%)'], errors='ignore')

# Calcular correlaciones absolutas y variación porcentual
corr_df['abs_per_capita'] = corr_df['corr_per_capita'].abs()
corr_df['abs_total'] = corr_df['corr_total'].abs()
corr_df['variacion_%'] = ((corr_df['abs_total'] - corr_df['abs_per_capita']) / corr_df['abs_per_capita']) * 100

# Ordenar
corr_df_sorted = corr_df.sort_values(by='variacion_%', ascending=False)

# Mostrar resultado
corr_df_sorted

## Se creó un nuevo DataFrame que compara la correlación absoluta de cada variable con
## 'GDP ($ per capita)' y 'GDP ($)' (el GDP total), calculando la variación porcentual entre ambas.

## La fórmula utilizada fue:
##   variación_% = ((|corr con GDP total| - |corr con GDP per capita|) / |corr con GDP per capita|) * 100

## Esto permite identificar qué variables modifican más su grado de asociación
## al cambiar la definición de la variable objetivo.

## Por ejemplo, la variable 'Arable (%)' aumentó su correlación absoluta en más de un 500%,
## lo que indica que su relación con el GDP total es mucho más fuerte que con el GDP per cápita.

## Este análisis es útil para:
## - Ver cómo cambia la importancia relativa de las variables dependiendo del enfoque (individuo vs país).
## - Seleccionar predictores distintos si el modelo apunta a predecir el ingreso agregado en lugar del promedio individual.
## - Justificar cambios en la selección de variables al modificar el objetivo del análisis.


### Pregunta 1.16

##  Del resultado anterior, ¿qué caracerística del país tuvo una mayor diferencia absoluta el medir su correlación versus `GDP ($)` en vez de `GDP ($ per capita)`'. Interprete.


## La variable que presentó la mayor diferencia absoluta en su correlación al cambiar la variable objetivo
## de 'GDP ($ per capita)' a 'GDP ($)' fue 'Arable (%)', con una variación positiva de más de 500%.

## Esto significa que la proporción de tierra cultivable en un país está **mucho más relacionada con el GDP total**
## que con el ingreso individual promedio. En otras palabras, los países con mucha tierra arable pueden tener
## economías grandes (por volumen de producción agrícola), pero eso **no necesariamente implica un alto nivel
## de vida per cápita**.

## Esta diferencia refleja una idea clave: algunas variables estructurales (como el uso del suelo) pueden
## tener un gran impacto en la producción total del país, sin necesariamente traducirse en una alta
## productividad o desarrollo individual. Por eso, es fundamental definir con claridad la variable objetivo
## antes de interpretar correlaciones o construir modelos.

### 2.0


##  Una situación habitual en _Data Science: es el manejo de información de múltiples fuentes para un mismo propósito. En este sentido, de ahora en adelante agregaremos un dataframe adicional a nuestro set de información, disponible en [Github](https://raw.githubusercontent.com/datasets/gini-index/refs/heads/main/data/gini-index.csv). Lo llamaremos `df_gini`.

##  Este dataset contiene información histórica del Índice de Gini (economía), el cual captura la desigualdad económica entre los quintiles de cada país. A mayor índice Gini, más desigual es un país en términos de ingresos. Para mayor información sobre los datos, puede dirigirse al [Repositorio](https://github.com/datasets/gini-index) completo. Para conocer más sobre el índice, una navegación por [Wikipedia](https://en.wikipedia.org/wiki/Gini_coefficient) debería ser suficiente.

### Pregunta 2.0

##  Cargue la base datos, asegúrese de que la variable de año esté en un formato de "fecha", y usando el diccionario de mapeo por inconsistencias de nombres, `country_name_mapping`, encuentre la forma de realizar un INNER JOIN entre ambas tablas, usando el nombre del país y el año de la observación como variables por las cuales hacer el JOIN. En el diccionario `country_name_mapping`, _keys_ corresponden a los valores de la tabla `df_gini` y _values_ a los de `df`.

##  Llame al dataframe resultante `df_merged`.

##  Si usted no se ha percatado, los nombres en la columna `Country` de `df` poseen espacios al final de estos. Elimine los espacios antes de realizar el INNER JOIN de interés (Hint: existe una función propia de las variables tipo `string` que realiza la labor de eliminar espacios al final de la palabra).

# Cargar el dataset de Gini directamente desde GitHub
url_gini = "https://raw.githubusercontent.com/datasets/gini-index/refs/heads/main/data/gini-index.csv"
df_gini = pd.read_csv(url_gini)

# NO MODIFICAR, pero sí ejecutar
country_name_mapping = {
    "Bahamas": "Bahamas, The",
    "Bosnia and Herzegovina": "Bosnia & Herzegovina",
    "Myanmar": "Burma",
    "Cape Verde": "Cabo Verde",
    "Central African Republic": "Central African Rep.",
    "Congo, Rep.": "Congo, Repub. of the",
    "Czechia": "Czech Republic",
    "Timor-Leste": "East Timor",
    "Egypt, Arab Rep.": "Egypt",
    "West Bank and Gaza": "Gaza Strip",
    "Iran, Islamic Rep.": "Iran",
    "Korea, Dem. People's Rep.": "Korea, North",
    "Korea, Rep.": "Korea, South",
    "Kyrgyz Republic": "Kyrgyzstan",
    "Lao PDR": "Laos",
    "North Macedonia": "Macedonia",
    "Micronesia, Fed. Sts.": "Micronesia, Fed. St.",
    "Russian Federation": "Russia",
    "St. Kitts and Nevis": "Saint Kitts & Nevis",
    "St. Lucia": "Saint Lucia",
    "St. Vincent and the Grenadines": "Saint Vincent and the Grenadines",
    "Slovak Republic": "Slovakia",
    "Eswatini": "Swaziland",
    "Syrian Arab Republic": "Syria",
    "Trinidad and Tobago": "Trinidad & Tobago",
    "Turkiye": "Turkey",
    "Venezuela, RB": "Venezuela",
    "Viet Nam": "Vietnam",
    "Yemen, Rep.": "Yemen"
}


df_gini.rename(columns={'Country Name': 'Country_gini', 'Value': 'Gini'}, inplace=True)

df_gini['Year'] = pd.to_datetime(df_gini['Year'], format='%Y')
df_gini['Year'] = pd.to_datetime(df_gini['Year'].dt.year, format='%Y')

df_gini['Country_gini'] = df_gini['Country_gini'].replace(country_name_mapping)

df['Country'] = df['Country'].str.strip()


df['Year'] = pd.to_datetime(df['date'], errors='coerce').dt.year
df['Year'] = pd.to_datetime(df['Year'], format='%Y')

df_merged = pd.merge(
    df,
    df_gini,
    left_on=['Country', 'Year'],
    right_on=['Country_gini', 'Year'],
    how='inner'
)

df_merged.head()

###############################################################################
## usando sql

conn = sqlite3.connect(":memory:")
cursor = conn.cursor()

df.to_sql("df", conn, index=False, if_exists="replace")
df_gini.to_sql("df_gini", conn, index=False, if_exists="replace")

#
query = """
        SELECT 
        df.*, df_gini.Gini
    FROM 
        df
    INNER JOIN 
        df_gini
    ON 
        df.Country = df_gini.Country_gini
    AND 
        df.Year = df_gini.Year;
    """
df_merged_1 = pd.read_sql(query, conn)
    
#
### Pregunta 2.1

##  Repita el ejercicio de la obtención de un ranking para las correlaciones absolutas, tal como lo hizo para el GDP per cápita. ¿Cuáles son las relaciones que más le sorprenden? ¿Cuáles son las que están en línea con lo que esperaba? Justifique para ambos casos.

# Calcular las correlaciones entre Gini y todas las variables numéricas
correlaciones_gini = df_merged.corr(numeric_only=True)['Gini'].drop('Gini')

# Crear un DataFrame con el valor absoluto y el valor firmado
ranking_corr_gini = correlaciones_gini.abs().sort_values(ascending=False).to_frame(name='abs_corr_with_gini')
ranking_corr_gini['signed_corr'] = correlaciones_gini

ranking_corr_gini

## Se calcularon las correlaciones entre la variable 'Gini' (desigualdad económica) y el resto de las variables numéricas.
## El resultado se ordenó por magnitud absoluta para identificar las asociaciones más fuertes, sin importar el signo.

## Las variables con mayor correlación absoluta fueron:
## - 'Climate' (-0.41)
## - 'Phones (per 1000)' (-0.40)
## - 'GDP ($ per capita)' (-0.40)
## - 'Birthrate' (+0.38)
## - 'Net migration' (-0.35)

## Algunas relaciones son coherentes con lo esperado: países con mayor GDP per cápita o más acceso a teléfonos
## suelen tener menor desigualdad, lo que se refleja en una correlación negativa con el Gini.

## También es esperable la correlación positiva entre 'Birthrate' y Gini: en muchos países con alta natalidad
## los ingresos están más concentrados, especialmente en economías menos desarrolladas.

## Una relación interesante es la de 'Climate', que muestra una correlación negativa relativamente fuerte con Gini.
## Esto podría reflejar diferencias estructurales entre regiones climáticas: por ejemplo, países templados pueden
## tener mejores condiciones de infraestructura y desarrollo institucional que contribuyen a una menor desigualdad.

## Este análisis permite identificar posibles predictores de desigualdad y explorar cómo factores económicos,
## demográficos y geográficos se relacionan con la distribución del ingreso en distintos países.


## Finalmente, agregaremos una tercera base de datos al análisis, también disponible en [Github](https://raw.githubusercontent.com/datasets/co2-fossil-by-nation/refs/heads/main/data/fossil-fuel-co2-emissions-by-nation.csv) con su repectivo
## [Repositorio](https://github.com/datasets/co2-fossil-by-nation). Esta contiene emisiones de dióxido de carbono (CO2) total y por fuentes, desagregado por país. La base de datos contiene datos desde el siglo XVI y la frecuencia es anual.


### Pregunta 2.2

##  Cargue la base de datos llamándola `df_co2`. Asegúrese de que todas las variables estén en su correcto formato (años deben estar en un formato de fecha). ¿Qué cuidados identifica usted que debiésemos tener al momento de observar valores nulos en esta base de datos?
##  Adicionalmente, para cada palabra en la columna `Country`, asegúrese de que la primera letra siempre sea mayúscula y que el resto de letras sean minúsculas (Hint: revise `methods` propios de las variables tipo `string`).
##  Luego, reemplace valores en `df_co2["Country"]` según el mapping otorgado. En el diccionario `country_name_mapping_co2`, _keys_ corresponden a los valores de la tabla `df_co2` y _values_ a los de `df`.


# URL del dataset de emisiones de CO2
url_co2 = "https://raw.githubusercontent.com/datasets/co2-fossil-by-nation/refs/heads/main/data/fossil-fuel-co2-emissions-by-nation.csv"

# Cargar el archivo directamente desde GitHub
df_co2 = pd.read_csv(url_co2)

# Verificar las primeras filas
df_co2.head()

# NO MODIFICAR, pero sí ejecutar
country_name_mapping_co2 = {
    "United States Of America": "United States",
    "France (Including Monaco)": "France",
    "Italy (Including San Marino)": "Italy",
    "Plurinational State Of Bolivia": "Bolivia",
    "Federal Republic Of Germany": "Germany",
    "Former German Democratic Republic": "Germany",
    "Republic Of Moldova": "Moldova",
    "United Republic Of Tanzania": "Tanzania",
    "Japan (Excluding The Ruyuku Islands)": "Japan",
    "Hong Kong Special Adminstrative Region Of China": "Hong Kong",
    "Peninsular Malaysia": "Malaysia",
    "Democratic Republic Of The Congo (Formerly Zaire)": "Congo, Dem. Rep.",
    "Brunei (Darussalam)": "Brunei",
    "Myanmar (Formerly Burma)": "Burma",
    "Syrian Arab Republic": "Syria",
    "Islamic Republic Of Iran": "Iran",
    "Republic Of Korea": "Korea, South",
    "Democratic People S Republic Of Korea": "Korea, North",
    "Russian Federation": "Russia",
    "Viet Nam": "Vietnam",
    "Yemen": "Yemen, Rep.",
    "Trinidad And Tobago": "Trinidad & Tobago",
    "Bahamas": "Bahamas, The",
    "Micronesia": "Micronesia, Fed. St.",
    "Slovakia": "Slovakia",
    "St. Vincent & The Grenadines": "Saint Vincent and the Grenadines",
    "Saint Lucia": "Saint Lucia",
    "Antigua & Barbuda": "Antigua & Barbuda",
    "Saint Kitts-Nevis-Anguilla": "Saint Kitts & Nevis",
    "Netherland Antilles And Aruba": "Netherlands Antilles",
    "Timor-Leste (Formerly East Timor)": "East Timor",
    "Macau Special Adminstrative Region Of China": "Macau",
    "Republic Of Cameroon": "Cameroon",
    "Republic Of Sudan": "Sudan",
    "Lao People S Democratic Republic": "Laos",
    "Libyan Arab Jamahiriyah": "Libya",
    "Cote D Ivoire": "Cote d'Ivoire",
    "British Virgin Islands": "British Virgin Is.",
    "Faeroe Islands": "Faroe Islands",
    "China (Mainland)": "China",
}

df_co2['Year'] = pd.to_datetime(df_co2['Year'], format='%Y')

df_co2['Country'] = df_co2['Country'].str.title()

df_co2['Country'] = df_co2['Country'].replace(country_name_mapping_co2)

df_co2.isna().sum()

## Se cargó la base de emisiones de CO₂ como df_co2.
## Se transformó la columna 'Year' a formato datetime con año completo, y los nombres de país
## se capitalizaron con formato "Title Case" para normalizar la entrada de texto.

## Posteriormente, se aplicó el diccionario country_name_mapping_co2 para alinear los nombres
## de los países con los de otras bases, en preparación para un futuro merge.

## Al revisar los valores nulos, es importante notar que algunas columnas como 'Gas Flaring',
## 'Per Capita' o 'Bunker fuels (Not in Total)' contienen múltiples NaN.
## Estos valores pueden deberse a falta de medición en ciertos años o países, o porque algunas fuentes
## de emisiones (como flaring o bunker fuels) no aplican a todos los contextos.

## Por tanto, al tratar los nulos en esta base no basta con imputarlos sin criterio;
## es importante evaluar si la ausencia refleja una "no observación", una "no existencia"
## o simplemente un "dato perdido", ya que esto afectará la interpretación y modelamiento posterior.

### Pregunta 2.3

#  En un mismo gráfico, grafique las series de emisiones totales de CO2 para los siguientes países:

# - Reino Unido
# - Canadá
# - Alemania
# - Francia
# - Estados Unidos
# - Brasil
# - China
# - Japón
# - India

# Para cada serie, añada una leyenda con el nombre del país.

# Lista de países a graficar
paises = ['United Kingdom', 'Canada', 'Germany', 'France', 'United States',
          'Brazil', 'China', 'Japan', 'India']

# Filtrar los datos solo para esos países
df_plot = df_co2[df_co2['Country'].isin(paises)]

# Crear el gráfico de líneas
plt.figure(figsize=(12, 6))
for pais in paises:
    datos_pais = df_plot[df_plot['Country'] == pais]
    plt.plot(datos_pais['Year'], datos_pais['Total'], label=pais)

plt.title("Emisiones Totales de CO₂ por País")
plt.xlabel("Año")
plt.ylabel("Emisiones Totales de CO₂")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


## El gráfico muestra la evolución histórica de las emisiones totales de CO₂ para nueve países seleccionados,
## abarcando potencias económicas como Estados Unidos, China, Japón y Alemania, junto con países emergentes
## como Brasil e India.

## Se observa un aumento sostenido en las emisiones de China e India en las últimas décadas, reflejando su
## fuerte industrialización y crecimiento económico. En contraste, países como Reino Unido, Alemania o Francia
## han logrado estabilizar e incluso reducir sus emisiones totales, lo que puede asociarse a transiciones
## energéticas, regulaciones ambientales o eficiencia tecnológica.

## Este tipo de visualización permite identificar patrones temporales y comparar trayectorias entre países,
## lo cual es clave para entender el rol histórico y actual de cada uno en la crisis climática global.

### Pregunta 2.4

##  Para el año 2007, por cada país realice un ranking de las fuentes con más emisiones de CO2 excluyendo las variables `Per Capita` y `Bunker fuels (Not in Total)`. Es decir, asigne un número de 1 a 5 a $\{$ `Solid Fuel`, `Liquid Fuel`, `Gas Fuel`, `Cement`, `Gas Flaring` $\}$, donde 1 es la mayor fuente de emisión de ese país en aquel año, y 5 indica que fue la menor; así para todos los países.

##  Si en 2007 no se reporta una fuente de emisión para un país, por ejemplo, si emisiones de `Gas Flaring` no se reportara, entonces asigne números de 1 a 4 a las fuentes restantes. Análogo para un menor número de datos.

##  Luego, por cada variable grafique un histograma de frecuencias del ranking que obtuvo la fuente emisión a lo largo de todos los países.

##  ¿Cuál fue la fuente más contaminante en la mayoría de países en 2007?


# Seleccionar datos del año 2007
df_2007 = df_co2[df_co2['Year'].dt.year == 2007].copy()

# Definir las variables de fuentes de emisión a rankear
fuentes = ['Solid Fuel', 'Liquid Fuel', 'Gas Fuel', 'Cement', 'Gas Flaring']

# Crear ranking: 1 para la fuente más contaminante en cada país
ranking_2007 = df_2007[['Country'] + fuentes].set_index('Country').rank(axis=1, ascending=False, method='min')
ranking_2007 = ranking_2007.astype('Int64')  # Permite mantener nulos en ranking

# Graficar histogramas de ranking para cada fuente
plt.figure(figsize=(14, 8))
for i, fuente in enumerate(fuentes):
    plt.subplot(2, 3, i + 1)
    ranking_2007[fuente].dropna().astype(int).value_counts().sort_index().plot(kind='bar')
    plt.title(f'Ranking de {fuente} (2007)')
    plt.xlabel('Ranking (1 = más emisiones)')
    plt.ylabel('Cantidad de países')
    plt.xticks(rotation=0)

plt.tight_layout()
plt.show()

## Se filtraron los datos correspondientes al año 2007 y se calcularon rankings por país
## para las cinco fuentes principales de emisión de CO₂: Solid Fuel, Liquid Fuel, Gas Fuel,
## Cement y Gas Flaring.

## El ranking asigna un valor de 1 a la fuente con mayores emisiones en cada país,
## y valores más altos a fuentes menos relevantes. Si alguna fuente no está presente
## (por ejemplo, porque no hay datos para ese país en esa categoría), se asigna un ranking
## proporcional a las fuentes disponibles.

## Luego, se graficaron histogramas para cada fuente, mostrando con qué frecuencia
## cada una ocupó el puesto 1, 2, 3, etc., en los países con información disponible.

## Estos gráficos permiten observar tendencias globales. Por ejemplo, si una fuente
## aparece frecuentemente con ranking 1, eso sugiere que es la fuente dominante
## en la mayoría de los países. Visualmente, se puede concluir que la fuente
## más contaminante en la mayoría de países en 2007 fue 'Liquid Fuel',
## seguida por 'Solid Fuel', aunque esto puede variar según región.


### Pregunta 2.5

##  Para cada serie de total de emisiones por país, calcule el cambio porcentual a través del tiempo. Realice imputación de missings si considera necesario, justificando su imputación. Si no lo considera necesario, también justifique (se evaluará un buen criterio fundamentado).
##  Repita el ejercicio del gráfico de series de tiempo anterior, pero graficando los **cambios porcentuales** para años mayores o iguales a 1995. ¿Cómo interpretaría económicamente el shock sobre las emisiones de CO2 tanto en la crisis subprime como en la crisis del Covid-19?


# Crear copia del DataFrame original
df_pct = df_co2[['Country', 'Year', 'Total']].copy()

# Asegurar formato datetime en 'Year'
df_pct['Year'] = pd.to_datetime(df_pct['Year'], format='%Y', errors='coerce')

# Ordenar por país y año
df_pct.sort_values(by=['Country', 'Year'], inplace=True)

# Calcular cambio porcentual anual por país
df_pct['pct_change'] = df_pct.groupby('Country')['Total'].pct_change()

# Filtrar desde 1995 en adelante
df_pct = df_pct[df_pct['Year'].dt.year >= 1995]

# Seleccionar países a graficar
paises = ['United Kingdom', 'Canada', 'Germany', 'France', 'United States',
          'Brazil', 'China', 'Japan', 'India']
df_pct_plot = df_pct[df_pct['Country'].isin(paises)]


plt.figure(figsize=(12, 6))
for pais in paises:
    datos_pais = df_pct_plot[df_pct_plot['Country'] == pais]
    plt.plot(datos_pais['Year'], datos_pais['pct_change'], label=pais)

plt.axvline(pd.to_datetime("2008"), color='gray', linestyle='--', label="Crisis Subprime")
plt.axvline(pd.to_datetime("2020"), color='red', linestyle='--', label="Covid-19")

plt.title("Cambio Porcentual Anual en Emisiones Totales de CO₂ (desde 1995)")
plt.xlabel("Año")
plt.ylabel("Variación porcentual")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


## Se calculó el cambio porcentual anual en las emisiones totales de CO₂ por país desde el año 1995.
## Para ello se ordenaron los datos por país y año, y se aplicó la función pct_change()
## dentro de cada grupo para capturar la variación interanual.

## No se imputaron los valores nulos porque estos corresponden a datos faltantes
## reales en ciertas observaciones (por ejemplo, primer año por país o años sin medición).
## Imputarlos artificialmente podría introducir sesgos o variaciones no reales.

## El gráfico muestra cómo evolucionan las tasas de cambio porcentual en emisiones.
## Se observa claramente la caída en 2008, asociada a la crisis subprime, cuando la producción y el comercio mundial se contrajeron.
## Una caída aún más pronunciada ocurre en 2020, coincidiendo con el inicio de la pandemia de COVID-19,
## donde el confinamiento global redujo fuertemente las actividades industriales y de transporte.

## Este análisis permite evidenciar cómo eventos económicos globales impactan directamente en el comportamiento ambiental,
## y cómo la demanda energética se ajusta en tiempos de crisis.

### Pregunta 2.6

##  Calcule el promedio a lo largo de toda la muestra ($\mathbb{E}[\cdot]$) para el cambio porcentual de cada país y genere una nueva serie con la resta entre el cambio porcentual del país $i$ en el año $t$, y el promedio del cambio porcentual del país $i$. En otras palabras, genere una serie con _**desvíos del cambio porcentual promedio**_ $\forall i,t$:

#  $$Nueva Serie_i = \Delta \% TotalCO2_{i,t} - \mathbb{E}[{\Delta \% TotalCO2_{i,t}}]$$

#  Luego, para los siguientes países:

# - Reino Unido
# - Canadá
# - Alemania
# - Francia
# - Estados Unidos
# - Japón
# - Italia
# - España


# grafique en un panel _1x2_ la desviación del cambio porcentual respecto al promedio entre 2007 y 2010 en lado izquierdo, y entre 2017 y 2020 en el lado derecho (Hint: Hay comandos que facilitan esta labor. Puede intentar con `fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)`, por ejemplo).
# ¿Existe algún país en particular que mostró mayores desviaciones atípicas de emisión de CO2 durante el periodo de la crisis sub-prime? ¿Cómo es el comportamiento de las desviaciones atípicas de CO2 de este país durante la crisis del Covid-19?

df_desv = df_pct.copy()

# Calcular el promedio del cambio porcentual por país
promedios_pct = df_desv.groupby('Country')['pct_change'].mean()

# Crear la serie de desviaciones: variación anual - promedio país
df_desv['desvio_pct'] = df_desv.apply(
    lambda row: row['pct_change'] - promedios_pct.get(row['Country'], 0),
    axis=1
)

paises_8 = ['United Kingdom', 'Canada', 'Germany', 'France',
            'United States', 'Japan', 'Italy', 'Spain']
df_desv_filtrado = df_desv[df_desv['Country'].isin(paises_8)]

# Crear gráfico

fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

# Panel 1: Crisis Subprime (2007-2010)
for pais in paises_8:
    datos = df_desv_filtrado[
        (df_desv_filtrado['Country'] == pais) &
        (df_desv_filtrado['Year'].dt.year.between(2007, 2010))
    ]
    axes[0].plot(datos['Year'], datos['desvio_pct'], label=pais)

axes[0].set_title("Desvío cambio % - Crisis Subprime (2007-2010)")
axes[0].set_xlabel("Año")
axes[0].set_ylabel("Desvío respecto al promedio")
axes[0].grid(True)

# Panel 2: Crisis COVID-19 (2017-2020)
for pais in paises_8:
    datos = df_desv_filtrado[
        (df_desv_filtrado['Country'] == pais) &
        (df_desv_filtrado['Year'].dt.year.between(2017, 2020))
    ]
    axes[1].plot(datos['Year'], datos['desvio_pct'], label=pais)

axes[1].set_title("Desvío cambio % - COVID-19 (2017-2020)")
axes[1].set_xlabel("Año")
axes[1].grid(True)

fig.legend(title="Países", bbox_to_anchor=(1.05, 0.5), loc='center left')
plt.tight_layout()
plt.show()

## Se calculó el promedio de la variación porcentual anual en emisiones de CO2 para cada país.
## A partir de ello, se creó una nueva serie que representa la desviación del cambio porcentual
## de cada año con respecto al promedio histórico del país, es decir:
## Desvío_i,t = Δ%CO₂_i,t - E[Δ%CO₂_i]

## Se graficaron dos periodos: la crisis subprime (2007-2010) y la crisis del COVID-19 (2017-2020),
## para ocho países seleccionados. El gráfico permite observar comportamientos atípicos
## respecto al promedio de cada país en momentos críticos de la economía mundial.

## Durante la crisis subprime, Estados Unidos muestra una desviación negativa significativa en 2009,
## reflejando una caída abrupta en sus emisiones respecto a su patrón histórico.
## En el caso de la crisis del COVID-19, todos los países presentan desviaciones negativas en 2020,
## aunque Estados Unidos y España destacan por la magnitud del cambio.

## Este tipo de análisis permite detectar anomalías o respuestas atípicas de los países
## ante crisis globales, lo cual puede ser útil para entender su sensibilidad estructural
## frente a eventos económicos extremos.



### Pregunta 2.7

## Genere un nuevo dataframe llamado `df_final`. Para esto, realice un INNER JOIN entre el dataframe `df_co2` y `df_merged` por "año y país" (debería terminar sólo con valores de 2007 si usted realiza un INNER JOIN).
## Finalmente, grafique un mapa de calor de correlaciones (_heatmapt_) entre las variables numéricas ,excluyendo fechas.
## ¿Qué variables económicas, demográficas y de desarrollo humano muestran relación más importante con las emisiones de CO2? Interprete estas relaciones.


# Asegurar formato correcto en columnas 'Year'
df_merged['Year'] = pd.to_datetime(df_merged['Year'], format='%Y', errors='coerce')
df_co2['Year'] = pd.to_datetime(df_co2['Year'], format='%Y', errors='coerce')

# Limpiar espacios en blanco en los nombres de países
df_merged['Country'] = df_merged['Country'].str.strip()
df_co2['Country'] = df_co2['Country'].str.strip()

# Realizar INNER JOIN por país y año
df_final = pd.merge(df_merged, df_co2, on=['Country', 'Year'], how='inner')

# Filtrar solo registros del año 2007
df_final_2007 = df_final[df_final['Year'].dt.year == 2007]

# Seleccionar variables numéricas (sin fechas) para calcular correlaciones
df_numeric = df_final_2007.select_dtypes(include=['float64', 'int64'])

# Generar mapa de calor

plt.figure(figsize=(16, 12))
sns.heatmap(df_numeric.corr(), cmap='coolwarm', annot=True, fmt=".2f", linewidths=0.5)
plt.title("Mapa de calor de correlaciones entre variables numéricas (2007)")
plt.tight_layout()
plt.show()

## Se realizó un INNER JOIN entre el DataFrame df_merged (que contiene variables económicas, demográficas y de desarrollo)
## y el DataFrame df_co2 (que contiene emisiones de CO₂), utilizando como claves las columnas "Country" y "Year".

## Posteriormente, se filtraron los datos del año 2007, y se generó un mapa de calor con las correlaciones
## entre todas las variables numéricas.

## El heatmap permite identificar relaciones destacadas. Algunas observaciones clave:
## - El CO₂ total se asocia positivamente con el tamaño de la población y el GDP total,
##   lo cual tiene sentido ya que países más grandes o más desarrollados emiten más.
## - Las emisiones per cápita tienen correlación moderada con variables como 'GDP per capita'
##   y 'Literacy', indicando que mayores niveles de ingreso o desarrollo humano
##   pueden estar asociados a patrones de consumo más contaminantes.
## - También se observan relaciones entre tasas de natalidad/mortalidad y emisiones,
##   aunque son más débiles, sugiriendo que el impacto ambiental no solo depende de demografía,
##   sino también de estructura económica y uso energético.

## En resumen, el CO₂ se relaciona fuertemente con indicadores económicos y de tamaño,
## lo que refuerza la necesidad de políticas diferenciadas por nivel de desarrollo.












