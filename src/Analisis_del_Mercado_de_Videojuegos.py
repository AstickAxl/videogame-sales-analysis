#!/usr/bin/env python
# coding: utf-8

# Importamos librerias
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats


# Cargamos el archivo con el que trabajaremos
df = pd.read_csv('/datasets/games.csv')


# **Podemos observar los datos del archivo con el cual estaremos trabajando, observamos distintas columnas como nombre, plataforma año de lanzamiento, ventas por region, critica, entre otros. Asi como valores ausentes en algunas de nuestras columnas**

# In[33]:


# Preparando los datos

# Le daré formato snake_case a los nombres de todas las columnas
df.columns = df.columns.str.lower().str.replace(' ', '_')
df.info()
df.describe()


# **Vamos a comenzar a corregir algunos tipos de datos. Voy a tratar los valores 'tbd' que existen en la columna 'user_score' como valores ausentes para poder trabajar con la columna de manera numerica.**:

# In[34]:


# Convertimos 'tbd' en valores NaN y convertimos a tipo float
df['user_score'] = df['user_score'].replace('tbd', np.nan)
df['user_score'] = df['user_score'].astype(float)

# Convertiremos la columna year_of_release en Int64, asi como critic_score
df['year_of_release'] = df['year_of_release'].astype('Int64')
df['critic_score'] = df['critic_score'].astype('Int64')
df.info()


# **Haciendo un breve analisis de los valores ausentes y como los vamos a tratar he determinado que los valores nulos de la columna 'name' coinciden con la columna 'genre', por lo que considero oportuno eliminar estas filas ya que las ventas del videojuego son minimas (no afectaria en los analisis estadisticos), no tiene reseñas y tenemos datos incompletos, por lo tanto, el dato no sirve.
# Por otro lado, la columna 'year_of_release' tambien tiene valores nulos, probablemente los juegos son poco conocidos o hubo errores al obtener la informacion; por lo que prefiero dejarlos como NaN y no inventar datos.
# En las columnas critic_score, user_score y rating también tienen muchos valores nulos, puede que no todos los criticos o usuarios opinen sobre el juego, algunos no tienen clasificacion ESRB, por lo que prefiero dejar los valores ausentes como NaN y no sesgar los analisis.**

# In[35]:


# Eliminamos filas donde name y genre estan nulos
df.dropna(subset=['name', 'genre'], inplace=True)
df.info()


# In[36]:


# Procederemos a calcular las ventas totales para cada juego
df['total_sales'] = df['na_sales'] + \
    df['eu_sales'] + df['jp_sales'] + df['other_sales']
df.head()


# **Paso 3. Analizar los Datos**

# In[37]:


# Calculamos cuantos juegos se lanzaron en diferentes años
games_per_year = df['year_of_release'].value_counts(
).sort_index(ascending=False)


# In[38]:


# Graficando
plt.figure(figsize=(10, 5))
plt.bar(games_per_year.index, games_per_year.values)
plt.xlabel('Año de lanzamiento')
plt.ylabel('Numero de juegos lanzados')
plt.title('Cantidad de juegos lanzados por año')
plt.grid(axis='y')
plt.tight_layout()
plt.show()


# **Como podemos observar, en los 80's y mitades de los 90's, la produccion de videojuegos era baja, no pasaban de 100 videojuegos por año (incluyendo todas las plataformas), sin embargo, la tendenecia ha aumentado drasticamente a partir del año 1995, teniendo como resultado mas de mil videojuegos lanzados por año. Teniendo el año 2008 como el año con mas videojuegos lanzados con una cantidad de 1427.**
# **Para conseguir muestras representativas actuales voy a tomar unicamente los datos mayores al año 1995, que es donde podemos ver una notable tendencia en aumento para la industria de videojuegos.**
#

# In[39]:


# Aplicamos el filtro para mayores de 1995.
df_recent = df[df['year_of_release'] >= 1995]


# In[40]:


# Calculamos las ventas totales por plataforma a lo largo del tiempo
platform_year_sales = df_recent.groupby(['year_of_release', 'platform'])[
    'total_sales'].sum().reset_index()

# Filtramos las plataformas mas exitosas
# Agregamos .index al final ya que solo necesitamos los nombres de las plataformas.
top_platforms = df_recent.groupby('platform')['total_sales'].sum(
).sort_values(ascending=False).head(10).index
platform_year_sales_top = platform_year_sales[platform_year_sales['platform'].isin(
    top_platforms)]


# In[41]:


# Obtenemos el gráfico
plt.figure(figsize=(12, 6))
sns.lineplot(data=platform_year_sales_top, x='year_of_release',
             y='total_sales', hue='platform')
plt.title('Ventas totales por año y plataforma')
plt.xlabel('Año de lanzamiento')
plt.ylabel('Ventas globales [M]')
plt.grid(True)
plt.tight_layout()
plt.show()


# **Podemos observar con el grafico que se realizó el top 10 plataformas desde el año 1995 hasta el 2016. Este grafico nos será de gran ayuda para los siguientes pasos del analisis.**
#
# **Creare un filtro desde un año mas reciente (2010) para enfocarnos en tendencias "Actuales"**

# In[42]:


# Creamos un filtro desde 2010 en adelante
recent_years = df_recent[df_recent['year_of_release'] >= 2010]

# Agrupamos por año y plataforma
platform_trends = recent_years.groupby(['year_of_release', 'platform'])[
    'total_sales'].sum().reset_index()

# Plataformas que son frecuentes desde 2010
top_recent_platforms = (platform_trends.groupby('platform')[
                        'total_sales'].sum().sort_values(ascending=False).head(6).index)
platform_trends_top = platform_trends[platform_trends['platform'].isin(
    top_recent_platforms)]

# Obtenemos la grafica
plt.figure(figsize=(12, 6))
sns.lineplot(data=platform_trends_top, x='year_of_release',
             y='total_sales', hue='platform')
plt.title('Tendencia de ventar por plataforma (desde 2010)')
plt.xlabel('Año de lanzamiento')
plt.ylabel('Ventas globales [M]')
plt.grid(True)
plt.tight_layout()
plt.show()


# **Podemos observar que, aunque la mayoría de las plataformas muestran una tendencia a la baja en sus ventas, PS4 mantiene una ventaja significativa al ser la que registra las mayores ventas recientes.**
#
# **A pesar del descenso general, plataformas como PS4 y Xbox One (XOne) siguen generando ingresos considerables y pueden considerarse activas y relevantes para el análisis de 2017.**
#
# **Además, se observa que el ciclo de vida promedio de una plataforma es de aproximadamente 7 a 10 años, lo cual refuerza la viabilidad de PS4 y XOne para continuar en el mercado en el corto plazo.**

# In[43]:


# Filtramos los datos desde 2010
recent_years = df_recent[df_recent['year_of_release'] >= 2010]

# Agrupamos por plataformas mas importantes
active_platforms = ['PS4', 'XOne', '3DS']
recent_platforms_data = recent_years[recent_years['platform'].isin(
    active_platforms)]

# Creamos los diagramas de caja
plt.figure(figsize=(12, 6))
sns.boxplot(data=recent_platforms_data, x='platform', y='total_sales')
plt.title('Distribución de ventas por plataforma (2010-2016)')
plt.xlabel('Plataforma')
plt.ylabel('Ventas globales [M]')
plt.grid(True)
plt.show()


# **Analizando los resultados de nuestro boxplot, podemos observar que la mayoria de los juegos venden cifras muy similares, y que son generalmente bajas, por esta razon tenemos rangos intercuartilicos muy pequeños. Los puntos atipicos que tenemos, quiero pensar que son los videojuegos "super-ventas" o juegos que son muy famosos como GTA, FIFA, CoD, etc. PS4 ha tenido mas de estos ejemplos que las otras 2 plataformas**
#
# **PS4 y XOne tienen una distribucion muy similar, pero PS4 suele tener la media mas alta, por lo cual sugiere que tiene una mejor recepcion promedio. Y por otro lado tenemos a la plataforma 3DS, que si bien sigue siendo activa, como podemos ver tiene mucho menores ventas en comparacion con sus rivales.**

# In[44]:


# Analizaremos como las reseñas afectan las ventas
ps4_games = df_recent[df_recent['platform'] == 'PS4']
ps4_games[['critic_score', 'user_score', 'total_sales']].isna().sum()

# Graficamos User Score vs Total Sales
plt.figure(figsize=(12, 5))
sns.scatterplot(data=ps4_games, x='user_score', y='total_sales', alpha=0.5)
plt.title('PS4 Ventas vs Calificaciones de Usuarios')
plt.xlabel('Calificacion de Usuario')
plt.ylabel('Ventas Globales [M]')
plt.grid(True)
plt.show()

# Grafico de Critic Score vs Total Sales
plt.figure(figsize=(12, 5))
sns.scatterplot(data=ps4_games, x='critic_score', y='total_sales', alpha=0.5)
plt.title('PS4 Ventas vs Calificaciones Criticas')
plt.xlabel('Calificacion de Critica')
plt.ylabel('Ventas Globales [M]')
plt.grid(True)
plt.show()


# **Podemos observar en el grafico user_score vs total_sales una ligera tendencia donde si la critica es buena, las ventas usualmente tambien son buenas. Sin embargo los datos son muy dispersos para poder concluir que la calificacion del usuario tiene un impacto claroen las ventas.**
#
# **En el grafico critic_score vs total_sales podemos observar de igual forma, una ligera tendencia ascendente, aunque hay mucha dispersion es mas probable que los juegos con un critic_score mayor a 80 tengan mejores ventas. Sin embargo por la gran heterogeneidad de los datos no podemos confirmarlo.**

# In[45]:


# Filtramos por PS4 y quitamos los NaN
ps4 = df_recent[df_recent['platform'] == 'PS4'][[
    'user_score', 'critic_score', 'total_sales']].dropna()
ps4 = ps4.astype(float)

# Obtenemos Correlaciones
cor_user = ps4['user_score'].corr(ps4['total_sales'])
cor_critic = ps4['critic_score'].corr(ps4['total_sales'])

print(f"Correlación entre user_score y total_sales: {cor_user:.2f}")
print(f"Correlación entre critic_score y total_sales: {cor_critic:.2f}")


# **Podemos observar que para la correlacion entre user_score y total sales es de -0.03, este valor es casi 0, por lo que podemos afirmar que las calificaciones de los usuarios no estan relacionadas con las ventas**
#
# **Por otro lado, la correlacion entre Critic_score y total_sales es de 0.41, lo cual indica una correlacion moderada donde las calificaciones de los criticos tienden a subir las ventas, sin embargo como mencionaba, la relacion es moderada.**

# In[46]:


# Comparamos las relaciones con las otras plataformas populares:

platforms_to_check = ['PS4', 'XOne', '3DS']

# Creamos un diccionario para guardar los resultados
correlations = {}

for platform in platforms_to_check:
    subset = df_recent[df_recent['platform'] == platform][[
        'user_score', 'critic_score', 'total_sales']].dropna()
    subset = subset.astype(float)  # Asegura que los datos sean numéricos
    user_corr = subset['user_score'].corr(subset['total_sales'])
    critic_corr = subset['critic_score'].corr(subset['total_sales'])
    correlations[platform] = {
        'User Score Corr': user_corr,
        'Critic Score Corr': critic_corr
    }

# Mostramos los resultados
correlation_df = pd.DataFrame(correlations)  # Convertimos a DF el diccionario
print(correlation_df)


# **Haciendo la comparativa entre las correlaciones en las diferentes plataformas, podemos observas que los resultados son similares. Por un lado tenemos al PS4 y XOne donde el user_score no influye mucho en ventas pero el critic_score tiene más peso al momento de hacer la compra. Y por otro lado tenemos al 3DS, donde si bien las correlaciones son bajas, las ventas fluctuan un poco por las criticas de los expertos como de los usuarios.**

# In[47]:


# Pasamos con la comparacion de la distribucion general de los juegos por genero

# Numero de juegos por genero
genre_counts = df_recent['genre'].value_counts()

# Ventas totales por genero
genre_sales = df_recent.groupby(
    'genre')['total_sales'].sum().sort_values(ascending=False)

# Calculamos media y mediana de ventas por genero
genre_mean_sales = df_recent.groupby(
    'genre')['total_sales'].mean().sort_values(ascending=False)
genre_median_sales = df_recent.groupby(
    'genre')['total_sales'].median().sort_values(ascending=False)

# Mostramos los resultados
print("Número de juegos por género:")
print(genre_counts)
print("\nVentas globales totales por género:")
print(genre_sales)
print("\nPromedio de ventas por género:")
print(genre_mean_sales)
print("\nMediana de ventas por género:")
print(genre_median_sales)


# In[48]:


# Graficando los resultados
plt.figure(figsize=(14, 6))

# Cantidad de juegos por genero
plt.subplot(1, 3, 1)
sns.barplot(x=genre_counts.values, y=genre_counts.index,
            alpha=0.7, palette='viridis')
plt.title('Número de juegos por género')
plt.xlabel('Número de juegos')
plt.ylabel('Género')

# Ventas totales por genero
plt.subplot(1, 3, 2)
sns.barplot(x=genre_sales.values, y=genre_sales.index,
            alpha=0.7, palette='plasma')
plt.title('Ventas totales por género')
plt.xlabel('Ventas globales [M]')
plt.ylabel('Género')

# Promedio de ventas por genero
plt.subplot(1, 3, 3)
sns.barplot(x=genre_mean_sales.values, y=genre_mean_sales.index,
            alpha=0.7, palette='magma')
plt.title('Promedio de ventas por género')
plt.xlabel('Promedio de ventas [M]')
plt.ylabel('Género')

plt.tight_layout()
plt.show()


# **Vemos que aunque Action domina en cantidad de juegos y ventas totales, su promedio de ventas por juego es bajo, lo cual sugiere que se producen muchos juegos de action pero tienen ventas bajas, solo unos pocos alcanzan cifras altas.**
#
# **Shooter, Platform y Role-Playing tienen menos cantidad de juegos pero destacan en el promedio de ventas, lo cual indica que aunque se produzcan menos videojuegos de este genero, sus ventas son mejores.**

# **Paso 4. Perfil de usuario por región**

# In[49]:


# Agruparemos las ventas por región y plataforma para ver cuales dominan en cada mercado
top_platforms_na = df_recent.groupby(
    'platform')['na_sales'].sum().sort_values(ascending=False).head(5)
top_platforms_eu = df_recent.groupby(
    'platform')['eu_sales'].sum().sort_values(ascending=False).head(5)
top_platforms_jp = df_recent.groupby(
    'platform')['jp_sales'].sum().sort_values(ascending=False).head(5)

# Mostramos los resultados
print("Top 5 plataformas en NA:\n", top_platforms_na)
print("\nTop 5 plataformas en EU:\n", top_platforms_eu)
print("\nTop 5 plataformas en JP:\n", top_platforms_jp)


# In[50]:


# Creamos una figura con 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Norteamérica
df_recent.groupby('platform')['na_sales'].sum().sort_values(
    ascending=False).head(10).plot(kind='bar', ax=axes[0], color='skyblue')
axes[0].set_title('Ventas por plataforma en Norteamérica')
axes[0].set_ylabel('Ventas en millones')
axes[0].set_xlabel('Plataforma')

# Europa
df_recent.groupby('platform')['eu_sales'].sum().sort_values(
    ascending=False).head(10).plot(kind='bar', ax=axes[1], color='lightgreen')
axes[1].set_title('Ventas por plataforma en Europa')
axes[1].set_xlabel('Plataforma')

# Japón
df_recent.groupby('platform')['jp_sales'].sum().sort_values(
    ascending=False).head(10).plot(kind='bar', ax=axes[2], color='salmon')
axes[2].set_title('Ventas por plataforma en Japón')
axes[2].set_xlabel('Plataforma')

plt.tight_layout()
plt.show()


# **Podemos observar que el top 5 en NA se compone de X360, PS2, Wii, PS3 y DS. Mientras que en EU es PS2, PS3, X360, Wii, PS. Sin embargo en JP tenemos DS, PS2, PS, 3DS, PS3. Esto nos dice que en NA Y EU tienen gustos similares, con fuerte presencia de plataformas como PlayStation y Xbox, JP por otro lado tiene una inclinación marcada hacia las consolas de Nintendo, especialmente DS y 3DS.**

# In[51]:


# Analisis por género y región

# Gráficas de ventas por género en cada región
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# NA
df_recent.groupby('genre')['na_sales'].sum().sort_values(ascending=False).head(
    10).plot(kind='bar', ax=axes[0], color='mediumslateblue')
axes[0].set_title('Ventas por género en NA')
axes[0].set_ylabel('Ventas en millones')
axes[0].set_xlabel('Género')

# EU
df_recent.groupby('genre')['eu_sales'].sum().sort_values(ascending=False).head(
    10).plot(kind='bar', ax=axes[1], color='mediumseagreen')
axes[1].set_title('Ventas por género en EU')
axes[1].set_xlabel('Género')

# JP
df_recent.groupby('genre')['jp_sales'].sum().sort_values(
    ascending=False).head(10).plot(kind='bar', ax=axes[2], color='coral')
axes[2].set_title('Ventas por género en JP')
axes[2].set_xlabel('Género')

plt.tight_layout()
plt.show()


# **Observamos que el genero con más ventas tanto en NA y en EU es de Action, seguido por el genero Sports y Shooters. Mientras que en JP, el primero en la lista corresponde a el genero Role-Playing, seguido de generos como Action y Sports.**

# In[52]:


# Clasificacion ESRB

# Ventas promedio por clasificacion ESRB y región
esrb_sales = df_recent.pivot_table(
    index='rating', values=['na_sales', 'eu_sales', 'jp_sales'], aggfunc='mean')
# Para que salga ordenado el grafico
esrb_sales = esrb_sales[['na_sales', 'eu_sales', 'jp_sales']]

# Graficando
esrb_sales.plot(kind='bar', figsize=(10, 6))
plt.title('Ventas promedio por clasificacion ESRB y región')
plt.xlabel('Clasificación ESRB')
plt.ylabel('Ventas promedio [M]')
plt.grid(axis='y')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# **Podemos observar que las ventas mayores en la region NA se obtienen de las clasificaciones de AO y K-A, las ventas mayores de EU y JP son AO y M. Sin embargo, observando el documento que estamos trabajando, podemos observar muy pocos datos en AO y K-A. Y para comprobarlo ejecutamos la siguiente linea:**

# In[53]:


df_recent['rating'].value_counts()


# **Como podemos observar, no tenemos muchos datos representativos para las clasificaciones "RP", "AO", "K-A", "EC". Esto se debe a que son clasificaciones antiguas para videojuegos. En los dias actuales, las nuevas clasificaciones son las que tienen mas datos como "E", "T", "M", "E10+".**
#
# **Esto nos puede decir que NA tiene el mercado mas activo y diversificado, EU tiene ventas mas bajas, sin embargo tambien es un mercado diversificado y por ultimo en JP con ventas todavia mas bajas.**

# **Paso 5. Prueba de hipotesis**

# **Hipotesis 1**
# **"Las calificaciones promedio de los usuarios para las plataformas Xbox One y PC son las mismas."**
#
# - Hipotesis Nula (H0):
# La media del user_score para XOne es igual a la media de user_score para PC.
#
# - Hipotesis Alternativa (H1):
# La media de user_score para XOne es diferente a la media para PC

# In[54]:


# Filtramos los puntajes de usuario por plataforma
xone_scores = df_recent[(df_recent['platform'] == 'XOne') & (
    df_recent['user_score'].notna())]['user_score']
pc_scores = df_recent[(df_recent['platform'] == 'PC') & (
    df_recent['user_score'].notna())]['user_score']

# Realizamos la prueba de hipotesis
alpha = 0.05
results = stats.ttest_ind(xone_scores, pc_scores, equal_var=False)

print(f"p-valor: {results.pvalue:}")

# Analizamos los resultados
if results.pvalue < alpha:
    print("Rechazamos la hipótesis nula. Las calificaciones promedio de los usuarios son significativamente diferentes.")
else:
    print("No rechazamos la hipótesis nula. Las calificaciones promedio de los usuarios son similares.")


# **Rechazamos la hipotesis, ya que el p-valor es < a alpha. Esto nos indica que las calificaciones promedio de los usuarios son significativamente diferentes. Esto nos quiere decir que los usuarios califican de forma distinta los huegos en cada plataforma.**

# **Hipotesis 2**
# **"Las calificaciones promedio de los usuarios para los géneros de Acción y Deportes son diferentes."**
#
# - Hipotesis Nula (H0):
# Las medias de user_score para juegos de Action y Sports son iguales
#
# - Hipotesis Alternativa (H1):
# Las medias de user_score para action y Sports son diferentes

# In[55]:


# Filtramos los puntajes por genero
action_scores = df_recent[(df_recent['genre'] == 'Action') & (
    df_recent['user_score'].notna())]['user_score']
sports_scores = df_recent[(df_recent['genre'] == 'Sports') & (
    df_recent['user_score'].notna())]['user_score']

# Realizamos la prueba de hipotesis
results_genre = stats.ttest_ind(action_scores, sports_scores, equal_var=False)

print(f"p-valor: {results_genre.pvalue:}")

# Analizamos los resultados
if results_genre.pvalue < alpha:
    print("Rechazamos la hipótesis nula. Las calificaciones promedio de los usuarios son significativamente diferentes.")
else:
    print("No rechazamos la hipótesis nula. Las calificaciones promedio de los usuarios son similares.")


# **Como el p-valor es mayor que alpha, NO se rechaza la hipotesis nula. Esto significa que no hay evidencia estadistica suficiente para afirmar que los usuarios califican de manera diferente los juegos de Action y Sports.**

# **CONCLUSION GENERAL DEL PROYECTO**

# **A lo largo de este proyecto, abordamos distintas etapas clave para comprender el comportamiento de ventas, las plataformas mas exitosas, la influencia de las calificaciones y las preferencias regionales.**
#
# **Comenzamos a analizar los datos desde 1995 en adelante ya que a partir de esta fecha, se obtienen mas y mejores datos y de esta forma podremos tener un analisis representativo del mercado actual. La produccion de videojuegos mostro un crecimiento importante a partir del año 2000, alcanzando su punto maximo de videojuegos producidos en 2008.**
#
# **Por el lado de las ventas globales, pudimos observar que las plataformas mas exitosas de los ultimos años son PS4 y XOne, en las graficas podemos ver que la vida estimada de las plataformas son aproximadamente de 7 a 10 años, por ende estas plataformas tienden a la baja. Las plataformas con mayores ventas son las consolas de SONY y MICROSOFT, sin embargo, en el mercado japones podemos observar que NINTENDO se posiciona con mas fuerza.**
#
# **Pudimos observar que el genero mas comun y con mayores ventas totales es el genero de Action, pero al analizar el promedio de ventas por juego, los Shooters y Role-Playing destacan con mejores resultados por titulo.**
#
# **Para el analisis regional nos podemos dar cuenta que NA y EU compartes gustos similares ya que dominan los generos Action, Sports y Shooters. En el mercado Japones hay una clara preferencia por el genero Role-Playing y plataformas como NINTENDO 3DS y DS muestran preferencia.**
#
# **De la clasificacion ESRB tuvimos algunas observaciones, ya que tenemos clasificaciones antiguas donde tenemos muy pocos datos como: "RP" o "AO". Sin embargo las clasificaciones con mejores ventas son: "E", "T" y "M".**
#
# **Para las hipotesis obtuvimos resultados interesantes, donde se rechazo la hipotesis de que las calificaciones promedio de XOne y PC fueran iguales, lo que indica diferencias significativas en la percepcion de los usuarios.**
# **Y en la segunda prueba No se rechazo la hipotesis de igualdad entre los generos de Action y Sports, lo cual propone que estadisticamente, sus calificaciones promedio de usuarios son similares.**
#
# **Como comentarios finales, me gustaria agregar que el mercado de videojuegos es sensible a factores como la region, tipo de plataforma y de las opiniones de expertos/usuarios. A traves de este analisis logré entender de una mejor forma los factores que influyen en el exito de un videojuego, y poder hacer predicciones para futuras decisiones comerciales o de desarrollo en la industria. Este proyecto fue retador y puso aprueba todos los conocimientos adquiridos hasta ahora, si bien tardé mucho en desarrollar este proyecto, considero que he tenido un avance muy significativo en este camino como Data Scientist.**

# In[ ]:
