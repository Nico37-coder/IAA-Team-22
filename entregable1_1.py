#!/usr/bin/python3.9
# -*- coding: utf-8 -*-

# source
# https://github.com/DiploDatos/IntroduccionAprendizajeAutomatico.git

# I-  ===========================  importacion de modulos/librerias-.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import subprocess as sp
from sklearn.datasets import load_boston  # @ cargar el DS-.
import seaborn as sns
from matplotlib import gridspec
import scipy.stats as st


# @ AA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,\
    mean_absolute_error,\
    mean_squared_error,\
    explained_variance_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# newline
nl = '\n'  # used in f's print formats-.
sp.run(['clear'])  # para limpiar la terminal-.

def new_print(str_to_print: str,
              start_print= True,
              end_print  = False):
    if start_print:
        enc = '< ========//=========== > INI NUEVO PRINT' + \
            '< ==========//========== > '
        print('{0}{1}{2}{3}{4}{5}{6}'.
              format('\t', '\n', enc, '\n',
                     '\t\t\t', str_to_print, '\n'))
    elif end_print:
        enc = '< ========//=========== > END NUEVO PRINT' + \
            '< ==========//========== > '
        print('{0}'.format(enc))

    return 


# II- =========================== load dataSet (DS)-.
boston = load_boston()

# transform DS to PandasDataFrame-.
df = pd.DataFrame(boston.data, columns=boston.feature_names)
# print(f'Tamano del DF ANTES de agregar la variable de interes'
#      f'{df.shape}')
df['target'] = boston.target  # agregamos la variable MEDV == target del DF.-
# print(f'Tamano del DF LUEGO de agregar la variable de interes'
#      f'{df.shape}')

# print(f'{nl}{type(boston)}{nl}')  # Cont. Obj. exposing keys as attributes.
# FUNDAMENTAL MAS IMPORTANTE ==> llaves del diccionario (valores==>values())-.
print(f'{nl}{boston.keys()}{nl}')
# print(f'{nl}{boston}{nl}')  # llaves y valores del diccionario-.

# que contiene cada llave del diccionario?-.
print('{0}{1} ******** Descripcion del dataset ******** {2}{3}'.
      format('\n', '\t', boston['DESCR'], '\n')) # descripcion del dataset-.
# PDF/matriz con los datos de entrada (atributos)-.
# print(boston['data'])
# vector con los valores de precio (-media-) ||| np array (.size == 506)-.
# print(boston['target'])
# nombres de los atrib. p c/column de 'data'  ||| np array (.size == 13)-.-.
# print(boston['feature_names'].size)

'''
< ==//=== > Ejercicio 1: Descripcion de los Datos y de la Tarea < ==//== > 
************************************************************************
Descripcion/caracteristicas del DataSet (DS):
Cada registro/fila del DS corresponde a una ciudad o suburbio de
Boston. Inicialmente el DS fue hecho por el SMSA (Boston Standard
Metropolitan Statistical en 1970). Los creadores del DS son
Harrison, D. and Rubinfeld, D.L.; este DS es una copia del  UCI ML
housing dataset.
https://archive.ics.uci.edu/ml/machine-learning-databases/housing/
y fue tomado del StatLib library el cual es mantenido por la Universidad de
Carnie Mellon-.

IMPORTANTE: el DS, tal como esta,
presenta problemas ETICOS (para mi vinculados a DISCRIMINACION RACIAL Y 
SOCIAL en su lugar deberia trabajarse con el DS original).

====================================================================
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
====================================================================

* Numero de casos reportados (registros): 506 (registro/fila/
  ciudad/suburbio)-.
* Numero de atributos/variables predictivas por registro: 13 (trece)
  -NUMERICAS/CATEGORICAS-. El valor medio (atributo/variable 14) es el
  blanco u objetivo -notar que la key que corresponde a esta variable,
  "target", es diferente al key del DS "MEDV"-.
************************************************************************

1-1- Descripcion/caracteristicas del DataSet (DS):
DS conlos PRECIOS y caracteristicas de casas en diferentes lugares
(suburbios/ciudades) de Boston-.

1-2- ¿Cual es la variable objetivo que hay que predecir?,
     ¿Que significado tiene?
La variable objetivo (target) es MDEV/target: representa el precio medio 
de las casas en miles de dolares norteamericanos-.

1-3- ¿Que informacion (atributos) hay disponibles para hacer la 
     prediccion?
Las variables/atributos (columnas) corresponden a (tomado de la UCI
Machine Learning Repository):
CRIM   : tasa de crimen/per capita por ciudad/suburbio-.
ZN     : proporcion de lotes en zona residencial @ lotes de >
         >25000 pies cuadrados.
INDUS  : proporcion de acres (1 acre = 0.4-0.5 hectareas) comerciales no
         minoristas por ciudad-.
CHAS   : variable ficticia de Charles River (= 1 si limita con el rio,
         0 si no lo hace-.
NOX    : concentracion de Oxido Nitrico (1 parte por 10 millones)-.
RM     : numero promedio de habitaciones por vivienda-.
AGE    : proporcion de unidades ocupados por sus duenos construidas
         antes de 1940-.
DIS    : distancias ponderadas a 5 centros de empleo en Boston-.
RAD    : indice de accesibilidad a la carretera radial-.
TAX    : impuesto a la propiedad de valor total por USD 10000-.
PTRATIO: relacion alumno/maestro por ciudad-.
B      : 1000 (Bk−0.63)2 donde Bk es la proporcion de negros por ciudad-.
LSTAT  : porcentaje de la poblacion de bajo estrato social-.

Variable objetivo:
MDEV/target : valor medio de casas ocupados por sus propietarios 
              (en miles)-.


1-4- Que atributos imagina ud. que seran los mas determinantes para la 
     prediccion?.
RM     : numero promedio de habitaciones por vivienda-.
DIS    : distancias ponderadas a 5 centros de empleo en Boston-.
TAX    : impuesto a la propiedad de valor total por USD 10000-.



1-5- ¿Que problemas observa a priori en el conjunto de datos? ¿Observa 
      posibles sesgos, riesgos, dilemas eticos, etc.?. Piense que los 
      datos pueden ser utilizados para hacer predicciones futuras.

Presenta problemas ETICOS (para mi vinculados a DISCRIMINACION RACIAL Y 
SOCIAL en su lugar deberia trabajarse con el DS original).

'''

# adicional, visualizamos el DS @ chequear si existen valores nulos,
# Tipos de Datos (TdD), cantidad de datos, balanceo de datos, etc.
new_print('Columnas del DF', True)
print(df.columns) # listo las columnas-.
new_print(None, False, True)

new_print('Estadistica de las variables del DF', True)
print(df.describe().T) # estadistica descriptiva del DS-.
new_print(None, False, True)

'''
Al analizar los valores minimos, medios y maximos de las 
variables/atributos numericos se desprende que corresponden a
cantidades de diferentes naturaleza a las cuales les corresponde 
diferentes "escalas" y "unidades".
El aspecto observado sera importante y debera tenerse en cuenta a la
hora de seleccionar y aplicar un modelo de AA. Posteriormente y en 
funcion de los resultados que se obtengan al aplicar un modelo de AA,
se deberan aplicar TRANSFORMACIONES tendientesa mejorar la performance
de los modelos usados en el estudio-.
'''

new_print('Datos duplicados en el DF', True)
print(df.duplicated()) # chequeo si existen datos DUPLICADOS-.
if df.duplicated().sum() == 0:
    print('No existen datos repetidos')
else:
    print('Cantidad de datos repetidos {0}'.
          format(df.duplicated().sum()))
new_print(None, False, True)

'''
No existen datos repetidos en el DF.
'''

new_print('Datos nulos en el DF', True)
print(df.isnull().sum()) # chequeo explicito si exi. datos nulos-.
new_print(None, False, True)

new_print('NaNs en el DF', True)
print(df.isna().sum()) # chequeo explicito si exi. NaNs-.
new_print(None, False, True)

new_print('NaNs en el DF', True)
print(df.info())
new_print(None, False, True)

new_print('Tipos de Datos de las variables del DF', True)
print(df.dtypes)
new_print(None, False, True)

# a partir de la observacion de los TdD de las variables del DS,
# identifico dos variables que son "categoricas",
# las imprimo para chequear-.
new_print('Check vars que pueden ser categoricas en el DF', True)
print(df.loc[:,['RAD', 'CHAS']]) # idem print(df[df['RAD', 'CHAS']])
print(df.loc[:,'RAD'].unique())
print(df.loc[:,'CHAS'].unique())
new_print(None, False, True)

# claramente son variables categoricas:
# RAD: variable categorica que toma los valores: 1, 2, 3, 4, 5, 
# 6, 7, 8, 24. No entiendo que significa cada indice ni su
# implicancia-.
# CHAS: variable categorica binaria. Toma los valores: 0 y 1-.

# convierto ambas variables a enteras-.
cols_to_convert = {
    'RAD': np.int64,
    'CHAS': np.int64
}
df = df.astype(cols_to_convert)


new_print('Tipos de Datos de las variables del DF '
          'MODIFICADAS', True)
print(df.dtypes)
print(df.loc[:,['RAD', 'CHAS']])
print('Valores unicos de la variable {0}{1}{2}'.
      format('RAD', '\n', df.loc[:,'RAD'].unique()))
print('Valores unicos de la variable {0}{1}{2}'.
      format('CHAS', '\n', df.loc[:,'CHAS'].unique()))
print('Total de cada valor unico de la variable {0}{1}{2}'.
      format('RAD', '\n', df.loc[:,'RAD'].value_counts()))
print('Total de cada valor unico de la variable {0}{1}{2}'.
      format('CHAS', '\n', df.loc[:,'CHAS'].value_counts()))
new_print(None, False, True)

# print(df.corr()) # correlation matrix-.

''' ANALISIS:
* MEDV
El valor de MEDV que corresponde al 3º cuartil en 75% (Q3) es 
25m, esto indica que el 75% de los valores de las viviendas de 
Boston ocupadas por sus propietarios es menor a 25 m. Por otro lado 
y teniendo en cuenta que el valor maximo de MEDV es 50m, el doble
del valor de la misma variable para el Q3, esto refleja que existen
pocos registros con valores de MEDV altos y una gran cantidad de 
registros cuyo valor de MEDV esta enter 0m y 25m. Se analizara en 
el siguiente punto empleando encodings visuales.

* ZN
(proporcion de terreno residencial zonificado para lotes con sup.>
25000 pies cuadrados). Existen 0% hasta el Q2 y el valor del 3º 
cuartil es 12.5% y el maximo 100%. Esto pone de manifiesto que la 
mayoria de los terrenos residenciales corresponden a lotes con sup.<
a 25000 pies cuadrados-.

* CHAS
(variable ficticia de Charles River -= 1 si limita con el rio, 
0 si no lo hace-. Es = a 0 -cero- hasta el Q3. Claramente, solo el 
25% de las personas en Boston viven cerca del rio. Ver en los analisis 
de los encodings visuales (graficas)-.

Luego podemos decir que las variables/atributos CRIM, ZN, INDUS, NOX, 
RM, AGE, DIS, y TAX son numericas continuas de tipo real. En tanto las 
variables CHAS  y RAD son variables discretas/categoricas (comos e 
vio). 
Todas las variables y/o atributos cuentan con 506 registros y no 
presentan datos nulos, ni repetidos ni erroneos-.
'''


# Examinamos con mayor detalle los % de datos de cada variable
# correspondientes a OUTLIERS-.
def get_out(ps_val, tot_vals):
    iqr = abs(ps_val.quantile(0.25, interpolation='nearest')-
              ps_val.quantile(0.75, interpolation='nearest'))
    llp = ps_val.quantile(0.25) - (1.5*iqr) # lower_limit_point-.
    ulp = ps_val.quantile(0.75) + (1.5*iqr) # upper_limit_point-.
    # check if there is/are outlier/s-.
    if ps_val.min() > llp and ps_val.max() < ulp:
        por_out= 0.0
    else:
        ps_val_col = ps_val[(ps_val < llp) | (ps_val > ulp)]
        por_out= np.shape(ps_val_col)[0]*100/tot_vals
    return por_out

for ps_name, ps_val in df.items():
    por_out = get_out(ps_val, np.shape(df)[0])
    print('Variable {0:<20} {1:<20} {2:>5.2f}{3:>2}'.
          format(ps_name, 'OUTLIERS', por_out, '%'))
    # val = ps_val.apply(get_out, convert_dtype=False)

'''
Vemos que las variables que presentan OUTLIERS son: CRIM, ZN, CHAS,
RM, B y MDEV/target presentan OUTLIERS (13.04%, 13.44%, 6.92%, 
5.93%, 15.22% y 7.91% respectivamente). Lo observaremos mediante
encodings visuales en el siguiente ejercicio-.
'''

'''
< ==//=== > Ejercicio 2: Visualizacion de los Datos < ==//== > 
'''
# Adenda 1: visualizamos los OUTLIERS reportados en el punto
#            anterior-.
plt.rc('font', size=8)       # set the axes title font size-.
plt.rc('axes', titlesize=8)  # set the axes labels font size-.
plt.rc('axes', labelsize=8)  # set the font size for x tick labels-.
plt.rc('xtick', labelsize=8) # set the font size for y tick labels-.
plt.rc('ytick', labelsize=8) # set the legend font size-.
plt.rc('legend', fontsize=8) # set the font size of the figure title-.
plt.rc('figure', titlesize=8)

if True: # @ comentado 30052022 --FIG_1--
    cat_cols=['RAD', 'CHAS']
    nrows= 3; ncols=5
    plt.figure(figsize=(14,8))
    df_nums_cols= df.drop(cat_cols, axis=1)
    for idx, col in enumerate(df_nums_cols, start=1):
        plt.subplot(nrows, ncols, idx)
        ax= sns.boxplot(data=df, x=col)
        plt.title(col)
        ax.set_xlabel(None)
#    plt.show()

'''
Al analizar la Figura puede corroborarse los valores de los % de 
OUTLIERS reportados al final del Ejercicio 1, es decir:  CRIM, ZN, 
CHAS, RM, B y MDEV/target; 13.04%, 13.44%, 6.92%, 5.93%, 15.22% y 
7.91% respectivamente. Esto debera ser tenido en cuenta a la hora de 
seleccionar y aplicar un modelo de AA (claramente un modelo de AA 
mas robusto frente a OUTLIERS como el de Regresion Logistica -RLog- 
aplica mejor que uno de RL y/o RP) y/o, si el modelo de AA se 
considera dado (ej. RL y/o RP), a la hora de curar los datos para 
analizar y comparar las metricas resultantes del modelo con y sin 
curacion de los OUTLIERS-.
'''

# Adenda 2: visualizacion de las variables categoricas del DS usando
#           graficos de barras y tortas-.
# luego me di cuenta que simplemente con '%.2f%%' se solucionaba
# lo dejo por las dudas (extraida de stackoverflow)-.
if False:
    def autopct_format(values):
        def my_format(pct):
            total = np.sum(values)
            val = int(np.round(pct*total/100.0))
            return '{:.2f}%\n({v:d})'.format(pct, v=val)
        return my_format

if True: # @ comentado 30052022 #--FIG_2--
    nrows= 2; ncols= 2
    figure1, axes= plt.subplots(nrows, ncols, figsize=(12,6)) 
    total= len(df['CHAS'])
    axes[0,0].pie(df['CHAS'].value_counts(),
                  labels= df['CHAS'].value_counts(),
                  autopct='%.2f%%'
                  )
    sns.countplot(x='CHAS',
                  data=df,
                  palette='Set3',
                  ax= axes[0,1]
                  )
    axes[1,0].pie(df['RAD'].value_counts(),
                  labels= df['RAD'].value_counts(),
                  autopct='%.2f'
                  )
    sns.countplot(x='RAD',
                  data=df,
                  palette='Set3',
                  ax= axes[1,1]
                  )
    axes[0,0].set_title('CHAS')
    axes[1,0].set_title('RAD')
    axes[0,1].grid()
    axes[1,1].grid()
# plt.show()

'''
Se observa que la mayoria de los habitantes de BOSTON viven 
en zonas alejadas del RIO (variable CHAS). Solo un 6.92% vive 
en zonas cercanas al Rio de Boston.
En lo que respecta a la  variable RAD, indice que indica el acceso 
a la carretera radial (circunvalacion bostoneana?), a 132 casos 
(26.09%) le corresponde el mayor indice (24); seguidos por 115 y 110 
casos, 22.73% y 21.74% respectivamente, a los indices 5 y 4 
respectivamente. El resto de los indices, 1, 2, 3, 6, 7 y 8 presentan 
la menor cantidad de indices variando entre 17 y 38, indices 1 y 3 
respectivamente.
Estas variables seran consideradas para analizar las graficas de la
variable objetivo y/o target en funcion de los parametros agrupadas 
segun ambas variables categoricas-.
'''

# Adenda 3: analizo la distribucion de los atributos y de la variable
# target u objetivo-.
'''
# only to practice lambda function -the code was removed-.
lista = ['RAD', 'CHAS']
target_columns = list(filter(lambda x: x not in
                             lista,
                             df.columns)
                      )
'''
if True: # @ comentado 30052022  #--FIG_3--
    target_columns= list(df[df_nums_cols.columns].columns) # pythonic form-.
    nsfigs= len(target_columns) # number of subfigs-.
    nc= 3
    nr= int(np.ceil(nsfigs/nc))
    # @ gridspec see Arranging multiple Axes in a Figure Matplotlib doc.-.
    gs= gridspec.GridSpec(nr, nc) 
    fig= plt.figure(figsize=(14,11))
    for idx, col in enumerate(target_columns, start=0):
        ax= fig.add_subplot(gs[idx])
        sns.histplot(x= col,
                     data= df, # the same result that with df_nums_cols-.
                     kde= True,
                     ax= ax)
'''
# idem previous code but more complicated -not pythonics--.
nr=4; nc=3
#plt.figure(figsize=(14,8))
rel_col_matr_list = [[0, 0], [0, 1], [0, 2],
                     [1, 0], [1, 1], [1, 2],
                     [2, 0], [2, 1], [2, 2],
                     [3, 0], [3, 1], [3, 2]
                     ]
fig, ax= plt.subplots(nrows= nr,
                      ncols=nc,
                      figsize=(14,8)
                      )
#for idx, col in enumerate(df_nums_cols, start=1):
for idx, col in enumerate(df_nums_cols.columns):
    print(idx)
    print(col)
    ir, ic= rel_col_matr_list[idx]
    sns.histplot(x=df[col],
                 data=df,                    
                 kde=True,
                 ax= ax[ir,ic]
                 )
    plt.title(col)
'''

'''
Los atributos presentan diferentes tipos de distribuciones. A saber:
* CRIM: exponencial.
* ZN: exponencial.
* INDUS: bimodal.
* NOX: bimodal sesgada hacia la derecha??.
* RM: normal o gaussiana.
* AGE: log-normal bivariada (existe?) o exponencial?.
* DIS: log-normal?
* TAX: bimodal?.
* PTRATIO: vayo Dios a saber !!!.
* B: normal-.
* LSTAT: log-normal

Por otro lado, la variable objetivo, target o MDEV tiene distribucion 
parecida a una normal sesgada hacia la derecha, esto nos indica que 
existen OUTLIERS correspondientes a unos pocos pero grandes valores de
la mencionada variable-.
* target/MDEV: normal o gaussiana sesgada hacia la derecha-.

Estas caracteristicas de las variables seran tenidas en cuenta a la hora 
de: (i) procesar y curar los datos, y/o (ii) previamente a la aplicacion 
del modelo de AA-.
'''

# Adenda 4: grafico de correlacion de las diferentes variables
# (numericas y categoricas)-.
if True: # @ comentado 30052022 --FIG_4--
#    fig= plt.figure(figsize=(100,100))
    sns.plotting_context(font_scale=0.75)
    sns.set(font_scale = 0.75)
    sns.color_palette("rocket", as_cmap=True)
    
    fig= sns.pairplot(df,
                      diag_kind= 'kde',
                      height= 0.7,
                      aspect= 2.0,
                      corner= True,
                      palette='Dark2',
                      #                  hue='CHAS',
                      #                  kind= 'reg',
                      markers='o',
                      diag_kws= dict(shade=True),
                      plot_kws={'s': 5}
                      )
'''
Analisis general: a continuacion se efectua un analisis de correlacion
de cada una de las variables por separado.
* CRIM: presenta correlacion con target, LSTAT, DIS, AGE, y NOX
        (con cariño)-
* ZN: presenta correlacion con LSTAT, B, TAX, RAD, DIS, AGE, RM, 
      (con cariño) NOX, INDUS-.
* INDUS: presenta correlacion con RAD, DIS y NOX -.
* CHAS: -.
* NOX: presenta correlacion con DIS, AGE y RM (con cariño las dos 
ultimas)-.
* RM: presenta correlacion con target, LSTAT-.
* AGE: presenta correlacion con DIS  (con cariño)-.
* DIS: presenta correlacion con target, LASTAT-.
* RAD: -.
* TAX: -.
* PTRATIO: -.
* B: -.
* LSTAT: presenta correlacion con target-.

Por otro lado y en funcion del significado y de la correlacion que tienen
con la variable objetivo (targe/MEDV), se efectua un analisis a OJO de las
variables mas informativas contemplando el siginificado y la correlacion 
(asumimos que las variables que presentan algun tipo de correlacion son las
que deben tenerse en cuenta)-.
target/MEDV: CRIM, INDUS, NOX, RM, AGE, DIS, y LASTAT-.

target/MEDV-CRIM: a medida que aumenta la tasa de crimen/per capita por 
                  ciudad/suburbio- el precio medio de la casa 
                  disminuye (exponencial?)-.
target/MEDV-INDUS: a medida que aumenta la sup. de comercios mayoristas 
                   por ciudad/suburbio- el el precio medio de la casa 
                   disminuye (MEDIO/MEDIO, no tan marcado, exponencial)-.
target/MEDV-NOX: a medida que aumenta la concentracion de Oxido Nitrico por 
                 ciudad/suburbio-.el el precio medio de la casa disminuye 
                 (MEDIO/MEDIO, no tan marcado, lineal)-.
target/MEDV-RM: a medida que aumenta el numero promedio de habitaciones por 
                vivienda por ciudad/suburbio el precio medio de la casa 
                aumena (MARCADAZO, relacion lineal)-.
target/MEDV-AGE: a medida que aumenta la proporcion de unidades ocupados por 
                 sus duenos construidas antes de 1940 por ciudad/suburbio el 
                 precio medio de la casa disminuye (MEDIO/MEDIO, no tan 
                 marcado)-.
target/MEDV-DIS: a medida que aumenta la distancia ponderada a 5 centros de 
                 empleo en Boston por ciudad/suburbio el precio medio de la 
                 casa aumenta hasta aprox. 2.5 para luego mantenerse 
                 constante (MARCADO)-.
target/MEDV-LSTAT: a medida que aumenta el porcentaje de la poblacion de bajo 
                   estrato social por ciudad/suburbio el precio medio de la 
                   casa disminuye (MARCADAZO, relacion exponencial)-.
-------------------
'''

# 2-1- Para cada atributo de entrada, haga una grafica que muestre su
#      relacion con la variable objetivo.
if True: # @ comentado 30052022  --FIG_5--
    target_columns= list(df.columns) # pythonic form-.
    nsfigs= len(target_columns) # number of subfigs-.
    nc= 5
    nr= int(np.ceil(nsfigs/nc))
    # print(nr)
    # @ gridspec see Arranging multiple Axes in a Figure Matplotlib doc.-.
    gs= gridspec.GridSpec(nr, nc) 
    fig= plt.figure(figsize=(19,10))
    for idx, col in enumerate(target_columns, start=0):
        ax= fig.add_subplot(gs[idx])
        r, p = st.pearsonr(df['target'],df[col])
        sns.regplot(data= df,
                    x= 'target',
                    y= col,
                    fit_reg= True,
                    scatter_kws={"color": "black"},
                    line_kws={"color": "red"},
                    ax= ax
                    )
        ax.text(-.05, 1.05, 'r={:.2f}, p={:.2g}'.format(r, p),
                transform=ax.transAxes)
        # ax.set_title('{0} en funcion de {1}'.format('target', col)))
        plt.suptitle('Relacion de al variable objetivos con los atributos'+
                     ' (numericos y categoricos)', fontsize=16)

# 2-2- Estudie las graficas, identificando a ojo los atributos que a su
#      criterio sean los mas informativos para la prediccion.
'''
Como resultado del analisis 4 y 2-1, se desprende que las variables mas 
informaticas para la prediccion son CRIM,  INDUS, LSTAT, RM, y RAD-.
'''

# 2-3- Para ud., ¿cuales son esos atributos?. Listelos en orden
#      de importancia.-.
'''
Atributos mas importantes desde el punto de vista de la prediccion de la 
variable objetivo/target:
1- RM-.
2- LSTAT-.
3- CRIM-.
4- INDUS-.
5- RAD-.
'''

# Adenda 5: para corroborar los resultados presentados a continuacion
#           hacemos los graficos de la matriz de corrlacion con haetmap-.
if True: # @ comentado 30052022 --FIG_6--
    corr = df.corr(method='pearson')
    mask = np.triu(np.ones_like(corr, dtype=bool))
    corr= corr.mask(mask)
    figure, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr,
                ax=ax,
                cmap="YlGnBu",
                linewidths=0.1,
                annot=True
                )
'''
Al analizar la figura vemos que las variables/atributos identificados como 
l@s mas importantes desde el punto de vista de la prediccion de la variable
objetivo/target (RM y LSTAT), son las que presentan mayor correlacion. Este
aspecto debera tenerse en cuenta a la hora de aplicar un modelo de AA con el
fin de prevenir, ya que: (adapatado de -no lo recuerdo, ja ja ja ja !!-)-.

"... las variables/atributos correlacionadas en general no mejoran los 
modelos de AA (aunque depende de las caracteristicas especificas del problema,
como el numero de variables/atributos y el grado de correlacion), sin embargo,
afectan modelos especificos de diferentes maneras y en diferentes grados:
# modelos lineales, ej., RL y/o RLog, la multicolinealidad puede dar lugar a
soluciones que varian ampliamente y que posiblemente sean numericamente 
inestables.
Los modelos de Random Forest pueden ser buenos para detectar interacciones 
entre diferentes atributos/variables, sin embargo los atributos/variables
altamente correlacionados pueden enmascarar estas interacciones.
En general, esto puede verse como un caso especial de la NAVAJA DE OCCAM 
(Occam's razor). Un modelo simple es preferible y, en gran modo, un modelo 
con menos features es mas simple. El concepto de LONGITUD MINIMA DE 
DESCRIPCION lo hace mas preciso
..." 
ver
https://en.wikipedia.org/wiki/Multicollinearity#Consequences_of_multicollinearity
https://statisticsbyjim.com/regression/multicollinearity-in-regression-analysis/
'''
'''
Luego del analiss realizado y pensando en los puntos que siguen, deberemos
pensar en las transformaciones que podriamos usar luego para exponer/poner de 
manifiesto la estructura del DS en general por medio las cuales podamos mejorar 
la precision del modelo de AA. De aqui que tengamos que considerar:
* Una correcta seleccion de atributos/variables basada en remover los 
  atributos/variables mas correlacionados (ver comentario anterior),
* Normalizacion del DS para reducir los efectos de las diferentes escalas de las
  variables/atributos del DS (ver comentario al Punto 1-5)-.
* Estandizarizacion del DS @ reducir los efectos de las diferentes distribuciones
  observadas en los atributos/variables del DS (ver comentario punto )
'''

'''
< ==//=== > Ejercicio 3: Regresion lineal < ==//== > 
'''
# 3-1- Seleccione un solo atributo que considere puede ser el mas
#      apropiado-.
'''
Atributo selecciondo: segun el analisis desarrollado y solo como un primer 
approach, seleccionaremos el atributo/variable: RM-.
'''

# DSs to create a model-.
atrib= df.loc[:,['RM']]; target= df.target
# atrib= atrib.reshape(-1,1)

# split the DS in trainng and test 
X_train , X_test , y_train, y_test = train_test_split(atrib,
                                                      target,
                                                      test_size=0.8,
                                                      random_state=0
                                                      )

# visualizamos los conjuntos de puntos de entrenamiento y
# validacion/test-.
if True:  # --FIG_7-- 
    figure, ax = plt.subplots(figsize=(12, 10))
    ax= plt.scatter(x= X_test.RM,
                    y= y_test,
                    marker='^',
                    color='black',
                    alpha=0.5,
                    label='conjunto de prueba'
                    )
    ax= plt.scatter(x= X_train.RM,
                    y= y_train,
                    marker='o',
                    color='red',
                    alpha=0.5,
                    label='conjunto de test'
                    )
    plt.legend()
    plt.grid(True)
    plt.xlabel('habitaciones/casa')
    plt.ylabel('target/MDEV (precio medio de vivienda)')

# RL-.
# create and training the model-.
# modelo ==> instancia de la clase LinearRegression-.
#model= LinearRegression()
model= LinearRegression()

# vemos si estan definidos los coeficientes del modelo de RLsimple-.
# print(model.coef_) # error: the coef_ are created when fit method is called-.

# entrenamos el modelo
model.fit(X_train, y_train)
# print(model.coef_)
# print(model.intercept_)

# generamos predicciones (con el conjunto de validacion/test)-.
model_test = model.predict(X_test)

# calculamos los errores de la prediccion (val_esperado - val_predicho)-.
def calc_errors(yval:float, yaprox :float):
    r2 = r2_score(yval, yaprox)
    mse= mean_squared_error(yval, yaprox)
    mae= mean_absolute_error(yval, yaprox)
    mevs= explained_variance_score(yval, yaprox)
    return r2, mse, mae, mevs

# puntaje de regresion de la varianza explicada. best == 1.0, worse == 0-.
r2, mse, mae, mevs= calc_errors(y_test, model_test)

print('{0:<40}{1:>5.2f}'.
      format('R2', r2)
      )
print('{0:<40}{1:>5.2f}{2:>3}'.  
      format('Error medio cuadratico (MSE)=', mse, '%')
      )
print('{0:<40}{1:>5.2f}{2:>3}'.  
      format('Error medio absoluto (MAE)=', mae, '%')
      )
print('{0:<40}{1:>5.2f}{2:>3}'.  
      format('Puntaje de regresion de la varianza explicada=',
             mevs, '%'
             )
      )

if True:   # --FIG_8-- 
    # fig= plt.figure(figsize=(14,11))
    figure, ax = plt.subplots(figsize=(12, 10))
    # sns.set(rc={'figure.figsize':(10,8)})
    # sns.scatter(y_test,model_test)
    sns.regplot(data= df,
                x= y_test,
                y= model_test,
                fit_reg= True,
                scatter_kws={"color": "black"},
                line_kws={"color": "red"},
                ax= ax
                )
    plt.xlabel('y Test (True Values)')
    plt.ylabel('predicted values')

if True:   # --FIG_9-- 
    # distribucion-.
    fig= plt.figure(figsize=(14,11))
    sns.displot(y_test-model_test,
                bins=50,
                kde=True
                )

# 3-4 - Grafique el modelo resultante, junto con los puntos de
#       entrenamiento y evaluacion-.
if True: #   # --FIG_10-- 
    figure, ax = plt.subplots(figsize=(12, 10))
    ax= plt.scatter(x= X_test.RM,
                    y= y_test,
                    marker='^',
                    color='black',
                    alpha=0.5,
                    label='conjunto de prueba'
                    )
    ax= plt.scatter(x= X_train.RM,
                    y= y_train,
                    marker='o',
                    color='red',
                    alpha=0.5,
                    label='conjunto de test'
                    )
    ax= plt.scatter(x= X_test.RM,
                    y= model_test,
                    marker='*',
                    color='blue',
                    alpha=0.5,
                    label='conjunto predecido'
                    )
    plt.legend()
    plt.grid(True)
    plt.xlabel('habitaciones/casa')
    plt.ylabel('target/MDEV (precio medio de vivienda)')

if True: # --FIG_11-- 
    fig= plt.figure(figsize=(14,11))
    sns.displot(y_test-model_test,
                bins=50,
                kde=True
                )
# 3-5 - Interprete el resultado, haciendo algun comentario sobre
#       las cualidades del modelo obtenido.
'''
impecable !!!!, ja ja ja ja !!!. Pendig task (tbc == to be completed)-.
'''

'''
< ==//=== > Ejercicio 4: Regresion Polinomial < ==//== > 
'''
# 4-1- Para varios grados de polinomio, haga lo siguiente:
# 4-1-A- Instancie y entrene una regresion polinomial.
# 4-1-B- Prediga y calcule error en entrenamiento y evaluacion.
#        Imprima los valores.
# 4-1-C- Guarde los errores en una lista.
# extracted from
# https://scikit-learn.org/stable/auto_examples/linear_model/
#       plot_polynomial_interpolation.html
# to store train and test errors-.
train_errors = []
val_errors = []
if True: # --FIG_12-- 
    fig, ax = plt.subplots()
    ax.set_prop_cycle(color=["black",    # incompleto (tbc in f(poliegrees)
                             "teal",
                             "yellowgreen",
                             "gold",
                             "darkorange",
                             "tomato"
                             ]
                      )

# 4-1-A- Instancie y entrene una regresion polinomial.
# grados de los polinomios empleados en la RPol-.
# 4-1-B- Prediga y calcule error en entrenamiento y evaluacion.
#        Imprima los valores.
# 4-1-C- Guarde los errores en una lista.
pol_degree= [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
for degree in pol_degree:
    # entrenamiento-.
    pf= PolynomialFeatures(degree=degree)
    lr = LinearRegression()
    model = make_pipeline(pf, lr)
    model.fit(X_train, y_train)

    # predicciones-.
    y_train_pred= model.predict(X_train)
    y_test_pred= model.predict(X_test)

    # store errors-.
    #train_errors.append(mean_squared_error(y_train, y_train_pred)
    #                    )
    #val_errors.append(mean_squared_error(y_test, y_test_pred)
    #                  )
    r2, mse, mae, mevs= calc_errors(y_train, y_train_pred)
    val_to_add= [r2, mse, mae, mevs]
    train_errors= train_errors + val_to_add
    r2, mse, mae, mevs= calc_errors(y_test, y_test_pred)
    val_to_add= [r2, mse, mae, mevs]
    val_errors= val_errors + val_to_add
    
    # grafico-.
    if True: # --FIG_13 and others-- 
        ax.scatter(X_test.RM,
                   y_test_pred,
                   label=f'grado{degree}',
                   alpha=0.5
                   )
        ax.grid(True)

if True: # --FIG_13 and others-- 
    ax.scatter(X_train, y_train, color="blue", label="train")
    ax.legend(loc="upper left")

# imprimo los valores de los errores (mse==> error medio cuadratic)
print('{0}{1}{2}'.format('2\n', '\3\t', 'ERRORES DE ENTRENAMIENTO'))
print('{0:<10}{1}{2}'.
      format('R2', ':', list(np.around(np.array(
          train_errors[0:len(train_errors):4]),2)))) # r2
print('{0:<10}{1}{2}'.
      format('mse', ':', list(np.around(np.array(
          train_errors[1:len(train_errors):4]),2)))) # mse
print('{0:<10}{1}{2}'.
      format('mae', ':', list(np.around(np.array(
          train_errors[2:len(train_errors):4]),2)))) # mae
print('{0:<10}{1}{2}'.
      format('mevs', ':', list(np.around(np.array(
          train_errors[3:len(train_errors):4]),2)))) # mevs
print('{0}{1}{2}'.format('2\n', '\3\t', 'ERRORES DE TEST'))
print('{0:<10}{1}{2}'.
      format('R2', ':', list(np.around(np.array(
          val_errors[0:len(val_errors):4]),2)))) # r2
print('{0:<10}{1}{2}'.
      format('mse', ':', list(np.around(np.array(
          val_errors[1:len(val_errors):4]),2)))) # mse
print('{0:<10}{1}{2}'.
      format('mae', ':', list(np.around(np.array(
          val_errors[2:len(val_errors):4]),2)))) # mae
print('{0:<10}{1}{2}'.
      format('mevs', ':', list(np.around(np.array(
          val_errors[3:len(val_errors):4]),2)))) # mevs

# 4-2- Grafique las curvas de error en terminos del grado del
#      polinomio.
if True: # @ comentado 30052022  # --FIG_14-- 
    print(len(val_errors))
    target_columns= len(val_errors)/len(pol_degree) # pythonic form-.
    nsfigs= target_columns # number of subfigs-.
    nc= 2
    nr= int(np.ceil(nsfigs/nc))
    # @ gridspec see Arranging multiple Axes in a Figure Matplotlib doc.-.
    gs= gridspec.GridSpec(nr, nc) 
    fig= plt.figure(figsize=(14,11))
    for idx in range(int(nsfigs)):
        # type of errors
        if idx == 0: type_error= 'R2'
        elif idx == 1: type_error= 'mean square error (mse)'
        elif idx == 2: type_error= 'mean_absolute_error (mae)'
        elif idx == 3: type_error= 'explained_variance_score (evs)'
        else:
            print('Error type not contemplated (0-3)')
            exit
        ax= fig.add_subplot(gs[idx])
        ax.plot(pol_degree,
                val_errors[0+idx:len(val_errors):4],
                linestyle='--',
                marker='o',
                color='b',
                label='test_errors'
                )
        ax.plot(pol_degree,
                train_errors[0+idx:len(val_errors):4],
                linestyle=':',
                marker='^',
                color='r',
                label='train_errors'
                )
        plt.title('{0}'.format(type_error))
        ax.legend()
        ax.set_xlabel("degree")
        ax.set_ylabel("error")
        ax.grid(True)


# 4-3- Interprete la curva, identificando el punto en que comienza a
#      haber sobreajuste, si lo hay.
'''
Al analizar las curvas se observa que el overfitting comienza a partir del 
grado 8 del polinomio-.
'''
# 4-4- Seleccione el modelo que mejor funcione, y grafique el modelo
# conjuntamente con los puntos.
'''
Al analizar la evolucion y la diferencia del error medio cuadratico entre
el conjunto de entrenamiento y de validacion/test, se concluye que el mejor 
modelo corresponde a la RPol de grado 3-. 
A continuacion se grafican el modelo con el conjunto de datos de DS-. 
'''

# 3-4 - Grafique el modelo resultante, junto con los puntos de
#       entrenamiento y evaluacion-.

pol_degree_best= 3
# entrenamiento-.
pf= PolynomialFeatures(degree=pol_degree_best)
lr = LinearRegression()
model = make_pipeline(pf, lr)
model.fit(X_train, y_train)
# predicciones-.
y_test_pred= model.predict(X_test)
# graficamos-.
if True: # --FIG_15-- 
    figure, ax = plt.subplots(figsize=(12, 10))
    ax= plt.scatter(x= X_test.RM,
                    y= y_test,
                    marker='^',
                    color='black',
                    alpha=0.5,
                    label='conjunto de prueba'
                    )
    ax= plt.scatter(x= X_train.RM,
                    y= y_train,
                    marker='o',
                    color='red',
                    alpha=0.5,
                    label='conjunto de test'
                    )
    ax= plt.scatter(x= X_test.RM,
                    y= y_test_pred,
                    marker='*',
                    color='blue',
                    alpha=0.5,
                    label='conjunto predecido --'
                    )

    plt.legend()
    plt.grid(True)
    plt.xlabel('habitaciones/casa')
    plt.ylabel('target/MDEV (precio medio de vivienda)')

# 4-5- Interprete el resultado, haciendo algun comentario sobre las
#      cualidades del modelo obtenido.
'''
Impecable ja ja ja ja !!!-. (tbc) to be completed !!!!
'''

'''
< ==//=== > Ejercicio 5: Regresion con mas de un Atributo < ==//== > 
'''
# 3-1- Seleccione un solo atributo que considere puede ser el mas
#      apropiado-.

# En este ejercicio deben entrenar regresiones que toman mas de un
# atributo de entrada.
# Seleccione dos o tres atributos entre los mas relevantes encontrados
# en el ejercicio 2.
'''
Segun los analisis realizados (Punto 2-3), se seleccionaran las 
variables/atributos:
1- RM-.
2- CRIM-.
3- TAX-.
para desarrollar este punto-.
'''

# hago otra aproximacion para (not used here)-.
vars_to_delete = ['ZN', 'INDUS', 'CHAS', 'NOX',
                  'AGE', 'DIS', 'TAX', 'PTRATIO',
                  'B', 'LSTAT']
df_multicol= df.drop(vars_to_delete, axis=1) # (not used here)-.
# print(df_multicol.head)

# DSs to create a model-.
# atrib= df.loc[:,['RM', 'CRIM', 'RAD']]; target= df.target
atrib= df[['RM', 'CRIM', 'TAX']]; target= df.target

# split the DS in trainng and test 
X_train , X_test , y_train, y_test = train_test_split(atrib,
                                                      target,
                                                      test_size=0.6,
                                                      random_state=0 # seed
                                                      )
train_errors = [] # list to save training errors in function of polynomial degree-.
val_errors = [] # list to save tests errors in function of polynomial degree-.

if True: # --FIG_16-- 
    fig, ax = plt.subplots()
    ax.set_prop_cycle(color=["teal",        # incompleto (tbc in f(poliegrees)
                             "black",
                             "yellowgreen",
                             "gold",
                             "darkorange",
                             "tomato"
                             ]
                      )

# 5-1-A- Instancie y entrene una regresion polinomial.
# grados de los polinomios empleados en la RPol-.
# 5-1-B- Prediga y calcule error en entrenamiento y evaluacion.
#        Imprima los valores.
# 5-1-C- Guarde los errores en una lista.
pol_degree= [1,2,3,4,5,6,7,8,9,10]
#pol_degree= [3]
for degree in pol_degree:
    # entrenamiento-.
    pf= PolynomialFeatures(degree=degree, include_bias=True)
    
    X_train_poly = pf.fit_transform(X_train)
    lr = LinearRegression()
    lr.fit(X_train_poly, y_train)

    # predicting on training data-set-.
    y_train_pred= lr.predict(X_train_poly)
    
    # predicting on test data-set-.
    y_test_pred= lr.predict(pf.fit_transform(X_test))
    
    # store errors-.
    r2, mse, mae, mevs= calc_errors(y_train, y_train_pred)
    val_to_add= [r2, mse, mae, mevs]
    train_errors= train_errors + val_to_add
    r2, mse, mae, mevs= calc_errors(y_test, y_test_pred)
    val_to_add= [r2, mse, mae, mevs]
    val_errors= val_errors + val_to_add
    
    # grafico-.
    if True: # --FIG_17--
        ax.scatter(X_test.RM,
                   y_test_pred,
                   label=f'grado{degree}',
                   alpha=0.5
                   )
        ax.grid(True)
        # ax.scatter(X_train.RM, y_train, color="blue", label="train")
        ax.legend(loc="upper left")
    

# (viene de 3-4) - Grafique el modelo resultante, junto con los puntos de
# entrenamiento y evaluacion-.
if True: # --FIG_18--
    figure, ax = plt.subplots(figsize=(12, 10))
    ax= plt.scatter(x= X_test.RM,
                    y= y_test,
                    marker='^',
                    color='black',
                    alpha=0.5,
                    label='conjunto de prueba - RPol - Tres atributos'
                    )
    ax= plt.scatter(x= X_train.RM,
                    y= y_train,
                    marker='o',
                    color='red',
                    alpha=0.5,
                    label='conjunto de test - RPol - Tres atributos'
                    )
    ax= plt.scatter(x= X_test.RM,
                    y= y_test_pred,
                    marker='*',
                    color='blue',
                    alpha=0.5,
                    label='conjunto predecido - RPol - Tres atributos'
                )
    plt.legend()
    plt.grid(True)
    plt.xlabel('habitaciones/casa (RM)')
    plt.ylabel('target/MDEV (precio medio de vivienda)')
    
# imprimo los valores de los errores (mse==> error medio cuadratic)
print('{0}{1}{2}'.format('2\n', '\3\t',
                         'ERRORES DE ENTRENAMIENTO  - RPol - Tres atributos') 
      )
print('{0:<10}{1}{2}'.
      format('R2', ':', list(np.around(np.array(
          train_errors[0:len(train_errors):4]),2)))) # r2
print('{0:<10}{1}{2}'.
      format('mse', ':', list(np.around(np.array(
          train_errors[1:len(train_errors):4]),2)))) # mse
print('{0:<10}{1}{2}'.
      format('mae', ':', list(np.around(np.array(
          train_errors[2:len(train_errors):4]),2)))) # mae
print('{0:<10}{1}{2}'.
      format('mevs', ':', list(np.around(np.array(
          train_errors[3:len(train_errors):4]),2)))) # mevs
print('{0}{1}{2}'.format('2\n', '\3\t',
                         'ERRORES DE TEST - RPol - Tres atributos')
      )
print('{0:<10}{1}{2}'.
      format('R2', ':', list(np.around(np.array(
          val_errors[0:len(val_errors):4]),2)))) # r2
print('{0:<10}{1}{2}'.
      format('mse', ':', list(np.around(np.array(
          val_errors[1:len(val_errors):4]),2)))) # mse
print('{0:<10}{1}{2}'.
      format('mae', ':', list(np.around(np.array(
          val_errors[2:len(val_errors):4]),2)))) # mae
print('{0:<10}{1}{2}'.
      format('mevs', ':', list(np.around(np.array(
          val_errors[3:len(val_errors):4]),2)))) # mevs

# (viene de 4-2-) Grafique las curvas de error en terminos del grado del
#                 polinomio-.
if True: # @ comentado 30052022 # --FIG_19--
    print(len(val_errors))
    target_columns= len(val_errors)/len(pol_degree) # pythonic form-.
    nsfigs= target_columns # number of subfigs-.
    nc= 2
    nr= int(np.ceil(nsfigs/nc))
    # @ gridspec see Arranging multiple Axes in a Figure Matplotlib doc.-.
    gs= gridspec.GridSpec(nr, nc) 
    fig= plt.figure(figsize=(14,11))
    for idx in range(int(nsfigs)):
        # type of errors
        if idx == 0: type_error= 'R2'
        elif idx == 1: type_error= 'mean square error (mse)'
        elif idx == 2: type_error= 'mean_absolute_error (mae)'
        elif idx == 3: type_error= 'explained_variance_score (evs)'
        else:
            print('Error type not contemplated (0-3)')
            exit
        ax= fig.add_subplot(gs[idx])
        ax.plot(pol_degree,
                val_errors[0+idx:len(val_errors):4],
                linestyle='--',
                marker='o',
                color='b',
                label='test_errors - RPol - Tres atributos'
                )
        ax.plot(pol_degree,
                train_errors[0+idx:len(val_errors):4],
                linestyle=':',
                marker='^',
                color='r',
                label='train_errors  - RPol - Tres atributos'
                )
        plt.title('{0}'.format(type_error))
        ax.legend()
        ax.set_xlabel("degree")
        ax.set_ylabel("error")
        ax.grid(True)


# Repita el ejercicio anterior, pero usando los atributos seleccionados.
# No hace falta graficar el modelo final.

# Interprete el resultado y compare con los ejercicios anteriores.
# ¿Se obtuvieron mejores modelos?. ¿Porque?

'''
Impecable ja ja ja ja !!!-. (tbc) to be completed !!!!
'''

'''
< ==//=== > Ejercicio 6: A Todo Feature < ==//== > 
'''
'''
# 6-1- Entrene y evalue regresiones pero utilizando todos los atributos
# de entrada (va a andar mucho mas lento). 
X = df.drop(columns="target")
y = df.target
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.8,
                                                    random_state=0
                                                    )
# pol_degree= [3]
for degree in pol_degree:
    # entrenamiento-.
    pf= PolynomialFeatures(degree=degree, include_bias=True)
    
    X_train_poly = pf.fit_transform(X_train)
    lr = LinearRegression()
    lr.fit(X_train_poly, y_train)

    # predicting on training data-set-.
    y_train_pred= lr.predict(X_train_poly)
    
    # predicting on test data-set-.
    y_test_pred= lr.predict(pf.fit_transform(X_test))
    
    # store errors-.
    r2, mse, mae, mevs= calc_errors(y_train, y_train_pred)
    val_to_add= [r2, mse, mae, mevs]
    train_errors= train_errors + val_to_add
    r2, mse, mae, mevs= calc_errors(y_test, y_test_pred)
    val_to_add= [r2, mse, mae, mevs]
    val_errors= val_errors + val_to_add
    
    # grafico-.
    ax.scatter(X_test.RM,
               y_test_pred,
               label=f'grado{degree}',
               alpha=0.5
               )
    ax.grid(True)
    # ax.scatter(X_train.RM, y_train, color="blue", label="train")
    ax.legend(loc="upper left")
    

    # (viene de 3-4) - Grafique el modelo resultante, junto con los puntos de
    # entrenamiento y evaluacion-.
figure, ax = plt.subplots(figsize=(12, 10))
ax= plt.scatter(x= X_test.RM,
                y= y_test,
                marker='^',
                color='black',
                alpha=0.5,
                label='conjunto de prueba - RPol - Tres atributos'
                )
ax= plt.scatter(x= X_train.RM,
                y= y_train,
                marker='o',
                color='red',
                alpha=0.5,
                label='conjunto de test - RPol - Tres atributos'
                )
ax= plt.scatter(x= X_test.RM,
                y= y_test_pred,
                marker='*',
                color='blue',
                alpha=0.5,
                label='conjunto predecido - RPol - Tres atributos'
                )
plt.legend()
plt.grid(True)
plt.xlabel('habitaciones/casa (RM)')
plt.ylabel('target/MDEV (precio medio de vivienda)')


# 6-2- Estudie y comente los resultados.

'''
'''
< ==//=== > Ejercicio 7: Regularizacion < ==//== > 
'''
# 7-1- Entrene y evalue regresiones con regularizacion "ridge".
# Debera probar distintos valores de "alpha" (fuerza de la regularizacion).
# ¿Mejoran los resultados?

''' 
1- Source 1:rom
https://scikit-learn.org/stable/modules/generated/
sklearn.linear_model.Ridge.html

from sklearn.linear_model import Ridge
clf = Ridge(alpha=1.0)
clf.fit(X, y)


2- Source 2- 
https://www.analyticsvidhya.com/blog/2016/01/
ridge-lasso-regression-python-complete-tutorial/

# Fit the model
ridgereg = Ridge(alpha=alpha,normalize=True)
ridgereg.fit(data[predictors],data['y'])
y_pred = ridgereg.predict(data[predictors])
model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=1e-3))


PROOF Lasso Regression
from sklearn.linear_model import Lasso
# Fit the model
lassoreg = Lasso(alpha=alpha,normalize=True, max_iter=1e5)
lassoreg.fit(data[predictors],data['y'])
y_pred = lassoreg.predict(data[predictors])
'''

# 7-2- Estudie y comente los resultados.

plt.show()
