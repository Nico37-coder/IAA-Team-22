#!/usr/bin/env python
# coding: utf-8

# # Laboratorio 1: Regresión en Boston
# 
# En este laboratorio deben hacer experimentos de regresión con el conjunto de datos "Boston house prices dataset".
# 
# Estudiarán el dataset, harán visualizaciones y seleccionarán atributos relevantes a mano.
# 
# Luego, entrenarán y evaluarán diferentes tipos de regresiones, buscando las configuraciones que mejores resultados den.




import numpy as np
import matplotlib.pyplot as plt


# ## Carga del Conjunto de Datos
# 
# Cargamos el conjunto de datos y vemos su contenido.




from sklearn.datasets import load_boston
boston = load_boston()
boston.keys()





# print(boston['DESCR'])   # descripción del dataset
# boston['data']           # matriz con los datos de entrada (atributos)
# boston['target']         # vector de valores a predecir
boston['feature_names']  # nombres de los atributos para cada columna de 'data'





boston['data'].shape, boston['target'].shape


# ## División en Entrenamiento y Evaluación
# 
# Dividimos aleatoriamente los datos en 80% para entrenamiento y 20% para evaluación:




from sklearn.model_selection import train_test_split
X, y = boston['data'], boston['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)
X_train.shape, X_test.shape


# ## Ejercicio 1: Descripción de los Datos y la Tarea
# 
# Responda las siguientes preguntas:
# 
# 1. ¿De qué se trata el conjunto de datos?
# 2. ¿Cuál es la variable objetivo que hay que predecir? ¿Qué significado tiene?
# 3. ¿Qué información (atributos) hay disponibles para hacer la predicción?
# 4. ¿Qué atributos imagina ud. que serán los más determinantes para la predicción?
# 5. ¿Qué problemas observa a priori en el conjunto de datos? ¿Observa posibles sesgos, riesgos, dilemas éticos, etc? Piense que los datos pueden ser utilizados para hacer predicciones futuras.
# 
# **No hace falta escribir código para responder estas preguntas.**
      #**Responder todas las preguntas acá.**
# ## Ejercicio 2: Visualización de los Datos
# 
# 1. Para cada atributo de entrada, haga una gráfica que muestre su relación con la variable objetivo.
# 2. Estudie las gráficas, identificando **a ojo** los atributos que a su criterio sean los más informativos para la predicción.
# 3. Para ud., ¿cuáles son esos atributos? Lístelos en orden de importancia.




# 1. Resolver acá. Ayuda/ejemplo:
feature = 'CRIM'
selector = (boston['feature_names'] == feature)
plt.scatter(X[:, selector], y, facecolor="dodgerblue", edgecolor="k", label="datos")
plt.title(feature)
plt.show()

#**2. Responder acá****3. Responder acá**

# ## Ejercicio 3: Regresión Lineal
# 
# 1. Seleccione **un solo atributo** que considere puede ser el más apropiado.
# 2. Instancie una regresión lineal de **scikit-learn**, y entrénela usando sólo el atributo seleccionado.
# 3. Evalúe, calculando error cuadrático medio para los conjuntos de entrenamiento y evaluación.
# 4. Grafique el modelo resultante, junto con los puntos de entrenamiento y evaluación.
# 5. Interprete el resultado, haciendo algún comentario sobre las cualidades del modelo obtenido.
# 
# **Observación:** Con algunos atributos se puede obtener un error en test menor a 50.




# 1. Resolver acá. Ayuda:
feature = 'CRIM'  # selecciono el atributo 'CRIM'
selector = (boston['feature_names'] == feature)
X_train_f = X_train[:, selector]
X_test_f = X_test[:, selector]
X_train_f.shape, X_test_f.shape





# 2. Instanciar y entrenar acá.





# 3. Predecir y evaluar acá.





# 4. Graficar acá. Ayuda:
x_start = min(np.min(X_train_f), np.min(X_test_f))
x_end = max(np.max(X_train_f), np.max(X_test_f))
x = np.linspace(x_start, x_end, 200).reshape(-1, 1)
# plt.plot(x, model.predict(x), color="tomato", label="modelo")

plt.scatter(X_train_f, y_train, facecolor="dodgerblue", edgecolor="k", label="train")
plt.scatter(X_test_f, y_test, facecolor="white", edgecolor="k", label="test")
plt.title(feature)
plt.legend()
plt.show()

  #**5. Responder acá**
# ## Ejercicio 4: Regresión Polinomial
# 
# En este ejercicio deben entrenar regresiones polinomiales de diferente complejidad, siempre usando **scikit-learn**.
# 
# Deben usar **el mismo atributo** seleccionado para el ejercicio anterior.
# 
# 1. Para varios grados de polinomio, haga lo siguiente:
#     1. Instancie y entrene una regresión polinomial.
#     2. Prediga y calcule error en entrenamiento y evaluación. Imprima los valores.
#     3. Guarde los errores en una lista.
# 2. Grafique las curvas de error en términos del grado del polinomio.
# 3. Interprete la curva, identificando el punto en que comienza a haber sobreajuste, si lo hay.
# 4. Seleccione el modelo que mejor funcione, y grafique el modelo conjuntamente con los puntos.
# 5. Interprete el resultado, haciendo algún comentario sobre las cualidades del modelo obtenido.
# 
# **Observación:** Con algunos atributos se pueden obtener errores en test menores a 40 e incluso a 35.




# 1. Resolver acá.





# 2. Graficar curvas de error acá.

#**3. Responder acá**



# 4. Reconstruir mejor modelo acá y graficar.

#**5. Responder acá**

# ## Ejercicio 5: Regresión con más de un Atributo
# 
# En este ejercicio deben entrenar regresiones que toman más de un atributo de entrada.
# 
# 1. Seleccione **dos o tres atributos** entre los más relevantes encontrados en el ejercicio 2.
# 2. Repita el ejercicio anterior, pero usando los atributos seleccionados. No hace falta graficar el modelo final.
# 3. Interprete el resultado y compare con los ejercicios anteriores. ¿Se obtuvieron mejores modelos? ¿Porqué?




# 1. Resolver acá. Ayuda (con dos atributos):
selector = (boston['feature_names'] == 'CRIM') | (boston['feature_names'] == 'ZN')
X_train_fs = X_train[:, selector]
X_test_fs = X_test[:, selector]
X_train_fs.shape, X_test_fs.shape





# 2. Resolver acá.

#**3. Responder acá.**


# ## Más ejercicios (opcionales)
# 
# ### Ejercicio 6: A Todo Feature
# 
# Entrene y evalúe regresiones pero utilizando todos los atributos de entrada (va a andar mucho más lento). Estudie los resultados.
# 
# ### Ejercicio 7: Regularización
# 
# Entrene y evalúe regresiones con regularización "ridge". Deberá probar distintos valores de "alpha" (fuerza de la regularización). ¿Mejoran los resultados?
# 
