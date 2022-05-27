#!/usr/bin/env python
# coding: utf-8

# # Laboratorio 2: Armado de un esquema de aprendizaje automático
# 
# En el laboratorio final se espera que puedan poner en práctica los conocimientos adquiridos en el curso, trabajando con un conjunto de datos de clasificación.
# 
# El objetivo es que se introduzcan en el desarrollo de un esquema para hacer tareas de aprendizaje automático: selección de un modelo, ajuste de hiperparámetros y evaluación.
# 
# El conjunto de datos a utilizar está en `./data/loan_data.csv`. Si abren el archivo verán que al principio (las líneas que empiezan con `#`) describen el conjunto de datos y sus atributos (incluyendo el atributo de etiqueta o clase).
# 
# Se espera que hagan uso de las herramientas vistas en el curso. Se espera que hagan uso especialmente de las herramientas brindadas por `scikit-learn`.




import numpy as np
import pandas as pd

#Todo: Agregar las librerías que hagan falta
from sklearn.model_selection import train_test_split


# ## Carga de datos y división en entrenamiento y evaluación
# 
# La celda siguiente se encarga de la carga de datos (haciendo uso de pandas). Estos serán los que se trabajarán en el resto del laboratorio.




dataset = pd.read_csv("./data/loan_data.csv", comment="#")

# División entre instancias y etiquetas
X, y = dataset.iloc[:, 1:], dataset.TARGET

# división entre entrenamiento y evaluación
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# 
# Documentación:
# 
# - https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

# ## Ejercicio 1: Descripción de los Datos y la Tarea
# 
# Responder las siguientes preguntas:
# 
# 1. ¿De qué se trata el conjunto de datos?
# 2. ¿Cuál es la variable objetivo que hay que predecir? ¿Qué significado tiene?
# 3. ¿Qué información (atributos) hay disponible para hacer la predicción?
# 4. ¿Qué atributos imagina ud. que son los más determinantes para la predicción?
# 
# **No hace falta escribir código para responder estas preguntas.**

# ## Ejercicio 2: Predicción con Modelos Lineales
# 
# En este ejercicio se entrenarán modelos lineales de clasificación para predecir la variable objetivo.
# 
# Para ello, deberán utilizar la clase SGDClassifier de scikit-learn.
# 
# Documentación:
# - https://scikit-learn.org/stable/modules/sgd.html
# - https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
# 

# ### Ejercicio 2.1: SGDClassifier con hiperparámetros por defecto
# 
# Entrenar y evaluar el clasificador SGDClassifier usando los valores por omisión de scikit-learn para todos los parámetros. Únicamente **fijar la semilla aleatoria** para hacer repetible el experimento.
# 
# Evaluar sobre el conjunto de **entrenamiento** y sobre el conjunto de **evaluación**, reportando:
# - Accuracy
# - Precision
# - Recall
# - F1
# - matriz de confusión

# ### Ejercicio 2.2: Ajuste de Hiperparámetros
# 
# Seleccionar valores para los hiperparámetros principales del SGDClassifier. Como mínimo, probar diferentes funciones de loss, tasas de entrenamiento y tasas de regularización.
# 
# Para ello, usar grid-search y 5-fold cross-validation sobre el conjunto de entrenamiento para explorar muchas combinaciones posibles de valores.
# 
# Reportar accuracy promedio y varianza para todas las configuraciones.
# 
# Para la mejor configuración encontrada, evaluar sobre el conjunto de **entrenamiento** y sobre el conjunto de **evaluación**, reportando:
# - Accuracy
# - Precision
# - Recall
# - F1
# - matriz de confusión
# 
# Documentación:
# - https://scikit-learn.org/stable/modules/grid_search.html
# - https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

# ## Ejercicio 3: Árboles de Decisión
# 
# En este ejercicio se entrenarán árboles de decisión para predecir la variable objetivo.
# 
# Para ello, deberán utilizar la clase DecisionTreeClassifier de scikit-learn.
# 
# Documentación:
# - https://scikit-learn.org/stable/modules/tree.html
#   - https://scikit-learn.org/stable/modules/tree.html#tips-on-practical-use
# - https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
# - https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html

# ### Ejercicio 3.1: DecisionTreeClassifier con hiperparámetros por defecto
# 
# Entrenar y evaluar el clasificador DecisionTreeClassifier usando los valores por omisión de scikit-learn para todos los parámetros. Únicamente **fijar la semilla aleatoria** para hacer repetible el experimento.
# 
# Evaluar sobre el conjunto de **entrenamiento** y sobre el conjunto de **evaluación**, reportando:
# - Accuracy
# - Precision
# - Recall
# - F1
# - matriz de confusión
# 

# ### Ejercicio 3.2: Ajuste de Hiperparámetros
# 
# Seleccionar valores para los hiperparámetros principales del DecisionTreeClassifier. Como mínimo, probar diferentes criterios de partición (criterion), profundidad máxima del árbol (max_depth), y cantidad mínima de samples por hoja (min_samples_leaf).
# 
# Para ello, usar grid-search y 5-fold cross-validation sobre el conjunto de entrenamiento para explorar muchas combinaciones posibles de valores.
# 
# Reportar accuracy promedio y varianza para todas las configuraciones.
# 
# Para la mejor configuración encontrada, evaluar sobre el conjunto de **entrenamiento** y sobre el conjunto de **evaluación**, reportando:
# - Accuracy
# - Precision
# - Recall
# - F1
# - matriz de confusión
# 
# 
# Documentación:
# - https://scikit-learn.org/stable/modules/grid_search.html
# - https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
