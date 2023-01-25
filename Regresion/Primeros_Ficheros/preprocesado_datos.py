# -*- coding: utf-8 -*-
"""
Spyder Editor

Plantilla de preprocesado de datos.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Cargar el fichero csv de datos de prueba
dataset = pd.read_csv('Data.csv')


x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1:].values

# =============================================================================
# Hay datos que faltan en las tablas, están marcados como NaN y eso debemos resolverlo antes de introducir los datos al 
# algoritmo. 
# =============================================================================

# Tratamiento de los NaN
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy="mean") # Axis indica que coge la media de la fila (1) o de la columna (0)

imputer.fit(x[:, -2:]) # Funcion fit es la que se usa por defecto del objeto Imputer. pasando solo las columnas que tengan valores numericos
# Se puede poner tambien imputer.fit(x[:, 1:3]) el 3 es porque la ultima columna nunca se coge en python

x[:, -2:] = np.round(imputer.transform(x[:, -2:]), 1) # Asignamos de nuevo los valores modificados en imputer a nuestra variable de datos



# =============================================================================
# CATEGORÍAS A DATOS NUMÉRICOS
# =============================================================================


# =============================================================================
# Para procesar datos categóricos
# =============================================================================
# =============================================================================
# from sklearn import preprocessing
# le_X = preprocessing.LabelEncoder()
# x[:,0] = le_X.fit_transform(x[:,0])
# =============================================================================


# =============================================================================
# Las variables dummy son aquellas variables puramente categoricas a las que no se les pueden asignar 0-1-2 porque
# no siguen un orden. Por ello se realiza una subtabla con tantas columnas como parametros tenga esa categoria y se asigna unicamente el
# 1 en la variable que toque
# =============================================================================



# =============================================================================
# Para utilizar one hot encoder y crear variables dummy, 
# ya no hace falta utilizar previamente la función label enconder, si no 
# que para aplicar la dummyficación a la primera columna y dejar el resto 
# de columnas como están, lo podemos hacer con
# =============================================================================

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn import preprocessing

# Segun lo que pone en la ayuda de la clase ColumnTransformer, el primer parámetro es una tupla de [(name, transformer, columns)]
ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],   
                       remainder='passthrough')

x = np.array(ct.fit_transform(x), dtype=float)

# Hacemos el cambio para la categoria purchased, en este caso, como son solo si o no no hace falta el one hot encoder, sino que con el label encoder es suficiente
le_y = preprocessing.LabelEncoder()
y[:, 0] = le_y.fit_transform(y[:,0])


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0) # test_size = 0.2 es que se utiliza un 20% para test.
# El random_state es para reproducir el resultado (seed)


# =============================================================================
# ESCALADO DE DATOS NUMÉRICOS
# =============================================================================

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test) # Solo transform para que use la misma transformacion que ha usado en x_train

# Los valores categoricos que hemos transformado en numericos a veces habra que estandarizarlas y otras que no, dependiendo del criterio. En principio las categorias de X sí
# y las de Y depende (si es una prediccion categorica Si/No entonces no, sino sí)

# ============================================================================
# Cambios de validación cruzada y training/testing
# 
# La función sklearn.grid_search ha cambiado y ya no depende de ese paquete. Ahora debe cargarse con
# 
# from sklearn.model_selection import GridSearchCV
# 
# Cambios en Stats Models
# 
# La carga de la librería import statsmodels.formula.api as sm ahora es con import statsmodels.api as sm
# 
# Cambios en las predicciones
# 
# Ya no se pueden hacer predicciones directamente con valores, si no que deben ser arrays bidimensionales, 
# de modo que lo que antes era y_pred = regression.predict(6.5) ahora es y_pred = regression.predict([[6.5]]).
# 
# Colores
# 
# La instrucción ListedColormap(('red', 'green'))) os dará una advertencia que no os tiene 
# que preocupar (simplemente en lugar de usar el nombre del color, parece que ahora le gusta 
# más que le demos un array de valores numéricos, aunque acepte también los nombres).
# =============================================================================











