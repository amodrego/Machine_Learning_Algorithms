# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 19:09:37 2022

@author: alexm

MÁQUINAS DE SOPORTE VECTORIAL (SUPPORT VECTOR MACHINE)
SUPPORT VECTOR REGRESSION (SVR)

"""

# =============================================================================
# 
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Cargar y separar los datos en x e y
dataset = pd.read_csv('Position_Salaries.csv')

x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1:].values


# Hacer separacion en train y test --> Si hay pocos datos de entrenamiento este paso no lo hacemos y lo usamos todo para entrenar
# =============================================================================
# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=0)
# =============================================================================


# Tratar los datos que aparezcan como NaN
# =============================================================================
# from sklearn.impute import SimpleImputer
# imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# imputer.fit(x[:, :-1]) # Poner las columnas de x que sean valores numéricos. Aqui se asignan los valores de las columnas pasadas al imputer.
# =============================================================================

# A continuacion se tiene que asignar de nuevo las columnas modificadas por el imputer a la matriz x
# =============================================================================
# x[:, :-2] = np.round(imputer.transform(x[:,:-2]), 1)
# =============================================================================

# Escalado de variables (en categorias numericas no hace falta normalmente escalar porque hay librerias que lo hacen solas o porque no hace falta)
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)


# Crear el modelo de Support Vector Machine y entrenar
from sklearn.svm import SVR
svr = SVR(kernel='rbf') # rbf es el valor por defecto de kernel y es Radial Based Function (kernel gaussiano)
svr.fit(x, y)


# Prediccion de datos
py = svr.predict(sc_x.transform([[6.5]])) # En esta linea usamos el escalador sc_x porque el valor que le estamos pasando (6.5) es un valor de X
pred = sc_y.inverse_transform([py]) # Aqui ya hacemos la transformacion del escalador sc_y porque ya estamos con un valor escalado en y


# Visualización de los resultados del Modelo Polinómico
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color = "red")
plt.plot(sc_x.inverse_transform(x_grid), sc_y.inverse_transform(svr.predict(x_grid).reshape(-1, 1)), color = "blue")
plt.title("Modelo SVR")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()

