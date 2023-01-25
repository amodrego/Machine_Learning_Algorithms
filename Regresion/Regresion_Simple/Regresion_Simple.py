# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 11:54:07 2022

@author: alexm

Regresión lineal simple

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Salary_Data.csv')

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1:].values

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=0)


# =============================================================================
# MODELO DE REGRESION LINEAL
# =============================================================================

# Fase de entrenamiento
from sklearn.linear_model import LinearRegression

regression = LinearRegression()
regression.fit(x_train, y_train)
# Con esto ya tenemos el modelo entrenado con los datos suministrados

# Fase de prediccion en el conjunto de test
py = regression.predict(x_test)

plt.plot(py, label='prediccion', color='black')
plt.plot(y_test, label='valores reales', color='green')
plt.title("Predicción vs Valores reales")
plt.xlabel("Años experiencia")
plt.ylabel("Sueldo en $")
plt.show()

plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, regression.predict(x_train), color='blue')
plt.title("Sueldo vs años experiencia (Training)")
plt.xlabel("Años experiencia")
plt.ylabel("Sueldo en $")
plt.show()

plt.scatter(x_test, y_test, color='red')
plt.plot(x_test, py, color='blue') # La recta de regresion sale igual utilicemos los datos que utilicemos
plt.title("Sueldo vs años experiencia (Test)")
plt.xlabel("Años experiencia")
plt.ylabel("Sueldo en $")
plt.show()
