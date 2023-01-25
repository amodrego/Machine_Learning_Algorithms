# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 12:12:47 2022

@author: alexm
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')

# Preprocesado
x = dataset.iloc[:, 1:2].values # En este caso no usaremos la primera columna, ya que con la segunda tenemos esos casos contemplados
# Si ponemos dataset.iloc[:, 1].values la x la trata como vector, no como matriz
y = dataset.iloc[:, -1:].values


# Al tener pocos datos, no separamos en train y test

# Tampoco se escalan los datos para mantener la relacion de las variables sin alterarla


# Generamos el modelo polinomial. Generaremos tambien un modelo de regresion lineal para comparar los resultados arrojados por cada uno de los modelos

# Regresion lineal
from sklearn.linear_model import LinearRegression

lin_regression = LinearRegression()
lin_regression.fit(x, y)


# Regresion polinomial
from sklearn.preprocessing import PolynomialFeatures
pol_transform = PolynomialFeatures(degree=4)
x_poli = pol_transform.fit_transform(x) # En x_poli se ha añadido automatico el término independiente del bias. La ultima columna es el cuadrado de la variable inicial que teniamos


pol_regression = LinearRegression()
pol_regression.fit(x_poli, y)

# Modelo lineal
plt.scatter(x, y, color='red') # Modelo lineal
plt.plot(x, lin_regression.predict(x), color='blue')
plt.title('Modelo lineal')
plt.xlabel('Posicion del empleado')
plt.ylabel('Sueldo')
plt.show()

# Modelo polinomial
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)
x_poli = pol_transform.fit_transform(x_grid)
plt.scatter(x, y, color='red') # Modelo lineal
plt.plot(x_grid, pol_regression.predict(x_poli), color='blue')
plt.title('Modelo lineal')
plt.xlabel('Posicion del empleado')
plt.ylabel('Sueldo')
plt.show()



# Prediccion de los dos modelos
lin_regression.predict([[6.5]]) # Con solo un dato se devuelve un unico valor
pol_regression.predict(pol_transform.fit_transform([[6.5]]))


# =============================================================================
# En Python 3.7+ ha cambiado la forma en la que los datos son definidos. 
# En lugar de ser por defecto data frames ahora son ndarrays y esto provoca que al hacer la predicción en la regresión polinómica, ya no podáis hacerlo con
# 
# lin_reg.predict(6.5)
# 
# ya que el modelo ya no conoce acerca de números. 
# En su lugar hay que construir un ndarray con un solo elemento, el propio número que queréis usar para predecir. Es como crear una matriz 1x1, 
# que a su vez resulta ser un número, pero para ello necesitáis ese doble corchete para arreglar el problema que os surgirá en la próxima clase:
# 
# lin_reg.predict([[6.5]])
# 
# =============================================================================


