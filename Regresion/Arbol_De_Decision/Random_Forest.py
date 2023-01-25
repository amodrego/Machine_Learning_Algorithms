# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 11:43:15 2022

@author: alexm

REGRESIÓN MEDIANTE BOSQUES ALEATORIOS
"""

# =============================================================================
# Tratar datos
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')

x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1:].values

# =============================================================================
# Crear modelo y entrenar
# =============================================================================
from sklearn.ensemble import RandomForestRegressor


rforest = RandomForestRegressor(n_estimators=300, random_state=0)
rforest.fit(x, y)

# =============================================================================
# Predicciones
# =============================================================================
py = rforest.predict([[6.5]])


# =============================================================================
# Visualización de resultados
# =============================================================================
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(x, y, color='red')
plt.plot(x_grid, rforest.predict(x_grid), color='blue')
plt.title("Modelo SVR")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()