# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 10:25:50 2022

@author: alexm

ÁRBOLES DE DECISIÓN --> ÁRBOL DE REGRESIÓN
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
# Creación del modelo
# =============================================================================
from sklearn.tree import DecisionTreeRegressor
reg_tree = DecisionTreeRegressor(random_state=0) # Criterio por defecto es el de error cuadrado

# =============================================================================
# Entrenamiento del modelo
# =============================================================================
reg_tree.fit(x, y)

py = reg_tree.predict([[6.5]])


# =============================================================================
# Visualización de resultados
# =============================================================================
# x_grid = np.arange(min(x), max(x), 0.1)
# x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(x, y, color='red')
plt.plot(x, reg_tree.predict(x), color='blue')
plt.title("Modelo SVR")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()