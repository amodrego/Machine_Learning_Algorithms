# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 09:23:36 2022

@author: alexm

CLUSTERING JERÁRQUICO

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Mall_Customers.csv')
x = dataset.iloc[:, [3, 4]].values

# =============================================================================
# Dendrograma para encontrar el numero optimo de clusters 
# =============================================================================
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x, method='ward')) # ward minimiza la varianza que hay entre las distancias de los clusters

plt.title('Dendrograma')
plt.xlabel('Clientes')
plt.ylabel('Distancia euclídea')
plt.show()

# =============================================================================
# Ajustar clustering jerarquico a nuestro conjunto de datos
# =============================================================================
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
py = hc.fit_predict(x)

# =============================================================================
# Visualizacion de los datos de clustering
# =============================================================================
plt.scatter(x[py == 0, 0], x[py == 0, 1], c='red', label='Cautos')
plt.scatter(x[py == 1, 0], x[py == 1, 1], c='blue', label='Estandar')
plt.scatter(x[py == 2, 0], x[py == 2, 1], c='green', label='Objetivo')
plt.scatter(x[py == 3, 0], x[py == 3, 1], c='yellow', label='Descuidados')
plt.scatter(x[py == 4, 0], x[py == 4, 1], c='black', label='Conservadores')
plt.xlabel('Sueldo')
plt.ylabel('Puntuacion')
plt.title('Clusterización')
plt.legend(loc='upper right')
plt.show()