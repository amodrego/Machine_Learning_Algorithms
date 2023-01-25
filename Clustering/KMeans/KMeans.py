# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 18:01:03 2022

@author: alexm

Clustering por K Means

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Carga de dataset
dataset = pd.read_csv('Mall_Customers.csv')

# Como datos de entrada usamos los referentes al salario anual y la puntuacion del centro
x = dataset.iloc[:, -2:].values

# Elegir la cantidad de clústers --> Método del codo
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(x)
    
    wcss.append(kmeans.inertia_) # Parámetro que devuelve la suma de los cuadrados de las distancias

plt.plot(range(1, 11), wcss)
plt.title('Método del codo')
plt.xlabel('Número de clústers')
plt.ylabel('WCSS')
plt.show()

# Como el codo mas prominente lo hemos obtenido en 5, la mejor eleccion para el numero de clusters es de 5.

# Una vez que tenemos claro el numero de clusters inicial, se genera el modelo kmeans con el numero obtenido
kmeans = KMeans(n_clusters=5, random_state=0)
py = kmeans.fit_predict(x)


# =============================================================================
# Visualizacion de los datos
# =============================================================================
plt.scatter(x[:, 0], x[:, 1], c=py) # Ponemos c=py para que pinte los puntos en funcion del clúster al que pertenecen
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:, 1], c='red')
plt.xlabel('Sueldo')
plt.ylabel('Puntuacion')
plt.title('Clusterización')
plt.show()


