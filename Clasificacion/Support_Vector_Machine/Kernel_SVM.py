# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 08:18:15 2022

@author: alexm

KERNEL SVM
"""

# =============================================================================
# Carga del fichero
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Cargar y separar los datos en x e y
dataset = pd.read_csv('Social_Network_Ads.csv')

x = dataset.iloc[:, [2, 3]].values # Con el corchete seleccionamos solo las columnas que queramos sin que tengan que ser consecutivas
y = dataset.iloc[:, -1:].values


# =============================================================================
# Separar los datos en train y test
# =============================================================================
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=3/4, random_state=0)


# =============================================================================
# Escalado de datos
# =============================================================================

# Como la edad y los sueldos se mueven en escalas muy distintas, escalamos los datos
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)
# Los valores de y no los escalamos porque van de 0 a 1

# =============================================================================
# Ajustar el modelo de Regresión
# =============================================================================

from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(x_train, y_train)





# =============================================================================
# Predicción de resultados de test
# =============================================================================

py = classifier.predict(x_test)

# =============================================================================
# Evaluar el rendimiento del modelo en base a la matriz de confusión --> compara la prediccion con los datos de test que tenemos
# =============================================================================
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, py) # En la diagonal principal tenemos el numero de valores que se han clasificado correctamente y en la otra los que se han clasificado mal

# =============================================================================
# Representacion gráfica
# =============================================================================
from matplotlib.colors import ListedColormap
X_set, y_set = x_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    a = (y_set == j).ravel() # ravel() aplana el vector para que sea de dimensiones (n,)
    plt.scatter(X_set[a, 0], X_set[a, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
    
plt.title('Clasificador (Conjunto de Entrenamiento)')
plt.xlabel('Edad')
plt.ylabel('Sueldo Estimado')
plt.legend()
plt.show()

# =============================================================================
# La regresion logistica en el fondo usa la regresion lineal, de ahi que la frontera entre los dos lados es una recta. Porque el problema es bidimensional. 
# No todo tiene por que clasificarse con una linea recta
# =============================================================================


# =============================================================================
# Resultados con el conjunto de test
# =============================================================================
X_set, y_set = x_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), # contourf() pinta el plano en total de la gráfica
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    a = (y_set == j).ravel() # ravel() aplana el vector para que sea de dimensiones (n,)
    plt.scatter(X_set[a, 0], X_set[a, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
    
plt.title('Clasificador (Conjunto de Test)')
plt.xlabel('Edad')
plt.ylabel('Sueldo Estimado')
plt.legend()
plt.show()