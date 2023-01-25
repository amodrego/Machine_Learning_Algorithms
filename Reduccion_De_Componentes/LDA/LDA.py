# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 11:12:34 2022

@author: alexm

LDA -- ANALISIS DISCRIMINANTE LINEAL

Las observaciones deben venir etiquetdas y el algoritmo separa lo maximo posible los elementos de cada clase --> Algoritmo supervisado

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Carga del fichero
# =============================================================================
dataset = pd.read_csv('Wine.csv')

x = dataset.iloc[:, 0:13].values
y = dataset.iloc[:, 13].values

# =============================================================================
# Separar los datos en train y test
# =============================================================================
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=3/4, random_state=0)


from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)


# =============================================================================
# Antes de generar el modelo de clasificacion se reduce la dimension
# =============================================================================
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
lda = LinearDiscriminantAnalysis(n_components = 2) # Elegimos dos componentes para poder hacer la discriminacion grafica
x_train = lda.fit_transform(x_train, y_train) # Hay que suministrar las categorias de cada una de las observaciones debido a que es un algoritmo supervisado
x_test = lda.transform(x_test)

# Cuando hacemos el transform solo lo hacemos sobre el conjunto de prueba

# =============================================================================
# Ajustar el modelo de Regresión
# =============================================================================
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
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
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    a = (y_set == j).ravel() # ravel() aplana el vector para que sea de dimensiones (n,)
    plt.scatter(X_set[a, 0], X_set[a, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
    
plt.title('Clasificador (Conjunto de Entrenamiento)')
plt.xlabel('Edad')
plt.ylabel('Sueldo Estimado')
plt.legend()
plt.show()