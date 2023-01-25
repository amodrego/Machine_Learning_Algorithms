# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 13:10:53 2022

@author: alexm

GRADIENT BOOSTING --> BIG DATA
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# Carga del fichero
# =============================================================================
dataset = pd.read_csv('Churn_Modelling.csv') 

x = dataset.iloc[:, 3:13].values # Con el corchete seleccionamos solo las columnas que queramos sin que tengan que ser consecutivas
y = dataset.iloc[:, 13].values


# =============================================================================
# Cambiar las variables con One Hot Encoder
# =============================================================================
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn import preprocessing

labelencoder_x_1 = LabelEncoder()
x[:, 1] = labelencoder_x_1.fit_transform(x[:, 1])

labelencoder_x_2 = LabelEncoder()
x[:, 2] = labelencoder_x_2.fit_transform(x[:, 2])

# Segun lo que pone en la ayuda de la clase ColumnTransformer, el primer parámetro es una tupla de [(name, transformer, columns)]
ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [1])],   
                       remainder='passthrough')

x = np.array(ct.fit_transform(x), dtype=float)

# Quitamos la primera columna para eliminar el problema de multicolinealidad
x = x[:, 1:]


# =============================================================================
# Dividir dataset
# =============================================================================
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=3/4, random_state=0)


# =============================================================================
# Ajustar el modelo XGBoost al conjunto de entrenamiento
# =============================================================================
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(x_train, y_train)


# =============================================================================
# Predecir y evaluar el rendimiento
# =============================================================================

py = classifier.predict(x_test)

# =============================================================================
# Evaluar el rendimiento del modelo en base a la matriz de confusión --> compara la prediccion con los datos de test que tenemos
# =============================================================================
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, py)


# =============================================================================
# Aplicar KFold cross validation
# =============================================================================
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()
