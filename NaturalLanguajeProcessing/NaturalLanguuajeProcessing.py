# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 08:23:25 2022

@author: alexm

NATURAL LANGUAJE PROCESSING

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# =============================================================================
# Importar el dataset, en este caso no es un CSV (Comma Separated Values) sino
# TSV (Tab Separator Values) debido a que al tratar texto, es mas facil que aparezcan comas
# =============================================================================
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting= 3) 
# Con quoting = 3 el metodo importa los datos sin tener en cuenta las comillas dobles

# =============================================================================
# Limpieza de texto (eliminar palabras, signos de puntuación, simplificar...)
# =============================================================================
import re # Librería para Regular Expressions
import nltk # libreria para eliminar palabras irrelevantes
nltk.download('stopwords') # esta descarga devuelve las palabras irrelevantes
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

data = []

for line in dataset['Review']:
    
    # re.sub indica al programa las expresiones que queremos eliminar, si colocamos como en este caso, un ^
    # indica al programa las expresiones con las que nos queremos quedar, unicamente las letras.
    # El ' ' indica que todos los valores que no son letras se sustituyen por un espacio en blanco
    review = re.sub('[^a-zA-Z]', ' ', line) 
    
    # Pasar las letras que tenemos en el dataset a minúsculas úncamente
    review = review.lower()
    
    # Se separan las palabras de la review en palabras sueltas y se comprueba si alguna se puede eliminar
    review = review.split()
    
    
# =============================================================================
# A continuacion se limpian las conjugaciones y los derivados de palabras, donde se sustituirán por sus raíces, para simplificar el diccionario
# con PorterStemmer()
# =============================================================================
    
    
    # Eliminamos las palabras que sobran y, sobre las que nos quedan, hacemos el stem, donde se guarda la raíz de la palabra
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] # con set (conjunto) se hace la busqueda mucho mas rapida
    review = ' '.join(review) # unimos lo que contenga review con espacios en blanco
    data.append(review)

# =============================================================================
# Creamos el Bago of Words
# =============================================================================
from sklearn.feature_extraction.text import CountVectorizer # este metodo transforma las palabras en vectores de frecuencias
cv = CountVectorizer(max_features=1500) # Con max features nos quedmos con las palabras más frecuentes, así reducimos el numero de columnas de nuestra matriz


# =============================================================================
# Podríamos haber hecho lo mismo que en este algoritmo únicamente con el CountVectorizer pasandole los parámetros necesarios
# Aunque muchas veces sera necesario limpiar expresiones de forma mas controlada o manual y no se puede hacer de este modo.
# =============================================================================
# cv = CountVectorizer(input=dataset[:, 1], lowercase=True, stop_words=set(stopwords.words('english')), token_pattern='[^a-zA-Z]')



x = cv.fit_transform(data).toarray()
y = dataset.iloc[:, 1].values


# =============================================================================
# Con estos datos ya separados en x e y, se puede crear y entrenar un 
# modelo con un algoritmo de clasificacion (K-NN o Arboles de decision, por ejemplo)
# =============================================================================

# =============================================================================
# Separar los datos en train y test
# =============================================================================
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=3/4, random_state=0)

# =============================================================================
# Ajustar el modelo de Clasificacion
# =============================================================================


 # Naive Bayes
# from sklearn.naive_bayes import GaussianNB
# classifier = GaussianNB()
# classifier.fit(x_train, y_train)


# K-NN
# from sklearn.neighbors import KNeighborsClassifier
# classifier = KNeighborsClassifier()
# classifier.fit(x_train, y_train)


# SVM
# from sklearn.svm import SVC
# classifier = SVC()
# classifier.fit(x_train, y_train)


# Regression
# from sklearn.linear_model import LogisticRegression
# classifier = LogisticRegression(random_state=0)
# classifier.fit(x_train, y_train)



# =============================================================================
# NUEVOS MODELOS DE CLASIFICACION NO VISTOS EN APARTADO DE CLASIFICACION:
#   - CART
#   - C5.0
#   -MAX. ENTROPY
# =============================================================================


from sklearn.tree import DecisionTreeClassifier

# CART
classifier = DecisionTreeClassifier(criterion='gini')
classifier.fit(x_train, y_train)

# C5.0 -> DecissionTreeClassifier con un max_depth = 5
classifier = DecisionTreeClassifier(max_depth=5)
classifier.fit(x_train, y_train)

# Maxima entropia -> DecissionTreeClassifier con criterion='entropy'
classifier = DecisionTreeClassifier(criterion='entropy')
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

accuracy = round((cm[0, 1] + cm[1, 0])/(cm[0, 0] + cm[0, 1] + cm[1, 0] + cm[1, 1]), 2)
precision = round(cm[1, 1]/(cm[1, 0] + cm[1, 1]), 2)
recall = round(cm[1, 1] / (cm[0, 1] + cm[1, 1]), 2)

f1_score = (2 * precision * recall)/(precision + recall)


