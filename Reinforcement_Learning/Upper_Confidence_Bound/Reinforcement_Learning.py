# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 09:52:06 2022

@author: alexm

PROBLEMA DEL BANDIDO DE MULTIPLES BRAZOS --> UPPER CONFIDENCE BOUND

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# Tenemos 10 versiones de un anuncio en el que cada columna del dataset es uno de ellos. Las celdas en las que tenemos 1 indica que el usuario (10000 en total) 
# ha clicado en el y, por tanto, se considera exitoso
# =============================================================================
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')
# Muchas veces estas observaciones se van obteniendo en tiempo real, no se tienen en una tabla como se presentan aqui

# =============================================================================
# Random Selection --> selecciona aleatoriamente 
# =============================================================================
# =============================================================================
# import random
# usuarios = 10000
# anuncios = 10
# ads_selected = []
# total_reward = 0
# 
# for n in range(0, usuarios):
#     ad = random.randrange(d)
#     ads_selected.append(ad)
#     reward = dataset.values[n, ad]
#     total_reward = total_reward + reward
# =============================================================================


# =============================================================================
# En cada 'ronda' se consideran los dos numeros que determinan el anuncio i:
#   - Ni --> no de veces que se selecciona el anuncio i hasta la ronda n
#   - Ri --> suma de recompensas de i hasta la ronda n
# =============================================================================
N = 10000
d = 10
number_of_selections = [0] * d
sums_of_rewards = [0] * d
ads_selected = []
total_reward = 0

for n in range (0, N): # Se debe poner un limite con el cual conocer cuando el algoritmo debe empezar a sacar datos
    max_upper_bound = 0 # Para tener establecido cual es el mejor anuncio que se ha visto
    ad = 0
    
    for i in range (0, d):
        if number_of_selections[i] > 0:
            average_reward = sums_of_rewards[i] / number_of_selections[i]
            delta_i = np.sqrt(3/2 * np.log(n+1)/number_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
            
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i # Anuncio con el mayor upper bound
    ads_selected.append(ad)
    number_of_selections[ad] = number_of_selections[ad] + 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward + reward
    
# Histograma de los resultados
plt.hist(ads_selected)
plt.title("Histograma de anuncios")
plt.xlabel("Id del anuncio")
plt.ylabel("Frecuencia de visualizacion de anuncio")
plt.show() 
# Se observa que claramente el anuncio numero 5 (4+1) es el mejor calificado