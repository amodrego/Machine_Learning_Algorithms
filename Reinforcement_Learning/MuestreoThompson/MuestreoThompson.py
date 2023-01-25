# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 09:39:16 2022

@author: alexm

MUESTREO THOMPSON

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


# =============================================================================
# Tenemos 10 versiones de un anuncio en el que cada columna del dataset es uno de ellos. Las celdas en las que tenemos 1 indica que el usuario (10000 en total) 
# ha clicado en el y, por tanto, se considera exitoso
# =============================================================================
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')
# Muchas veces estas observaciones se van obteniendo en tiempo real, no se tienen en una tabla como se presentan aqui


# =============================================================================
# 
#   
#
# =============================================================================
N = 10000
d = 10
number_of_rewards_1 = [0] * d
number_of_rewards_0 = [0] * d
ads_selected = []
total_reward = 0

for n in range (0, N): # Se debe poner un limite con el cual conocer cuando el algoritmo debe empezar a sacar datos
    max_random = 0 # Para tener establecido cual es el mejor anuncio que se ha visto
    ad = 0
    
    for i in range (0, d):
        # Calcular la probabilidad de exito
        random_beta = random.betavariate(number_of_rewards_1[i] + 1, number_of_rewards_0[i] + 1)
        if random_beta > max_random:
            max_random = random_beta
            ad = i # Anuncio con el mayor upper bound
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    if reward == 1:
        number_of_rewards_1[ad] = number_of_rewards_1[ad] + 1
    else:
        number_of_rewards_0[ad] = number_of_rewards_0[ad] + 1
        
    total_reward = total_reward + reward
    
# Histograma de los resultados
plt.hist(ads_selected)
plt.title("Histograma de anuncios")
plt.xlabel("Id del anuncio")
plt.ylabel("Frecuencia de visualizacion de anuncio")
plt.show() 
# Se observa que claramente el anuncio numero 5 (4+1) es el mejor calificado