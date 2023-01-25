# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 12:02:10 2022

@author: alexm

APRIORI 
Optimizar ventas de un supermercado

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import Apriori_Python.apyori as apy

dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)

# Bucles anidados para pasarle al modelo solamente una lista de listas sin tener parametros NaN
transactions = []

for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])
    
# Entrenar algoritmo de apriori
from Apriori_Python.apyori import apriori
rules = apriori(transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_lenght=2)
# El minimo que buscamos (min support) es en 21 cestas de la compra a la semana, que es el espacio temporal en el que tenemos el dataset (3 veces al dia)
# Si ponemos un nivel de confianza alto al final no obtendremos ningun resultado ya que obtendremos casualidades mas que causalidades

result = list(rules)
