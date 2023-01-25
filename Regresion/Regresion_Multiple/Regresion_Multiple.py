# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 10:42:25 2022

@author: alexm

Regresión lineal múltiple

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 5 columnas ([Gasto I+D], [Gasto administrativo], [Gasto marketing], [Localizacion], [Beneficios])
dataset = pd.read_csv('50_Startups.csv')

# Preprocesado
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1:].values


# Tratar los datos de columna de localizacion, que son variables categoricas --> OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [3])], remainder='passthrough')

x = np.array(ct.fit_transform(x), dtype=float)


# Eliminamos una de las columnas que nos ha devuelto el hot encoder ya que siempre que hacemos este procedimiento hay que quitar una de las columnas para evitar
# que ocurra multicolinealidad

x = x[:, 1:]


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=0)  


from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(x_train, y_train)

# Prediccion de los resultados en el conjunto de testing
py = regression.predict(x_test)

# Eliminar las variables que no aportan para la prediccion del modelo y eliminarlas
import statsmodels.api as sm
# Se agrega una columna de 1 que corresponde al bias. Esta columna irá siempre al inicio (columna 0) --> necesaria para la eliminacion hacia atras
x = np.append(np.ones((50, 1)).astype(int), x, axis=1)

x_opt = x[:, [0, 1, 2, 3, 4, 5]]
sl = 0.05 # Nivel de significacion

regression_ols = sm.OLS(endog=y, exog=x_opt).fit()
regression_ols.summary()

# Buscar variable con el nivel de significacion mas elevado --> x2 que tiene un pvalor = 0.99 y se elimina del modelo
# =============================================================================
# x_opt = x[:, [0, 1, 3, 4, 5]]
# regression_ols = sm.OLS(endog=y, exog=x_opt).fit()
# 
# x_opt = x[:, [0, 3, 4, 5]]
# regression_ols = sm.OLS(endog=y, exog=x_opt).fit()
# 
# x_opt = x[:, [0, 3, 5]]
# regression_ols = sm.OLS(endog=y, exog=x_opt).fit()
# # El pvalor de la variable [5] es de 0.06>0.05 por lo que en este caso la eliminamos, aunque mas adelante se verán otros criterios para la eliminacion de categorias que no
# # solo se base en el pvalor
# 
# x_opt = x[:, [0, 3]] # Nos queda un modelo de regresion lineal simple --> gasto de marketing
# regression_ols = sm.OLS(endog=y, exog=x_opt).fit()
# =============================================================================


# =============================================================================
#  ELIMINACION HACIA ATRAS USANDO PVALORES DE FORMA AUTOMATICA
# =============================================================================

# =============================================================================
# def backwardElimination(x, sl):
#     numVars = len(x[0]) 
#     
#     for i in range(numVars):        
#         regressor_OLS = sm.OLS(y, x).fit()        
#         maxVar = max(regressor_OLS.pvalues).astype(float)        
#         if maxVar > sl:            
#             for j in range(numVars - i):                
#                 if (regressor_OLS.pvalues[j].astype(float) == maxVar):                    
#                     x = np.delete(x, j, 1)    
#     regressor_OLS.summary()    
#     return x 
#  
# sl = 0.05
# x_opt = x[:, [0, 1, 2, 3, 4, 5]]
# x_Modeled = backwardElimination(x_opt, sl)
# =============================================================================

# =============================================================================
# ELIMINACION HACIA ATRAS USANDO PVALORES Y EL VALOR DE R CUADRADO
# =============================================================================

# =============================================================================
# import statsmodels.formula.api as sm
# def backwardElimination(x, SL):    
#     numVars = len(x[0])    
#     temp = np.zeros((50,6)).astype(int)    
#     for i in range(0, numVars):        
#         regressor_OLS = sm.OLS(y, x.tolist()).fit()        
#         maxVar = max(regressor_OLS.pvalues).astype(float)        
#         adjR_before = regressor_OLS.rsquared_adj.astype(float)        
#         if maxVar > SL:            
#             for j in range(0, numVars - i):                
#                 if (regressor_OLS.pvalues[j].astype(float) == maxVar):                    
#                     temp[:,j] = x[:, j]                    
#                     x = np.delete(x, j, 1)                    
#                     tmp_regressor = sm.OLS(y, x.tolist()).fit()                    
#                     adjR_after = tmp_regressor.rsquared_adj.astype(float)                    
#                     if (adjR_before >= adjR_after):                        
#                         x_rollback = np.hstack((x, temp[:,[0,j]]))                        
#                         x_rollback = np.delete(x_rollback, j, 1)     
#                         print (regressor_OLS.summary())                        
#                         return x_rollback                    
#                     else:                        
#                         continue    
#     regressor_OLS.summary()    
#     return x 
#  
# SL = 0.05
# X_opt = X[:, [0, 1, 2, 3, 4, 5]]
# X_Modeled = backwardElimination(X_opt, SL)
# =============================================================================
