from sklearn.feature_selection import RFE
from sklearn.svm import SVR
import pandas as pd
import numpy as np

data = pd.read_csv('D:/Anaconda/datasets/ads/Advertising.csv')

feature_cols = ['TV','Radio','Newspaper']
#SEPARO EN DOS DATASETS
#X VAN LAS COLUMNAS RADIO,TV,NEWS
#Y ES LA COLUMNAS DE SALES QUE ES LO QUE QUIERO PREDECIR

x = data[feature_cols]
y = data['Sales']

estimator = SVR(kernel = 'linear')
#LE INDIQUE QUE QUIERO UTILIZAR SOLO 2 DE LAS 3 COLUMNAS DE X
selector = RFE(estimator,2,step = 1)
selector = selector.fit(x,y)

#ME MUESTRA CUALES COLUMNAS USO EN EL MODELO PARA PREDECIR
print(selector.support_)
#ME MUESTRA LA CALIDAD DE CADA COLUMNAS
print(selector.ranking_)


from sklearn.linear_model import LinearRegression
#ELIJO LAS X PARA PREDECIR Y
x_pred = x[['TV','Radio']]
#CREO QUE MODELO
lm = LinearRegression()
#LO ENTRENO PASSANDOLE LAS COLUMNAS X Y LA Y
lm.fit(x_pred,y)
#PREDICCION
print(lm.predict(x_pred))

#PRINTEO DE ALGUNOS DATOS DEL MODELO
print(lm.intercept_)
print(lm.coef_)
print(lm.score(x_pred,y))