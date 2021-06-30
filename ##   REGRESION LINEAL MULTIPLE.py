##   REGRESION LINEAL MULTIPLE

# IMPORTO LAS LIBRERIAS
from sklearn import datasets, linear_model

# PREPARO LA DATA

boston = datasets.load_boston()
print(boston)
print()

# ENTENDIMIENTO DE LA DATA

# VERIFICO LA INFORMACION CONTENIDA EN EL DATASET

print('Informacion del dataset:')
print(boston.keys())
print()

# VEO LAS CARACTERISTICAS DEL DATASET

print('Caracteristicas del dataset:')
print(boston.DESCR)
print()

# VEO LA CANTIDAD DE DATOS QUE HAY EN DATASET

print('Cantidad de datos dataset')
print(boston.data.shape)
print()

# VERIFICO LA INFORMACION DE LAS COLUMNAS

print('Nonmbre de columnsas:')
print(boston.feature_names)
print()

####  PREPARO LA DATA PARA REGRESION LINEAL MULTIPLE

# SELECCIONO LAS COLUMNAS 5 6 Y 7

X_multiple = boston.data[:,5:8]
print(X_multiple)
print()

# DEFINO LOS DATOS CORRESPONDIENTE A LAS ETIQUETAS

y_multiple = boston.target

### IMPLEMENTACION DE REGRESION LINEAL MULTIPLE

from sklearn.model_selection import train_test_split

# SEPARO LOS DATOS DE TRAIN EN ENTRENAMIENTO Y PRUEBA PARA PROBAR LOS ALGORITMOS

X_train,X_test,y_train,y_test = train_test_split(X_multiple, y_multiple, test_size =0.2)

# DEFINO EL ALGORITMO A UTILIZAR

lr_multiple = linear_model.LinearRegression()

# ENTRENO EL MODELO

lr_multiple.fit(X_train,y_train)

# REALIZO UNA PREDICCION CON LOS DATOS DE PRUEBA

Y_pred_multiple = lr_multiple.predict(X_test)
print(Y_pred_multiple)
print()

# DATOS DEL MODELO REGRESION LINEAL MULTIPLE

print('DATOS DEL MODELO REGRESION LINEAL MULTIPLE')
print()

print('Valor de las pendientes a')
print(lr_multiple.coef_)
print()

print('Valor de ordenada origen b')
print(lr_multiple.intercept_)
print()

print('Precision del modelo')
print(lr_multiple.score(X_train,y_train))
print()

