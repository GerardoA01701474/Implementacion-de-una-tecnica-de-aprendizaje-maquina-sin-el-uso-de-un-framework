import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression ## solo se usara para dividir el dataset en train y test

from sklearn.model_selection import train_test_split


def hyp(params, x):  # funcion que hace las hipotesis
    y = 0
    size = len(x)
    for i in range(size):
        y = y + params[i] * x[i] 
    return y


def optim(params,x,y):  # funcion para actualizar los parametros
    new_param = params   
    size_x = len(x)
    size_params = len(params)

    for j in range(size_params): ## for que pasa por cada parametro para actualizarlo
        sum = 0
        for i in range(size_x): ## for para hacer la sumatoria
            
            h = hyp(params, x[i])
            
            if j >= size_params-1: ## el ultimo parametros es la b, por lo tanto no se multiplica por alguna de las x's
                f = 1              
            else:
                f = x[i][j]
            sum = sum + (h - y[i]) * f #x[i][j]
        new_param[j] = params[j] - (alpha/size_x) * sum  ### actualiza el parametro j

    return new_param


def error(params,x,y):      ### queremos calcular el error promedio de las hipotesis cada que actualicemos parametros
    h = 0
    sum_error = 0
    for i in range(len(x)):   ## generamos hipotesis para cada instancia
        h = hyp(params, x[i])    
        sum_error = sum_error + (h-y[i]) **2  # formula para calcular el error
        print("hypotesis: " + str(h) + "       actual Y: " + str(y[i]))
    mean_error=sum_error/len(x)
    __error__.append(mean_error)
    return mean_error

def scaling(x):
    new = x


    min_0 = min(np.array(x)[:,0].tolist())   # obtenemos el maximo y minimo de ambos, x1 y x2
    min_1 = min(np.array(x)[:,1].tolist())
    minX = [min_0, min_1]

    max_0 = max(np.array(x)[:,0].tolist())
    max_1 = max(np.array(x)[:,1].tolist())
    maxX = [max_0, max_1]

    
    for i in range(len(x)):        # escalamos de 0 a 1
        for j in range(len(x[i])):
            new[i][j] = (x[i][j] - minX[j]) / (maxX[j] - minX[j])

    return new

    
################ parameters #####
params = [0,0,0]             ### parametros [m1,m2,b]
#x = [[1,1],[2,2],[3,3],[4,4],[5,5]] ## datos de prueba
#y = [2,4,6,8,10] 
__error__ = []   # lista vacia para llenar con el error promedio despues de cada actualizacion de parametros

######### leer valores con pandas
columns = ["Alcohol","Malic acid","Ash","Alcalinity of ash ", "Magnesium", "Total phenols", "Flavanoids", "Nonflavanoid phenols","Proanthocyanins","Color intensity","Hue","OD280/OD315 of diluted wines"
,"Proline "]
df = pd.read_csv(r"C:\Users\gerar\Downloads\wine.data", encoding='utf-8', names = columns)

df_x = df[["Malic acid","Ash"]]
df_y = df[["Alcohol"]]

x = df_x.values.tolist()   ##### uncomment for real values
y = df_y.values.tolist()

#x = scaling(x) ################# scaling 


aux_Y = np.array(y)
aux_Y = np.reshape(aux_Y,(len(aux_Y),)) # la Y es una matriz de [178,1], la pasamos a un arreglo de [178,]
y = aux_Y.tolist()

Xtrain, Xtest, ytrain, ytest = train_test_split(x, y,random_state=1) # dividimos el dataset en datos de train y datos de testeo

epochs = 0

######## global variables ###############
alpha = 0.001 # learning rate

while True:
    epochs += 1
    params = optim(params, x,y)

    
    error(params, Xtrain, ytrain)
    print(params)

    if(epochs == 600):
        error = error(params, Xtrain,ytrain)
        print("error promedio final: " + str(error))
        print('model trained')
        break




input("press enter for test")


__test_error__ = []
for i in range(len(Xtest)):
    #Xtest = scaling(Xtest)
    predict = hyp(params,Xtest[i])
    error_test = predict - ytest[i]
    print("error in test: " + str(error_test))
    __test_error__.append(error_test)




import matplotlib.pyplot as plt
plt.plot(__error__)
plt.plot(__test_error__)
plt.show()