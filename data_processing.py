import numpy as np 
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt

#input_file = "daily_MSFT.csv"
#data = pd.read_csv(input_file)
#print(data.shape)
#print(data.head())

def show_graphic(data_frame):
        plt.figure(figsize=(14,7))
        plt.plot(range(data.shape[0]),(data['low']+data['high'])/2.0)
        plt.xticks(range(0,data.shape[0],500),data['timestamp'].loc[::500],rotation=45)
        plt.xlabel('fecha',fontsize=18)
        plt.ylabel('precio medio',fontsize=18)
        plt.show()
#data = data.sort_values('timestamp')
#show_graphic(data)

def normalize(data):
        scaler = preprocessing.MinMaxScaler()
        data['open'] = scaler.fit_transform(data.open.values.reshape(-1,1))
        data['high'] = scaler.fit_transform(data.high.values.reshape(-1,1))
        data['low'] = scaler.fit_transform(data.low.values.reshape(-1,1))
        data['close'] = scaler.fit_transform(data.close.values.reshape(-1,1))
        data['volume'] = data['volume'].astype(float)
        data['volume'] = scaler.fit_transform(data.volume.values.reshape(-1,1))
        data['closef'] = data.close
        data.drop(['close'],1,inplace=True)
        return data


#Tratamiento de la data
def procesamiento_data(input_file, numero_dias):
    bolsa = pd.read_csv(input_file)
    bolsa.drop(['timestamp'],1,inplace=True)
    bolsa = normalize(bolsa)
    n_caracteristicas = len(bolsa.columns)
    data = bolsa.as_matrix()
    #print(data)
    entrenamiento = []
    n = len(data)
    for indice in range(n - numero_dias):
        entrenamiento.append(data[indice:indice+numero_dias+1])
    entrenamiento = np.array(entrenamiento)
    porcentaje_entrenamiento = round(0.8*len(entrenamiento))
    data_entrenamiento = entrenamiento[:int(porcentaje_entrenamiento),:]
    X_entrenamiento = data_entrenamiento[:,:-1]
    y_entrenamiento = data_entrenamiento[:,-1][:,-1]

    X_prueba = entrenamiento[int(porcentaje_entrenamiento):,:-1]
    y_prueba = entrenamiento[int(porcentaje_entrenamiento):,-1][:,-1]
    #   X_entrenamiento = np.reshape(X_entrenamiento,(X_entrenamiento.shape[0],X_entrenamiento.shape[1],n_caracteristicas))
    #X_prueba = np.reshape(X_prueba,(X_prueba.shape[0],X_prueba.shape[1],n_caracteristicas))
    return [X_entrenamiento,y_entrenamiento,X_prueba,y_prueba]