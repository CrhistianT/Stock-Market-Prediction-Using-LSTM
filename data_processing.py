import numpy as np 
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt


def show_graphic(data_frame):
        plt.figure(figsize=(14,7))
        plt.plot(range(data.shape[0]),(data['low']+data['high'])/2.0)
        plt.xticks(range(0,data.shape[0],500),data['timestamp'].loc[::500],rotation=45)
        plt.xlabel('fecha',fontsize=18)
        plt.ylabel('precio medio',fontsize=18)
        plt.show()
'''input_file = "daily_MSFT.csv"
data = pd.read_csv(input_file)
data = data.sort_values('timestamp')
show_graphic(data)'''

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

def denormalize(valor_normalizado, input_file):
        data = pd.read_csv(input_file)
        data = data.sort_values('timestamp')
        data.drop(['timestamp'],1,inplace=True)
        data['closef'] = data.close
        data.drop(['close'],1,inplace=True)
        data = data['closef'].values.reshape(-1,1)
        valor_normalizado = valor_normalizado.reshape(-1,1)

        reverse_scaler = preprocessing.MinMaxScaler()
        t = reverse_scaler.fit_transform(data)
        valor_escalado = reverse_scaler.inverse_transform(valor_normalizado)
        return valor_escalado

#Tratamiento de la data
def procesamiento_data(input_file, numero_dias):
    bolsa = pd.read_csv(input_file)
    bolsa = bolsa.sort_values('timestamp')
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
    porcentaje_entrenamiento = round(0.9*len(entrenamiento))
    data_entrenamiento = entrenamiento[:int(porcentaje_entrenamiento),:]
    X_entrenamiento = data_entrenamiento[:,:-1]
    y_entrenamiento = data_entrenamiento[:,-1][:,-1]
    #print(X_entrenamiento[0])
    X_prueba = entrenamiento[int(porcentaje_entrenamiento):,:-1]
    #print(len(X_prueba))
    y_prueba = entrenamiento[int(porcentaje_entrenamiento):,-1][:,-1]
    X_entrenamiento = np.reshape(X_entrenamiento,(X_entrenamiento.shape[0],X_entrenamiento.shape[1],n_caracteristicas))
    X_prueba = np.reshape(X_prueba,(X_prueba.shape[0],X_prueba.shape[1],n_caracteristicas))
    return [X_entrenamiento,y_entrenamiento,X_prueba,y_prueba]
#procesamiento_data("daily_MSFT.csv",30)
