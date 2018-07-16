import numpy as np 
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt

data = pd.read_csv("daily_MSFT.csv")

data = data.sort_values('timestamp')

def show_data(data):
	plt.figure(figsize = (18,9))
	plt.plot(range(data.shape[0]),(data['low']+data['high'])/2.0)
	plt.xticks(range(0,data.shape[0],500),data['timestamp'].loc[::500],rotation=45)
	plt.xlabel('Date',fontsize=18)
	plt.ylabel('Mid Price',fontsize=18)
	plt.show()




#data.drop(['timestamp'],1,inplace=True)
#Nonmalizando la data
scaler = preprocessing.MinMaxScaler()
data['open'] = scaler.fit_transform(data.open.values.reshape(-1,1))
data['high'] = scaler.fit_transform(data.high.values.reshape(-1,1))
data['low'] = scaler.fit_transform(data.low.values.reshape(-1,1))
data['close'] = scaler.fit_transform(data.close.values.reshape(-1,1))
data['volume'] = data['volume'].astype(float)
data['volume'] = scaler.fit_transform(data.volume.values.reshape(-1,1))




#Tratamos la data
def load_data(bolsa, tamanio_sequencia):
	n_caracteristicas = len(bolsa.columns)
	data = bolsa.as_matrix()
	resultados = []

	for indece in range(len(data) - (tamanio_sequencia + 1)):
		resultados.append(data[indece: indece + tamanio_sequencia + 1])

	resultados = np.array(resultados)
	fila = resultados.shape[0]

	entrenamiento = resultados[:int(fila),:]

	data_X= entrenamiento[:, :-1]
	data_y = entrenamiento[:, -1][:, -1]

	data_X = np.reshape(data_X,(data_X.shape[0],data_X.shape[1],n_caracteristicas))

	return [data_X,data_y]



