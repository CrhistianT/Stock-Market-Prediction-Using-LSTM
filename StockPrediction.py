from data_processing import *
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import LSTM, CuDNNLSTM
from keras.optimizers import RMSprop

class StockPrediction():

	def __init__(self, input_dim, input_size):
		
		self.input_dim = input_dim
		self.input_size = input_size

		self.RMS = RMSprop()

		self.model = Sequential()

		self.model.add(CuDNNLSTM(256, input_shape=(input_size, input_dim), return_sequences=True))
		self.model.add(CuDNNLSTM(256, input_shape=(input_size, input_dim), return_sequences=True))
		self.model.add(CuDNNLSTM(256, input_shape=(input_size, input_dim)))
		self.model.add(Dense(1))
		self.model.add(Activation('softmax'))

		self.model.compile(loss='mse', optimizer=self.RMS)

		self.X = np.array([])
		self.Y = np.array([])
		self.X_test = np.array([])
		self.Y_test = np.array([])

	def LoadData(self, input_file, input_size):
		self.X, self.Y, self.X_test, self.Y_test = procesamiento_data(input_file, input_size)

	def Train(self, num_epochs=100):
		self.model.fit(self.X, self.Y, batch_size=64, epochs=num_epochs, validation_split=0.1, verbose=1)

	def Predict(self, test):
		P = self.model.predict(test)
		return P

	def Evaluate(self, Xtrain, Ytrain, Xtest, Ytest):
		trainEval = self.model.evaluate(Xtrain, Ytrain, verbose=0)
		print('Train Evaluation = ', trainEval[0])

		testEval = self.model.evaluate(Xtest, Ytest, verbose=0)
		print('Test Evaluation = ', testEval[0])

	def SaveModel(self, name='Stock'):
		self.model.save("Modelos/" + name)

	def LoadModel(self, name='Stock'):
		self.model = load_model("Modelos/" + name)

S = StockPrediction(5, 1000)
S.LoadData("daily_MSFT.csv", 1000)
S.Train(10)
P = S.Predict(S.X_test)
print(P)
S.Evaluate(S.X, S.Y, S.X_test, S.Y_test)
S.SaveModel()