from data_processing import *
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import LSTM, CuDNNLSTM
from keras.optimizers import RMSprop, Adam

class StockPrediction():

	def __init__(self, input_dim, seq_size):
		
		d = 0.3

		self.input_dim = input_dim
		self.seq_size = seq_size

		self.Opt = Adam(decay=0.2)

		self.model = Sequential()

		self.model.add(CuDNNLSTM(256, input_shape=(seq_size, input_dim), return_sequences=True))
		self.model.add(Dropout(d))
		self.model.add(CuDNNLSTM(256, input_shape=(seq_size, input_dim), return_sequences=True))
		self.model.add(Dropout(d))
		self.model.add(CuDNNLSTM(256, input_shape=(seq_size, input_dim), return_sequences=False))
		self.model.add(Dropout(d))
		self.model.add(Dense(32, kernel_initializer="uniform", activation='relu'))
		self.model.add(Dense(1, kernel_initializer="uniform", activation='linear'))
		self.model.add(Activation('softmax'))

		self.model.compile(loss='mse', optimizer=self.Opt)

		self.X = np.array([])
		self.Y = np.array([])
		self.X_test = np.array([])
		self.Y_test = np.array([])

	def LoadData(self, input_file):
		self.X, self.Y, self.X_test, self.Y_test = procesamiento_data(input_file, self.seq_size)

	def Train(self, num_epochs=100):
		self.model.fit(self.X, self.Y, batch_size=64, epochs=num_epochs, validation_split=0.1, verbose=1)

	def Predict(self, test):
		P = self.model.predict(test)
		return P

	def Evaluate(self, Xtrain, Ytrain, Xtest, Ytest):
		trainEval = self.model.evaluate(Xtrain, Ytrain, verbose=0)
		print('Train Evaluation:')
		print(trainEval)

		testEval = self.model.evaluate(Xtest, Ytest, verbose=0)
		print('Test Evaluation:')
		print(testEval)

	def SaveModel(self, name='Stock'):
		self.model.save("Modelos/" + name)

	def LoadModel(self, name='Stock'):
		self.model = load_model("Modelos/" + name)

S = StockPrediction(5, 30)
S.LoadData("daily_MSFT.csv")
S.Train(100)
P = S.Predict(S.X_test)
print(P)
S.Evaluate(S.X, S.Y, S.X_test, S.Y_test)
S.SaveModel()