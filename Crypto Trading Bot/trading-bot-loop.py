import numpy as np
import pandas as pd
from sklearn import preprocessing
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from binance.client import Client
import time
import datetime
import pickle
print("done with the imports")


client = Client("***", "***")
train_x = pickle.load(open("train_x_.pickle", "rb"))

#Creamos un modelo basico de cuenta con simples metodos para "comprar" y "vender" un activo
class Account:
    def __init__(self, **kwargs):
        self.balance = {}
        prices = client.get_all_tickers()
        for coin in prices:
            self.balance[coin["symbol"]] = 0
        self.balance["USD"] = 0
        for symbol in kwargs:
            self.balance[symbol] = kwargs.get(symbol,0)
    
    @staticmethod
    def get_price(symbol):
        prices = client.get_all_tickers()
        for coin in prices:
            if coin["symbol"] == symbol:
                return coin["price"]
        
    
    def execute_buy(self, amount, symbol):
        price = self.get_price(symbol)
        self.balance["USD"] -= float(price) * float(amount)
        self.balance[symbol] += float(amount)
        
        
    def execute_sell(self, amount, symbol):
        price = self.get_price(symbol)
        self.balance["USD"] += float(price) * float(amount)
        self.balance[symbol] -= float(amount)



guille = Account(USD = 10000)

#Reunimos la tanda de datos iniciales
starter_data = []
btc_starter_data = []
ltc_starter_data = []
xrp_starter_data = []
eth_starter_data = []

klines = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1MINUTE, "10 Feb, 2019")[-60:]
for minute in klines:
    btc_starter_data.append([minute[4], minute[5]])

klines = client.get_historical_klines("LTCUSDT", Client.KLINE_INTERVAL_1MINUTE, "10 Feb, 2019")[-60:]
for minute in klines:
    ltc_starter_data.append([minute[4], minute[5]])

klines = client.get_historical_klines("XRPUSDT", Client.KLINE_INTERVAL_1MINUTE, "10 Feb, 2019")[-60:]
for minute in klines:
    xrp_starter_data.append([minute[4], minute[5]])

klines = client.get_historical_klines("ETHUSDT", Client.KLINE_INTERVAL_1MINUTE, "10 Feb, 2019")[-60:]
for minute in klines:
    eth_starter_data.append([minute[4], minute[5]])


for n in range(len(btc_starter_data)):
    starter_data.append(btc_starter_data[n]+ltc_starter_data[n]+xrp_starter_data[n]+eth_starter_data[n])

starter_data = np.reshape(np.array(starter_data), (1,60,8))


#creamos el modelo de RNN, que va a cargar los weights y biases de un entrenamiento previo
def create_model():
    model = Sequential()
    model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences = True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences = True))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())

    model.add(LSTM(128, input_shape=(train_x.shape[1:])))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(32, activation = "relu"))
    model.add(Dropout(0.2))

    model.add(Dense(2, activation="softmax"))

    opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

    model.compile(loss="sparse_categorical_crossentropy",
    	          optimizer = opt,
    	          metrics = ["accuracy"])

    return model

model = create_model()
model.load_weights(r"C:\Users\Familia\Bioquimica Guille\crypto_data\modelo_guardado.ckpt")


starttime = time.time()
x = list()
print("starting while loop")
#En este ciclo while, ejecutamos las ordenes en base a las predicciones del modelo y se junta nueva data al mismo
#tiempo que se descarta el punto mas viejo en starter_data
i = 0
while i < 45:
	klines_btc = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1MINUTE, "2 minutes ago UTC")[-1]
	klines_ltc = client.get_historical_klines("LTCUSDT", Client.KLINE_INTERVAL_1MINUTE, "2 minutes ago UTC")[-1]
	klines_xrp = client.get_historical_klines("XRPUSDT", Client.KLINE_INTERVAL_1MINUTE, "2 minutes ago UTC")[-1]
	klines_eth = client.get_historical_klines("ETHUSDT", Client.KLINE_INTERVAL_1MINUTE, "2 minutes ago UTC")[-1]
	now_data = np.array([[klines_btc[4], klines_btc[5]]+[klines_ltc[4], klines_ltc[5]]+[klines_xrp[4], klines_xrp[5]]+[klines_eth[4], klines_eth[5]]])
	if model.predict_classes(starter_data)[0] == 1:
		amnt = (guille.balance["USD"]/100)/float(guille.get_price("LTCUSDT"))
		guille.execute_buy(amnt, "LTCUSDT")
		print(f"buy executed, current balance is {guille.balance['LTCUSDT']} ltcs and {guille.balance['USD']} dollars")
		time.sleep(60.0 - ((time.time() - starttime) % 60.0))
		guille.execute_sell(amnt, "LTCUSDT")
		print(f"sell executed, current balance is {guille.balance['LTCUSDT']} ltcs and {guille.balance['USD']} dollars")
		now_data = np.reshape(now_data, (1,1,8))
		try:
			starter_data = np.append(starter_data, now_data, axis=1)
		except Exception as e:
			print(starter_data.shape, now_data.shape)
			raise(e)

		starter_data = np.delete(starter_data, 0, axis=1)
		
	elif model.predict_classes(starter_data)[0] == 0:
		now_data = np.reshape(now_data, (1,1,8))
		try:
			starter_data = np.append(starter_data, now_data, axis=1)
		except Exception as e:
			print(starter_data.shape, now_data.shape)
			raise(e)
		starter_data = np.delete(starter_data, 0, axis=1)
		print("No trade executed")
		
		time.sleep(60.0 - ((time.time() - starttime) % 60.0))

	
	print([guille.balance["USD"], guille.balance["LTCUSDT"]])
	x.append(guille.balance["USD"])
	i += 1

pickle.dump(x, open(f"{time.time}_run_data", "wb"))
	




