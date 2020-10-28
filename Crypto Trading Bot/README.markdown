Un simple bot para practicar keras,

crypto_data: 4 archivos csv como datos historicos para el entrenamiento del modelo

deep-net.py: procesa los archivos en crypto_data  y genera starter_data.pickle, luego
define el modelo y corre train.
El modelo toma valores historicos de las 4 coins y devuelve la prediccion para el proximo punto

trading_bot_loop.py: toma cotizaciones en vivo y maneja los datos para pasarlos por el modelo
a su vez, genera una clase "cuenta", para manejar los balances y las ganancias o perdidas.

##########################################################################################

A simple script to practice Keras,

crypto_data: 4 csv files that contain historical data

deep-net.py: It processes the files in crypto_data and generates starter_data.pickle, then it builds
the model and trains it. The model takes historical data from the 4 coins and returns a prediction

trading_bot_loop.py: takes live price data from the binance client and processes the data to feed it 
to the model. Also, here a simple Account class is defined to handle balance and transactions
