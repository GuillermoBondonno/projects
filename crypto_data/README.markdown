Un simple bot para practicar keras,

crypto_data: 4 archivos csv como datos historicos para el entrenamiento del modelo

deep-net.py: procesa los archivos en crypto_data  y genera starter_data.pickle, luego
define el modelo y corre train.
El modelo toma valores historicos de las 4 coins y devuelve la prediccion para el proximo punto

trading_bot_loop.py: toma cotizaciones en vivo y maneja los datos para pasarlos por el modelo
a su vez, genera una clase "cuenta", para manejar los balances y las ganancias o perdidas.
