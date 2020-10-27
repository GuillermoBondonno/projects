import pandas as pd
from sklearn import preprocessing
from collections import deque
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import time

import os



main_df = pd.DataFrame()




SEQ_LEN = 60
RATIO_TO_PREDICT = "LTCUSDT"
FUTURE_PERIOD_PREDICT = 3
EPOCHS = 1
BATCH_SIZE = 64
NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"




def preprocess_df(df):
    df = df.drop("future",1)
    
    for col in df.columns:
        if col != "target":
            df[col].replace(to_replace=0, value=1, inplace=True)
            df[col] = df[col].pct_change()
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values)
    df.dropna(inplace = True)

    
    
    sequential_data = []
    prev_days = deque(maxlen = SEQ_LEN)

    for i in df.values:
    	prev_days.append([n for n in i[:-1]])
    	#con ese -1 no se incluye a target
    	if len(prev_days) == SEQ_LEN:
    		sequential_data.append([np.array(prev_days), i[-1]])

    random.shuffle(sequential_data)

    buys = []
    sells = []

    for seq, target in sequential_data:
    	if target == 0:
    		sells.append([seq, target])
    	elif target == 1:
    		buys.append([seq, target])

    random.shuffle(buys)
    random.shuffle(sells)

    lower = min(len(buys), len(sells))

    buys = buys[:lower]
    sells = sells[:lower]

    sequential_data = buys + sells

    random.shuffle(sequential_data)

    X = []
    y = []

    for seq, target in sequential_data:
    	X.append(seq)
    	y.append(target)

    return np.array(X), y


# In[4]:


main_df = pd.DataFrame()
ratios = ["BTCUSDT","LTCUSDT", "XRPUSDT", "ETHUSDT"]
for ratio in ratios:
    dataset = r"C:\Users\Familia\Bioquimica Guille\crypto_data\crypto_data\{}.csv".format(ratio)
    df = pd.read_csv(dataset, names = ["time","close","volume"])
    df.rename(columns={"close": f"{ratio}_close", "volume": f"{ratio}_volume"}, inplace=True)
    df.set_index("time", inplace=True)
    df = df[[f"{ratio}_close", f"{ratio}_volume"]]
    
    
    if len(main_df) == 0:
        main_df = df
    else:
        main_df = main_df.join(df)
    


def classify(current, future):
    if float(future)/float(current) >1.0015 :
        return 1
    else:
        return 0



main_df['future'] = main_df[f'{RATIO_TO_PREDICT}_close'].shift(-FUTURE_PERIOD_PREDICT)



main_df['target'] = list(map(classify, main_df[f'{RATIO_TO_PREDICT}_close'], main_df['future']))


times = sorted(main_df.index.values)



last_5pc = times[-int(0.05*len(times))]



validation_main_df = main_df[(main_df.index >= last_5pc)]
main_df = main_df[(main_df.index < last_5pc)]





train_x, train_y = preprocess_df(main_df)

validation_x, validation_y = preprocess_df(validation_main_df)




print(f"train data: {len(train_x)} validation: {len(validation_x)}")
print(f"Dont buys: {train_y.count(0)}, buys: {train_y.count(1)}")
print(f"VALIDATION Dont buys: {validation_y.count(0)}, buys: {validation_y.count(1)}")



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

tensorboard = TensorBoard(log_dir=f"logs/{NAME}")

filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')) # saves only the best ones

checkpoint_path = r"C:\Users\Familia\Bioquimica Guille\crypto_data\modelo_guardado.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                 save_weights_only=True,
                                                 verbose=1)

history = model.fit(train_x,
	train_y,
	batch_size = BATCH_SIZE,
	epochs = EPOCHS,
	validation_data = (validation_x,validation_y))#,
	#callbacks=[tensorboard, checkpoint, cp_callback])


	