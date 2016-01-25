from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
import helper.dt_utils as du

maxlen=1024

# build the model: 2 stacked LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(512, return_sequences=True, input_shape=(maxlen,)))
model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(54))
model.add(Activation('softmax'))

model.compile(loss='mean_squared_error', optimizer='rmsprop')

train_data=du.laod_pose()
i, o = train_data[1]
model.fit(i, o, batch_size=1, nb_epoch=1)