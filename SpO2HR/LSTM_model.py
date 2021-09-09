
import SpO2HR.get_data as get_data
import os

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout, Reshape

import matplotlib.pyplot as plt

n_future = get_data.n_future
x_train = get_data.x_train
y_train = get_data.y_train

model = Sequential()
model.add(LSTM(units=30, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=30, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=30, return_sequences=False))
model.add(Dense(n_future*y_train.shape[2]))
model.add(Reshape([n_future, y_train.shape[2]]))

model.compile(optimizer='adam', loss='mse')
model.summary()

checkpoint_path = "SpO2HR/log/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 save_best_only=True)

history = model.fit(x_train, y_train, epochs=100, batch_size=1,
                    validation_split=0.20, verbose=1, shuffle=False, callbacks=[cp_callback])

# Save the entire model to a HDF5 file
model.save('HR_SPO2_PredictionModel.h5')

plt.title('Loss')
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()
