import math
import numpy as np

from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import SpO2HR.get_data as get_data

n_future = get_data.n_future
n_past = get_data.n_past

dataset = np.array(get_data.dataset)  # unscaled
x_train = get_data.x_train
y_train = get_data.y_train
x_test = get_data.x_test
y_test = get_data.y_test

scaler = get_data.scaler

# load model
model = load_model('SPO2HR/HR_SPO2_PredictionModel.h5')

# Train Prediction
trainPredict = model.predict(x_train)
trainPredict = np.reshape(trainPredict, (-1, trainPredict.shape[2]))

# Test Prediction
testPredict = model.predict(x_test)
testPredict = np.reshape(testPredict, (-1, testPredict.shape[2]))

# Unscale values
trainPredict = scaler.inverse_transform(trainPredict)
testPredict = scaler.inverse_transform(testPredict)

y_train = scaler.inverse_transform(y_train.reshape(-1, y_train.shape[2]))
y_test = scaler.inverse_transform(y_test.reshape(-1, y_test.shape[2]))

# Evaluate train/test scores
HRtrainScore = math.sqrt(mean_squared_error(y_train[:, 0], trainPredict[:, 0]))
print('HR Train Score: %.2f RMSE' % (HRtrainScore))
HRtestScore = math.sqrt(mean_squared_error(y_test[:, 0], testPredict[:, 0]))
print('HR Test Score: %.2f RMSE' % (HRtestScore))

SPO2trainScore = math.sqrt(mean_squared_error(y_train[:, 1], trainPredict[:, 1]))
print('SPO2 Train Score: %.2f RMSE' % (SPO2trainScore))
SPO2testScore = math.sqrt(mean_squared_error(y_test[:, 1], testPredict[:, 1]))
print('SPO2 Test Score: %.2f RMSE' % (SPO2testScore))

# Plot Predictions
a_axis = np.arange(n_past, len(y_train)+n_past)
b_axis = np.arange(len(y_train)+2*n_past, len(y_train)+len(y_test)+2*n_past)

plt.figure(figsize=(16,10))
plt.subplot(211)
plt.plot(dataset[:, 0], color='orange', alpha=0.7, label='Real')
plt.plot(a_axis, trainPredict[:, 0], color='green', label='train predict')
plt.plot(b_axis, testPredict[:, 0], color='red', label='test predict')
plt.title('Heart Rate LSTM')
plt.legend()

plt.subplot(212)
plt.plot(dataset[:, 1], color='orange', alpha=0.7, label='Real')
plt.plot(a_axis, trainPredict[:, 1], color='green', label='train predict')
plt.plot(b_axis, testPredict[:, 1], color='red', label='test predict')
plt.title('SPO2 LSTM')
plt.legend()

