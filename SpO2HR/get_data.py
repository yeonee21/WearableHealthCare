import numpy
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

np.random.seed(7)

dataset = pd.read_csv('./FinalData/SpO2HR.csv')
dataset = dataset[:720] # 12시간
dataset = dataset[['HeartRate', 'SpO2']].astype(float)

# Normalize dataset
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset.values.reshape(-1, 2))

# Split train/test data
train_dataset = numpy.array(scaled_data[:600]) #600분=10시간
test_dataset = numpy.array(scaled_data[600:]) #120분=2시간

# train dataset
x_train = []
y_train = []

n_past = 29  # 몇 분을 보고 예측할 것인지
n_future = 1  # 1분 뒤의 데이터를 예측

for i in range(n_past, len(train_dataset)-n_future+1):
    x_train.append(train_dataset[i-n_past:i, 0:train_dataset.shape[1]])
    y_train.append(train_dataset[i:i+n_future, 0:train_dataset.shape[1]])

x_train, y_train = np.array(x_train), np.array(y_train)


# test dataset
x_test = []
y_test = []

for i in range(n_past, len(test_dataset)- n_future+1):
    x_test.append(test_dataset[i-n_past:i, 0:test_dataset.shape[1]])
    y_test.append(test_dataset[i+n_future-1:i+n_future, 0:test_dataset.shape[1]])

x_test, y_test = np.array(x_test), np.array(y_test)





