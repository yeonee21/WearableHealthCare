import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('./FinalData/SpO2HR.csv')

dataset = df[['HeartRate', 'SpO2']].astype(float)