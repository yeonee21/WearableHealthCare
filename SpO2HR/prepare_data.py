import pandas as pd
import os
import natsort

data_dir = './data'
data = pd.DataFrame()
list = os.listdir(data_dir)
list = [file for file in list if file.endswith(".csv")]
list = natsort.natsorted(list)

for fname in list:
    path = os.path.join(data_dir, fname)
    df = pd.read_csv(path)
    data =pd.concat([data, df], ignore_index=True)

data = data[0::60] # 1분 간격으로 샘플링

base_dir = './FinalData'
if not os.path.exists(base_dir):
    os.mkdir(base_dir)

data.to_csv(os.path.join(base_dir,'SpO2HR.csv'))