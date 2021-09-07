import pandas as pd
from pandas import DataFrame
import os
import re

data_dir = './data'
data = pd.DataFrame()
list = os.listdir(data_dir)
list.sort(key=lambda f: int(re.sub('\D', '', f)))

for fname in os.listdir(data_dir):
    path = os.path.join(data_dir, fname)
    df = pd.read_csv(path)
    data =pd.concat([data, df], ignore_index=True)


base_dir = './FinalData'
if not os.path.exists(base_dir):
    os.mkdir(base_dir)

data.to_csv(os.path.join(base_dir,'SpO2HR.csv'))