import pandas as pd
from datetime import timedelta
import os
import numpy as np

time_test = pd.Timestamp("11-13-2019") + timedelta(1)
pred_length = 29
k = 0

path = "D:/Trend_reporter/"
dat = pd.read_csv(path + "output2.csv", encoding = "CP949")
dat["date"] = pd.to_datetime(dat["date"])
df = dat.set_index("date")

df1 = df.loc[time_test - timedelta(365) : time_test,]
df2 = df.loc[time_test:time_test+timedelta(365),]
temp = pd.DataFrame(df2.values - df1.values)

df1_1 = df.iloc[-pred_length-366:-366,]
df2_2 = df.iloc[-pred_length:,]
temp2 = pd.DataFrame(df2_2.values - df1_1.values)

UL = []
for i in range(temp.shape[1]):
    UL.append(temp.iloc[:, i].mean() + k * temp.iloc[:, i].std())

idx = []
for i in range(len(UL)):
    idx.append(temp2.iloc[:, i][temp2.iloc[:, i].values > UL[i]].sum())
idx

sorted_idx = []
for i in range(10):
    sorted_idx.append(np.where(sorted(idx, reverse=True)[i] == idx))

for i in range(10):
    print(df.columns[sorted_idx[i][0][0]])