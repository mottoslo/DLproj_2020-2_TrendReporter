import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models
from datetime import timedelta
import predict_lstm as predict_lstm
import os
import feature_select

pred_index = "11-13-2020"
model_path = "D:/Trend_reporter/saved_model/"
data_path = "D:/Trend_reporter/data/catgwise/생활+건강/"
s = 60
pred_time = 30

temp = pd.DataFrame()



for d in os.listdir(data_path):
    orig = pd.read_csv(data_path + d)
    orig["date"] = pd.to_datetime(orig["date"])
    orig_dat = orig.set_index("date")
    #print(orig_dat.iloc[:,0:1])
    for m in os.listdir(model_path):
        if "{}".format(d.replace(".csv","")) in m and "best" in m:
            feature = feature_select.feature_select(m)
            model = models.load_model(model_path + m)
            pred_df = predict_lstm.predict_lstm(orig_dat, model, s, pred_index, pred_time, "clicks_ma_ratio",feature)
            del pred_df["clicks_ma_ratio"]
            hap = orig_dat.iloc[:, 0:1].append(pred_df.iloc[1:, 0:1])
            hap.columns = [d]
            if temp.empty:
                temp = hap
            else:
                temp = temp.join(hap,how="left")

temp.to_csv("output2.csv",encoding="cp949")
