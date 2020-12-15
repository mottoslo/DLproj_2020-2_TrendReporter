
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from tensorflow.keras import models, layers, preprocessing, optimizers
from tensorflow.keras import losses
from tensorflow.keras.layers import Activation, Dense, Dropout, LSTM, Bidirectional
from sklearn import preprocessing as kprocessing
import keras.backend as K
from keras.callbacks import EarlyStopping
from tensorflow.compat.v1 import flags
FLAGS = flags.FLAGS

def del_all_flags(FLAG):
    flags_dict = FLAG._flags()
    keys_list = [keys for keys in flags_dict]
    
    for keys in keys_list:
        FLAGS.__delattr__(keys)
        
del_all_flags(FLAGS)

flags.DEFINE_string("f",'','kernel')

#parameters
flags.DEFINE_string("path", "./data/catgwise/생활+건강/", "path to data file")
flags.DEFINE_string("click_data", "clicks_ma_ratio", "clicks_minmax, clicks_first_ratio, clicks_ma_ratio")
flags.DEFINE_integer("s", 60, "seasonality")
flags.DEFINE_float("dropout", 0, "dropout rate(default=0)")
flags.DEFINE_integer("epoch", 40, "epoch")
flags.DEFINE_integer("batch_size", 1, "batch size")
flags.DEFINE_integer("pred_time", 30, "how much time to predict")
flags.DEFINE_string("pred_index", "05-01-2020", "when beginning prediction(month-date-year), default:'01-01-2020'")
flags.DEFINE_boolean("bi", True,"true if bidirectional")
def split_train_test(data, time="01-01-2020", click="clicks_minmax", temp=True, poll=True, sell=True):   #수정됨
    split_date = pd.Timestamp(time) 
    if temp:
        if poll:
            if sell:
                train = data.loc[:split_date, [click, 'clicks', 'kosis_catg_lv1_season', 'kosis_catg_lv2_season', 
                                               'kosis_catg_lv1_defla', 'kosis_catg_lv2_defla', 'temp_high', 'temp_low', 
                                               'temp_avg', 'pm10']]
                test = data.loc[split_date:, [click, 'clicks', 'kosis_catg_lv1_season', 'kosis_catg_lv2_season',
                                              'kosis_catg_lv1_defla', 'kosis_catg_lv2_defla', 'temp_high', 'temp_low',
                                               'temp_avg', 'pm10']]
                feature = [1,1,1]
            elif not sell:
                train = data.loc[:split_date, [click, 'clicks', 'temp_high', 'temp_low', 'temp_avg', 'pm10']]
                test = data.loc[split_date:, [click, 'clicks', 'temp_high', 'temp_low', 'temp_avg', 'pm10']]
                feature = [1,1,0]
        elif not poll:
            if sell:
                train = data.loc[:split_date, [click, 'clicks', 'kosis_catg_lv1_season', 'kosis_catg_lv2_season', 
                                               'kosis_catg_lv1_defla',
                                                 'kosis_catg_lv2_defla', 'temp_high', 'temp_low', 'temp_avg']]
                test = data.loc[split_date:, [click, 'clicks', 'kosis_catg_lv1_season', 'kosis_catg_lv2_season', 
                                              'kosis_catg_lv1_defla', 
                                                'kosis_catg_lv2_defla', 'temp_high', 'temp_low', 'temp_avg']]
                feature = [1,0,1]
            elif not sell:
                train = data.loc[:split_date, [click, 'clicks', 'temp_high', 'temp_low', 'temp_avg']]
                test = data.loc[split_date:, [click, 'clicks','temp_high', 'temp_low', 'temp_avg']]
                feature = [1,0,0]
    elif not temp:
        if poll:
            if sell:
                train = data.loc[:split_date, [click, 'clicks', 'kosis_catg_lv1_season', 'kosis_catg_lv2_season', 
                                               'kosis_catg_lv1_defla', 'kosis_catg_lv2_defla', 'pm10']]
                test = data.loc[split_date:, [click, 'clicks', 'kosis_catg_lv1_season', 'kosis_catg_lv2_season', 
                                              'kosis_catg_lv1_defla', 'kosis_catg_lv2_defla', 'pm10']]
                feature=[0,1,1]
            elif not sell:
                train = data.loc[:split_date, [click, 'clicks', 'pm10']]
                test = data.loc[split_date:, [click, 'clicks', 'pm10']]
                feature=[0,1,0]
        elif not poll:
            if sell:
                train = data.loc[:split_date, [click, 'clicks', 'kosis_catg_lv1_season', 'kosis_catg_lv2_season', 
                                               'kosis_catg_lv1_defla', 'kosis_catg_lv2_defla']]
                test = data.loc[split_date:, [click, 'clicks', 'kosis_catg_lv1_season', 'kosis_catg_lv2_season', 
                                              'kosis_catg_lv1_defla', 'kosis_catg_lv2_defla']]
                feature=[0,0,1]
            elif not sell:
                train = data.loc[:split_date, [click, 'clicks']]
                test = data.loc[split_date:, [click, 'clicks']]
                feature=[0,0,0]

    return train, test, feature

def preprocess(df, s=20, click="clicks_minmax"):  ##수정됨
    num_feature = df.shape[1]
    X=pd.DataFrame(index=df.index)
    y=pd.DataFrame(index=df.index)
    for column in df:
        for i in range(0, s):
            X['{}_{}'.format(column, i)] = df[column].shift(i)
        y[column] = df[column]
    X = X.dropna().values.reshape(-1, num_feature, s)
    y = y.iloc[s-1:].values.reshape(-1, num_feature, 1)

    return X, y, num_feature

def fitted_lstm(df, test, model, s=20, click='clicks_minmax'): 
    pred = model.predict(test)
    pred_df = pd.DataFrame(index=df.index, columns=df.columns)
    pred_df[:s] = df[:s]
    for i in range(0, pred.shape[0] - 1):
        pred_df.loc[pred_df.index[i + s]] = pred[i]
    return pred_df

def predict_lstm(data, model, s, start_index, pred_time, click, feature):  #수정됨
    temppollsell = [True, True, True]
    for i in range(3):
        if feature[i]==0:
            temppollsell[i] = False

    X, y, feature = split_train_test(data, time=start_index, click=click,
                                     temp=temppollsell[0], poll=temppollsell[1],
                                     sell=temppollsell[2])
    start_index = pd.Timestamp(start_index)
    pred_df = X[-s-1:-1]
    for i in range(0, pred_time):
        target = start_index + timedelta(days=i)
        pred = preprocess(pred_df[i:i+s],s=s, click=click)
        pred = model.predict(pred[0])
        pred_df = pred_df.append(pd.DataFrame(data = pred[[0]], index=[target], columns=pred_df.columns))

    return pred_df[start_index:]

def evaluate(test, predict, title, plot=True, figsize=(20, 13), s=20,filename=None): #수정됨
    try:
        test = test[s:]
        ## error
        predict["error"] = predict["clicks"] - predict["forecast"]
        predict["error_pct"] = predict["error"] / predict["clicks"]
        predict["error_rate"] = predict["error_pct"].apply(lambda x: np.abs(x))

        ## plot
        if plot == True:
            fig = plt.figure(figsize=figsize)
            fig.suptitle(title, fontsize=20)
            ax1 = fig.add_subplot(2, 2, 1)
            ax2 = fig.add_subplot(2, 2, 2)
            ax3 = fig.add_subplot(2, 2, 3)
            ax4 = fig.add_subplot(2, 2, 4)
            ### training
            test[["clicks","forecast"]].plot(
                color=["black", "green"], title="Test", grid=True, ax=ax1)
            ax1.set(xlabel=None)
            
            ### predict
            predict[["clicks", "forecast"]].plot(
                color=["black", "red"], title="Predict from {}".format(FLAGS.pred_index), grid=True, ax=ax2)
            ax2.set(xlabel=None)

            ### error
            predict[["error_rate"]].plot(ax=ax3, color=["green"], title="error", grid=True)
            ax3.set(xlabel=None)
            
            ### error distribution
            predict[["error_rate"]].plot(ax=ax4, color=["red"], kind='kde', title="error Distribution",
                                             grid=True)
            ax4.set(ylabel=None)

            #plt.show()
            if not os.path.exists("saved_fig"):
                os.mkdir("saved_fig")
            plt.savefig('saved_fig/{}_fig.png'.format(filename))
            #print("saved figure in saved_fig/{}.png".format(filename))


        return test[["clicks","forecast"]], predict[["clicks", "forecast", "error", "error_rate"]]


    except Exception as e:
        print("--- got error ---")
        print(e)

def fit_lstm(train, test, feature, click="clicks_minmax", s=20, figsize=(15, 5), bi=False): ##수정됨
    ## check
    '''
    print("Seasonality: using the last", s,
          "observations to predict the next {} from {}"
          .format(FLAGS.pred_time, FLAGS.pred_index))
    '''

    ## preprocess train
    X_train, y_train, num_feature = preprocess(train, s=s, click=click)
    X_test, y_test, num_feature = preprocess(test, s=s, click=click)


    ## model
    K.clear_session()
    model = models.Sequential()
    if bi:
        model.add(Bidirectional(LSTM(units=30, return_sequences=False),
                                       input_shape=(num_feature, s), merge_mode='ave') )
    else:    
        model.add(LSTM(input_shape=(num_feature, s), units=10, return_sequences=False) )
    model.add(Dropout(FLAGS.dropout))
    model.add(Dense(num_feature))
    #early_stopping = EarlyStopping(patience=5, verbose=1) 
    optimizer = optimizers.Adam(lr=0.01)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    #print(model.summary())

    ## train
    training = model.fit(X_train, y_train, batch_size=FLAGS.batch_size, epochs=FLAGS.epoch, shuffle=True) 
                         #validation_split=0.3, callbacks=[early_stopping])
    
    ## test
    pred_test = fitted_lstm(test, X_test, training.model, s=s, click=click)
    #print(pred_test)

    dtf_test = test.merge(
        pd.DataFrame(pred_test['clicks']).rename(columns={'clicks':'forecast'}),
        how='left', left_index=True, right_index=True)
    #print(dtf_test)

    ## prediction
    pred_check= predict_lstm(data, model, s=s,
                              start_index=FLAGS.pred_index, pred_time=FLAGS.pred_time,
                              click=click, feature=feature)
    pred_index = pd.Timestamp(FLAGS.pred_index)
    dtf_pred = test.loc[pred_index:pred_index + timedelta(days=FLAGS.pred_time - 1)]
    dtf_pred = dtf_pred.merge(pd.DataFrame(pred_check['clicks']).rename(columns={'clicks':'forecast'}),
                              how='left', left_index=True, right_index=True)
    #print(dtf_pred)

    ## evaluate
    test_data, predict_data = evaluate(test=dtf_test, predict=dtf_pred, figsize=figsize, s=s,
                                  title="LSTM (memory:" + str(s) + ") with feature {}".format(num_feature))
    return test_data, predict_data, training.model


def SAVE(category, dtf_test, dtf_predict, model):
    if not os.path.exists("saved_model"):
        os.mkdir("saved_model")
    model.save('saved_model/{}.h5'.format(category))
    #print("saved in saved_model/{}.h5".format(category))
    if not os.path.exists("saved_csv"):
        os.mkdir("saved_csv")
    dtf_test.to_csv('saved_csv/{}_test.csv'.format(category))
    #print("saved in saved_csv/{}_test.csv".format(category))
    dtf_predict.to_csv('saved_csv/{}_predict.csv'.format(category))
    #print("saved in saved_csv/{}_predict.csv".format(category))

catg_lst = os.listdir(FLAGS.path)
#temppollsell=[[True, True, True], [True, True, False], [True, False, True], [True, False, False],
                      #[False, True, True], [False, True, False], [False, False, True], [False,False,False]]
temppollsell=[[True, True, False], [True, False, False], [False, True, False], [False, False, False]]
for category in catg_lst:
    data_path = "{}{}".format(FLAGS.path, category)
    file = pd.read_csv(data_path, encoding='CP949')
    file['date'] = pd.to_datetime(file['date'])
    data = file.set_index('date')
    best_predict = pd.DataFrame(data = [100], columns=['error_rate'])
    for f in temppollsell:
        train, test, feature = split_train_test(data=data, time='01-01-2020', click=FLAGS.click_data,
                                                temp=f[0], poll=f[1], sell=f[2]) #수정됨
        dtf_test, dtf_predict, model = fit_lstm(train, test, feature, click=FLAGS.click_data, s=FLAGS.s, bi=FLAGS.bi)

        if dtf_predict['error_rate'].mean() < best_predict['error_rate'].mean():
            best_test = dtf_test
            best_predict = dtf_predict
            best_model = model
            best_feature = feature
    cat = category.replace(".csv", "best_{}{}{}_{}_bi".format(best_feature[0], best_feature[1], best_feature[2], FLAGS.click_data) )
    SAVE(category=cat, dtf_test=best_test, dtf_predict=best_predict, model=best_model)
    evaluate(test=dtf_test, predict=dtf_predict, figsize=(15, 5), s=FLAGS.s,filename=cat,
                                       title="LSTM (memory:" + str(FLAGS.s) + ") with feature {}".format(sum(best_feature)))