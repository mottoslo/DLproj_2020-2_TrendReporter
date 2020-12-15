import split_train_test
import fit_lstm
import evaluate
import SAVE
import make_prediction
from tensorflow.compat.v1 import flags
import pandas as pd
import os

flags.DEFINE_string("path", "./data/catgwise/생활+건강/", "path to data file")
flags.DEFINE_string("click_data", "clicks_ma_ratio", "clicks_minmax, clicks_first_ratio, clicks_ma_ratio")
flags.DEFINE_integer("s", 60, "seasonality")
flags.DEFINE_float("dropout", 0, "dropout rate(default=0)")
flags.DEFINE_integer("epoch", 40, "epoch")
flags.DEFINE_integer("batch_size", 1, "batch size")
flags.DEFINE_integer("pred_time", 30, "how much time to predict")
flags.DEFINE_string("pred_index", "05-01-2020", "when beginning prediction(month-date-year), default:'01-01-2020'")
flags.DEFINE_boolean("bi", True,"true if bidirectional")

FLAGS = flags.FLAGS


catg_lst = os.listdir(FLAGS.path)
#temppollsell=[[True, True, True], [True, True, False], [True, False, True], [True, False, False],
                      #[False, True, True], [False, True, False], [False, False, True], [False,False,False]]
temppollsell=[[True, True, False], [True, False, False],
                      [False, True, False], [False,False,False]]
for category in catg_lst:
    data_path = "{}{}".format(FLAGS.path, category)
    file = pd.read_csv(data_path, encoding='CP949')
    file['date'] = pd.to_datetime(file['date'])
    data = file.set_index('date')
    best_predict = pd.DataFrame(data = [100], columns=['error_rate'])
    for f in temppollsell:
        train, test, feature = split_train_test(data=data, time='01-01-2020', click=FLAGS.click_data,
                                                temp=f[0], poll=f[1], sell=f[2]) #수정됨
        dtf_test, dtf_predict, model = fit_lstm(train, test, feature, click=FLAGS.click_data, s=FLAGS.s, bi=FLAGS.bi,
                                                FLAGS=FLAGS)

        if dtf_predict['error_rate'].mean() < best_predict['error_rate'].mean():
            best_test = dtf_test
            best_predict = dtf_predict
            best_model = model
            best_feature = feature
    cat = category.replace(".csv", "best_{}{}{}_{}_bi".format(best_feature[0], best_feature[1], best_feature[2], FLAGS.click_data) )
    SAVE(category=cat, dtf_test=best_test, dtf_predict=best_predict, model=best_model)
    evaluate(test=dtf_test, predict=dtf_predict, figsize=(15, 5), s=FLAGS.s,filename=cat,
                                       title="LSTM (memory:" + str(FLAGS.s) + ") with feature {}".format(sum(best_feature)))