# -*- coding: utf-8 -*-
import sys
import numpy
import pandas
import matplotlib.pyplot as plt

from sklearn import preprocessing
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, CSVLogger

class Prediction :

  def __init__(self):
    self.length_of_sequences = 10
    self.in_out_neurons = 1
    self.hidden_neurons = 300

  def load_data(self, data, n_prev=10):
    X, Y = [], []
    for i in range(len(data) - n_prev):
      X.append(data.iloc[i:(i+n_prev)].as_matrix())
      Y.append(data.iloc[i+n_prev].as_matrix())
    retX = numpy.array(X)
    retY = numpy.array(Y)
    return retX, retY

  def create_model(self) :
    model = Sequential()
    lstm = LSTM(self.hidden_neurons,
                batch_input_shape=(None,
                                    self.length_of_sequences,
                                    self.in_out_neurons),
                return_sequences=False)

    model.add(lstm)
    model.add(Dense(self.in_out_neurons))
    model.add(Activation("linear"))
    model.compile(loss="mean_squared_error",
                  optimizer=RMSprop())
    return model


  def train(self, X_train, y_train) :
    model = self.create_model()
    csv_logger = CSVLogger('stock_train.log')
    # 学習
    hist = model.fit(X_train, y_train,
              batch_size=10,
              epochs=100,
              validation_split=0.2,
              callbacks=[csv_logger])

    return model, hist

if __name__ == "__main__":

  prediction = Prediction()

  # データ準備
  data = None
  for year in range(2013, 2017):
    data_ = pandas.read_csv('csv/stocks_1376-T_1d_' + str(year) +  '.csv', header=None)
    data = data_ if (data is None) else pandas.concat([data, data_])

  data.columns = ['date', 'open', 'high', 'low', 'close', 'yield', 'sales_value']
  data['date'] = pandas.to_datetime(data['date'], format='%Y-%m-%d')

  # 終値のデータを標準化
  data['close'] = preprocessing.scale(data['close'])
  data = data.sort_values(by='date')
  data = data.reset_index(drop=True)
  data = data.loc[:, ['date', 'close']]

  # 2割をテストデータへ
  split_pos = int(len(data) * 0.8)
  x_train, y_train = prediction.load_data(data[['close']].iloc[0:split_pos], prediction.length_of_sequences)
  x_test,  y_test  = prediction.load_data(data[['close']].iloc[split_pos:], prediction.length_of_sequences)

  (model, hist) = prediction.train(x_train, y_train)
  model_json_str = model.to_json()
  open('stock_model.json', 'w').write(model_json_str)
  model.save_weights('stock_model.h5');

  loss = hist.history['loss']
  val_loss = hist.history['val_loss']

  epochs = len(loss)
  plt.plot(range(epochs), loss, marker='.', label='loss')
  plt.plot(range(epochs), val_loss, marker='.', label='val_acc')
  plt.legend(loc='best', fontsize=10)
  plt.grid()
  plt.xlabel('epoch')
  plt.ylabel('loss')
  plt.show()

  # predicted = model.predict(x_test)
  # result = pandas.DataFrame(predicted)
  # result.columns = ['predict']
  # result['actual'] = y_test
  # result.plot()


