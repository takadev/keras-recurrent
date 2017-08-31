# -*- coding: utf-8 -*-
import sys
import numpy

import pandas
from sklearn import preprocessing
from keras.datasets import mnist
from keras.models import model_from_json
from keras.utils import np_utils
from keras.optimizers import RMSprop

import matplotlib.pyplot as plt

def load_data(data, n_prev=10):
	X, Y = [], []
	for i in range(len(data) - n_prev):
		X.append(data.iloc[i:(i+n_prev)].as_matrix())
		Y.append(data.iloc[i+n_prev].as_matrix())
	retX = numpy.array(X)
	retY = numpy.array(Y)
	return retX, retY

# モデルを読み込む
model = model_from_json(open('stock_model.json').read())

# 学習結果を読み込む
model.load_weights('stock_model.h5')

model.summary();

model.compile(loss="mean_squared_error",
                  optimizer=RMSprop())

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

split_pos = int(len(data) * 0.8)
x_train, y_train = load_data(data[['close']].iloc[0:split_pos], 10)
x_test,  y_test  = load_data(data[['close']].iloc[split_pos:], 10)

predicted = model.predict(x_test)
result = pandas.DataFrame(predicted)
result.columns = ['predict']
result['actual'] = y_test
result.plot()
plt.grid()
plt.show()
