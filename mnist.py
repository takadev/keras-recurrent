import sys
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, CSVLogger
import matplotlib.pyplot as plt

# Kerasに含まれるMNISTデータの取得
# 初回はダウンロードが発生するため時間がかかる
# X_trainは訓練用ベクトルデータ
# y_trainは訓練用正解データ
# X_testはテスト用ベクトルデータ
# y_testはテス用正解データ
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 配列の整形と、色の範囲を0-255 -> 0-1に変換
# 60000個の28行28列(28 x 28 = 784)の訓練データ
X_train = X_train.reshape(60000, 784)

# 10000個の28行28列(28 x 28 = 784)のテスト用データ
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# 正解データを数値からダミー変数の形式に変換
# これは例えば0, 1, 2, 3, 4の8個の値の分類の正解ラベル5件のデータが以下のような配列になってるとして
#   [0, 1, 2, 3, 4, 3, 1, 0]
# 以下のような形式に変換する
#   [[1, 0, 0, 0, 0],
#    [0, 1, 0, 0, 0],
#    [0, 0, 1, 0, 0],
#    [0, 0, 0, 1, 0],
#    [0, 0, 0, 0, 1],
#    [0, 0, 0, 1, 0],
#	 [0, 1, 0, 0, 0],
#	 [1, 0, 0, 0, 0]]
# 列方向が0, 1, 2, 3, 4、行方向が各データに対応し、元のデータで正解となる部分が1、それ以外が0となるように展開してる
Y_train = np_utils.to_categorical(y_train)
Y_test = np_utils.to_categorical(y_test)

# ネットワークの定義
# 各層や活性関数に該当するレイヤを順に入れていく
# 作成したあとにmodel.add()で追加する
model = Sequential()

# 隠れ層 1
# - ノード数：512
# - 入力：784次元
# - 活性化関数：relu
# - ドロップアウト比率：0.2
model.add(Dense(512, input_dim=784))
model.add(Activation('relu'))
model.add(Dropout(0.2))


# 隠れ層 2
# - ノード数：512
# - 活性化関数：relu
# - ドロップアウト比率：0.2
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))

# 出力層
# - ノード数：10
# - 活性化関数：softmax
model.add(Dense(10))
model.add(Activation('softmax'))

# モデルの要約を出力
model.summary()

# 学習過程の設定
# - 目的関数：categorical_crossentropy
# - 最適化アルゴリズム：rmsprop
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=2)
csv_logger = CSVLogger('training.log')

# 学習処理の実行
hist = model.fit(X_train, Y_train,
			batch_size=200,
			epochs=10,
			verbose=1,
			validation_data=(X_test, Y_test),
			callbacks=[es, csv_logger])

# 予測
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss :', score[0])
print('Test accuracy :', score[1])


# plot results
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
