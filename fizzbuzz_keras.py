import sys
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense
from keras.models import Model
from keras.utils import np_utils

def binary_encode(i, num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)])

def fizz_buzz_encode(i):
    if   i % 15 == 0: return np.array([0, 0, 0, 1])
    elif i % 5  == 0: return np.array([0, 0, 1, 0])
    elif i % 3  == 0: return np.array([0, 1, 0, 0])
    else:             return np.array([1, 0, 0, 0])

def fizz_buzz(i, prediction):
    return [str(i), "fizz", "buzz", "fizzbuzz"][prediction]

# 訓練用データ
NUM_DIGITS = 10
X_train = np.array([binary_encode(i, NUM_DIGITS) for i in range(101, 2 ** NUM_DIGITS)])
# 正解データ
Y_train = np.array([fizz_buzz_encode(i) for i in range(101, 2 ** NUM_DIGITS)])

# モデル

model = Sequential()
model.add(Dense(450, input_dim=10, activation="relu"))
model.add(Dense(450, activation="relu"))
model.add(Dense(4, activation="softmax"))
model.compile(loss='categorical_crossentropy',
			  optimizer='adam',
			  metrics=["accuracy"])
hist = model.fit(X_train, Y_train,
				epochs=100,
				batch_size=128)

model_json_str = model.to_json()
open('fizzbuzz_model.json', 'w').write(model_json_str)
model.save_weights('fizzbuzz_model.h5');

# テスト用データ
numbers = np.arange(1, 101)
X_test = np.transpose(binary_encode(numbers, NUM_DIGITS))

# 予測実行
Y_test = model.predict_classes(X_test)

print()
output = np.vectorize(fizz_buzz)(numbers, Y_test)
print(output)

# 正解表示
answer = np.array([])
for i in numbers:
    if i % 15 == 0: answer = np.append(answer, "fizzbuzz")
    elif i % 5 == 0: answer = np.append(answer, "buzz")
    elif i % 3 == 0: answer = np.append(answer, "fizz")
    else: answer = np.append(answer, str(i))
print(answer)

# 正解率
evaluate = np.array(answer == output)
print(np.count_nonzero(evaluate == True) / 100)

