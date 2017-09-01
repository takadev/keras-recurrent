# -*- coding: utf-8 -*-
import sys
import numpy as np
from keras.models import model_from_json

def binary_encode(i, num_digits):
	return np.array([i >> d & 1 for d in range(num_digits)])

def fizz_buzz(i, prediction):
    return [str(i), "fizz", "buzz", "fizzbuzz"][prediction]

# モデルを読み込む
model = model_from_json(open('fizzbuzz_model.json').read())

# 学習結果を読み込む
model.load_weights('fizzbuzz_model.h5')
model.summary();
model.compile(loss='categorical_crossentropy',
			  optimizer='adam',
			  metrics=["accuracy"])

# テストデータ
NUM_DIGITS = 10
numbers = np.arange(101, 200)
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
