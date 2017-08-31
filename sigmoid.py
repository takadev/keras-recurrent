# -*- coding: utf-8 -*-
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation

x = np.array([
    [0,1,2],
    [1,3,1],
    [3,1,-1],
    [5,2,0],
    [0,8,0]
  ])

y = np.array([
    [1,0,0],
    [1,0,0],
    [0,1,0],
    [0,0,1],
    [0,1,0]
  ])


Test_NN = Sequential()
Test_NN.add(Dense(4, input_dim=3))
Test_NN.add(Activation('sigmoid'))
Test_NN.add(Dense(3))
Test_NN.add(Activation('softmax'))
Test_NN.summary()
Test_NN.compile(optimizer='adam',
                loss='categorical_crossentropy')
Test_NN.fit(x, y, epochs=500)
Test_NN.predict(x)

