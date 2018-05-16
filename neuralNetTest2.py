#!/usr/bin/env python
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.layers.convolutional import Conv1D
from keras.utils import to_categorical
import lmdb
import numpy as np
import tensorflow

kmermatrix = np.load('kmermatrix.npy')


def make3D(data):
    data = data.reshape(data.shape[0], 1 ,data.shape[1])
    return data

def make_name_3D(data):
    data = data.reshape(27,1,1)
    return data

train_names = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1])

train_list       = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0]
bovine_test_list = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
human_test_list  = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1]
train_list1 = [bool(x) for x in train_list]
bovine_test_list1 = [bool(x) for x in bovine_test_list]
human_test_list1 = [bool(x) for x in human_test_list]

train_mask = np.array(train_list1)
bovine_mask = np.array(bovine_test_list1)
human_mask = np.array(human_test_list1)

train_data = kmermatrix[train_mask, :]
bovine_test = kmermatrix[bovine_mask, :]
human_test = kmermatrix[human_mask, :]

if len(train_data.shape) == 2:
        train_data = make3D(train_data)
        bovine_test = make3D(bovine_test)
        human_test = make3D(human_test)

#train_names=make_name_3D(train_names)

train_names = to_categorical(train_names, 2)

print(train_data.shape)
print(train_names.shape)

"""
model = Sequential()
#model.add(Conv1D(filters=10, kernel_size=1, activation='relu', input_shape=(1,train_data.shape[2])))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(train_data, train_names, epochs =25, verbose=1)
"""

model = Sequential()
model.add(Conv1D(filters=10, kernel_size=1, activation='relu', input_shape=(1,train_data.shape[2])))
model.add(Flatten())
model.add(Dense(12,input_dim=3634442,kernel_initializer='uniform',activation='relu'))
model.add(Dense(8,kernel_initializer='uniform',activation='relu'))
model.add(Dense(2,kernel_initializer='uniform',activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(train_data, train_names, epochs =50, verbose=1)

print(model.predict(bovine_test))
print(model.predict(human_test))
print(model.predict_classes(bovine_test))
print(model.predict_classes(human_test))
