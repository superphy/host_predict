#!/usr/bin/env python
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.layers.convolutional import Conv1D, Conv2D
from keras.utils import to_categorical
from keras.layers.advanced_activations import LeakyReLU, PReLU
import lmdb
import numpy as np
import tensorflow
from sklearn import svm
from sklearn.neural_network import MLPClassifier


# Define a random seeed for consistency
np.random.seed(123)

# Load in the matrix  
kmermatrix = np.load('kmermatrix.npy')

# Names for training; 0=bovine, 1=human
train_names = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1])

# Will use a mask to extract the training and testing data from the matrix
train_list       = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,  1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0]
bovine_test_list = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
human_test_list  = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1]

train_list1 = [bool(x) for x in train_list]
bovine_test_list1 = [bool(x) for x in bovine_test_list]
human_test_list1 = [bool(x) for x in human_test_list]

train_mask = np.array(train_list1)
bovine_mask = np.array(bovine_test_list1)
human_mask = np.array(human_test_list1)

# Set of training and testing data
train_data = kmermatrix[train_mask, :]
bovine_test = kmermatrix[bovine_mask, :]
human_test = kmermatrix[human_mask, :]

# Two categories, bovine (0) & human (1)
train_names = to_categorical(train_names, 2)

# Make the model
model = Sequential()
model.add(Dense(12,input_dim=3634442,kernel_initializer='uniform',activation='relu'))
model.add(Dense(8,kernel_initializer='uniform',activation='relu'))
model.add(Dense(2,kernel_initializer='uniform',activation='sigmoid'))


model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(train_data, train_names, epochs=20, batch_size=10, verbose=1)
#model.fit(train_data, train_names, epochs=50, batch_size=10, verbose=1)
#model.fit(train_data, train_names, epochs=25, batch_size=5, verbose=1)

print("\nKeras")
print("predict bov test class\n",model.predict_classes(bovine_test))
#print("predict bov test proba\n",model.predict_proba(bovine_test))
print("predict hum test class\n",model.predict_classes(human_test))
#print("predict hum test proba\n",model.predict_proba(human_test))


# Adapted from prediction.py
# Included to compare with the keras results

X_train = train_data	# Samples for training

# Class labels for the training set; dont need Y_test because that's the predicted output
Y_train = []
for i in range(14):
	Y_train.append('bovine')
for i in range(13):
	Y_train.append('human')


##### SVM
svmclf = svm.SVC(decision_function_shape='ovo', kernel='linear')
	
###### MLP
mlpclf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(20,), random_state=1)

print("\nSVM")
svmclf.fit(X_train, Y_train)
print(svmclf.predict(bovine_test))
print(svmclf.predict(human_test))

print("\nMLP")
mlpclf.fit(X_train, Y_train)
print(mlpclf.predict(bovine_test))
print(mlpclf.predict(human_test))