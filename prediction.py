#!/usr/bin/env python

from sklearn import svm
from sklearn.neural_network import MLPClassifier
import numpy as np
import lmdb


if __name__ == "__main__":
	"""
	Notes
		-this script is for the specific given input, will have to be changed if original data is altered.
		-should rotate the testing and training sets
		-could use a mask to create the training and testing sets, instead of iterating through

	Resources:
		SVM: http://scikit-learn.org/stable/modules/svm.html
			 http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
		MLP: http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier
	"""

	# Load the data
	kmermatrix = np.load('kmermatrix.npy')

	X_train = []	# Samples for training
	X_test = []		# Samples for testing
	for i in range(33):
		if (i>=0 and i<=13) or (i>=17 and i<=29):	# All but the last 3 of each set (human and bovine) will be for training
			X_train.append(kmermatrix[i])
		else:										# Last three of each set are for testing
			X_test.append(kmermatrix[i])

	# Class labels for the training set; dont need Y_test because that's the predicted output
	Y_train = []
	for i in range(14):
		Y_train.append('bovine')
	for i in range(13):
		Y_train.append('human')


	##### SVM - 5/6 correct
	#clf = svm.SVC(decision_function_shape='ovo', kernel='linear')
	
	##### Neural Network - 0/6 correct
	#clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
	
	###### Neural Network - 4/6 correct
	#clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)

	###### Neural Network - 5/6 correct
	clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(20,), random_state=1)

	###### Neural Network - seg fault
	#clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(25,), random_state=1)

	###### Neural Network - 5/6 correct
	#clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,15), random_state=1)

	print(clf.fit(X_train, Y_train))
	print(clf.predict(X_test))


