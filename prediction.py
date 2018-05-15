#!/usr/bin/env python

from sklearn import svm
from sklearn.neural_network import MLPClassifier
import numpy as np
import lmdb


if __name__ == "__main__":
	"""
	Note this script is for the specific given input, will have to be changed if original
	data is altered.

	Resources:
	http://scikit-learn.org/stable/modules/svm.html
	http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
	"""

	# Load the data
	kmermatrix = np.load('kmermatrix.npy')

	# training samples
	X_train = []
	X_test = []
	for i in range(33):
		if (i>=0 and i<=13) or (i>=17 and i<=29):
			X_train.append(kmermatrix[i])
		else:
			X_test.append(kmermatrix[i])

	# Y = class labels = genomes
	Y_train = []
	for i in range(14):
		Y_train.append('bovine')
	for i in range(13):
		Y_train.append('human')

	clf = svm.SVC(decision_function_shape='ovo', kernel='linear')
	
	print(clf.fit(X_train, Y_train))

	print(clf.predict(X_test))


