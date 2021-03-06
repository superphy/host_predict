import numpy as np
from numpy.random import seed
import sys
import os
from sklearn.externals import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif

seed(913824)

def encode_categories(data, class_dict):
	arry = np.array([], dtype = 'i4')
	for item in data:
		temp = str(item)
		temp = int(''.join(filter(str.isdigit, temp)))
		for index in range(len(class_dict)):
			check = class_dict[index]
			check = int(''.join(filter(str.isdigit, check)))
			#print(check)
			if temp == check:
				temp = index
		arry = np.append(arry,temp)
	#print(arry)
	return arry


if __name__ == "__main__":
	#######################################################
	# Creates a total of 5 sets of training/testing data
	# for each drug. Does feature selection using given
	# number of features.
	#
	# Call with "time python make_train_test.py <numfeats>"
	#######################################################

	NUM_FEATS = sys.argv[1] # defualt = 270

	df = joblib.load("amr_data/mic_class_dataframe.pkl") # Matrix of experimental MIC values
	mic_class_dict = joblib.load("amr_data/mic_class_order_dict.pkl") # Matrix of classes for each drug

	df_cols = df.columns
	for drug in df_cols:
		print("\n********************",drug,"*******************")
		num_classes = len(mic_class_dict[drug])

		matrix = np.load('amr_data/'+drug+'/kmer_matrix.npy')
		rows_mic = np.load('amr_data/'+drug+'/kmer_rows_mic.npy')
		rows_gen = np.load('amr_data/'+drug+'/kmer_rows_genomes.npy')

		X = SelectKBest(f_classif, k=int(NUM_FEATS)).fit_transform(matrix, rows_mic)
		Y = rows_mic
		Z = rows_gen

		cv = StratifiedKFold(n_splits=5, random_state=913824)

		if not os.path.exists('./amr_data/'+drug+'/'+str(NUM_FEATS)+'feats/'):
			os.mkdir('./amr_data/'+drug+'/'+str(NUM_FEATS)+'feats/')

		loop = 1
		for train,test in cv.split(X,Y,Z):

			filepath = './amr_data/'+drug+'/'+str(NUM_FEATS)+'feats/'

			Y[train] = encode_categories(Y[train], mic_class_dict[drug])
			Y[test]  = encode_categories(Y[test], mic_class_dict[drug])
			
			#y_train = to_categorical(Y[train], num_classes)
			#y_test  = to_categorical(Y[test], num_classes)
			### to_categorical for hyperas' data function, svm and xgboost etc dont need it (just neural net thing)

			y_train = Y[train]
			y_test = Y[test]
			x_train = X[train]
			x_test  =  X[test]

			filepath=filepath+'fold'+str(loop)+'/'
			if not os.path.exists(filepath):
				os.mkdir(filepath)

			np.save(filepath+'x_train.npy', x_train)
			np.save(filepath+'x_test.npy', x_test)
			np.save(filepath+'y_train.npy', y_train)
			np.save(filepath+'y_test.npy', y_test)
			np.save(filepath+'genome_train.npy', Z[train])
			np.save(filepath+'genome_test.npy', Z[test])

			loop+=1
