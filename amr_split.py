#!/usr/bin/env python

import numpy as np
import pandas as pd
from sklearn.externals import joblib
from math import floor


# Matrix of experimental MIC values
df = joblib.load("amr_data/mic_class_dataframe.pkl")
#df_rows = df.index.values 	# Row names are genomes
df_cols = df.columns		# Col names are drugs

for drug in df_cols:

	print("start: making train/test data for ", drug)

	matrix   = np.load('amr_data/'+drug+'/kmer_matrix.npy')
	rows_gen = np.load('amr_data/'+drug+'/kmer_rows_genomes.npy')
	rows_mic = np.load('amr_data/'+drug+'/kmer_rows_mic.npy')
	cols     = np.load('amr_data/'+drug+'/kmer_cols.npy')

	num_rows = len(rows_gen)

	## Create the training and testing data sets
	# Determine the size of the sets
	chunk = floor(num_rows/5)
	remainder = num_rows%5
	if remainder == 0 :
		train_size = chunk*4
		test_size  = chunk
	else:
		train_size = (chunk*4)+remainder
		test_size  = chunk

	# Create masks
	train_mask   = [1]*train_size
	train_mask_b = [0]*test_size
	train_mask   = train_mask + train_mask_b

	test_mask   = [0]*train_size
	test_mask_b = [1]*test_size
	test_mask   = test_mask + test_mask_b


	# Make data sets
	train_list = [bool(x) for x in train_mask]
	test_list  = [bool(x) for x in test_mask]
	#train_list  = np.array(train_list)
	#test_list   = np.array(test_list)

	#print(rows_mic.shape, len(train_mask))

	train_data  = matrix[train_list, :]
	train_names = rows_mic[train_list]
	test_data   = matrix[test_list, :]
	test_names  = rows_mic[test_list]

	print(matrix.shape)
	print(train_data.shape)
	print(test_data.shape)

	#print(test_names)
	#print(test_data)

	#print(matrix.shape)
	#print(len(train_mask))

	np.save('amr_data/'+drug+'/train_data.npy', train_data)
	np.save('amr_data/'+drug+'/train_names.npy', train_names)
	np.save('amr_data/'+drug+'/test_data.npy', test_data)
	np.save('amr_data/'+drug+'/test_names.npy', test_names)

	print("end: making train/test data for ",drug)