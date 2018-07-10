#!/usr/bin/env python

import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import StratifiedKFold
from math import floor

def decode_categories(data, class_dict):
	#print(class_dict)
	arry = np.array([])
	for item in data:
		arry = np.append(arry,class_dict[item])
	#print(arry)
	return arry


def encode_categories(data, class_dict):
	arry = np.array([], dtype = 'i4')
	for item in data:
		temp = int(float((item.decode('utf-8')).split("=")[-1]))
		# 0=1, 1=2, 2=4, 3=8, 4=16, 5=32
		# CHANGE ALL OF THIS CODE TO AN ENCODER
		for index in range(len(class_dict)):
			check = (int(float(class_dict[index].split("=")[-1])))
			#print(temp, check)
			if temp == check:
				temp = index
			#else:
				#print("AHHHHHHHHHHHHHHHHHHHHHHHHH")
		arry = np.append(arry,temp)
	print(arry)
	return arry


if __name__ == "__main__":
	# Matrix of experimental MIC values
	df = joblib.load("amr_data/mic_class_dataframe.pkl")

	# Matrix of classes for each drug
	mic_class_dict = joblib.load("amr_data/mic_class_order_dict.pkl")

	df_cols = df.columns # Col names are drugs
	for drug in df_cols:

		print("start: making train/test data for ", drug)

		num_classes = len(mic_class_dict[drug])

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

		train_data  = matrix[train_list, :]
		train_names = rows_mic[train_list]
		test_data   = matrix[test_list, :]
		test_names  = rows_mic[test_list]

		print(matrix.shape)
		print(train_data.shape)
		print(test_data.shape)

		np.save('amr_data/'+drug+'/train_data.npy', train_data)
		np.save('amr_data/'+drug+'/train_names.npy', train_names)
		np.save('amr_data/'+drug+'/test_data.npy', test_data)
		np.save('amr_data/'+drug+'/test_names.npy', test_names)

		print("end: making train/test data for ",drug)


