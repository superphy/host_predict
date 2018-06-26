#!/usr/bin/env python

import numpy as np
from numpy.random import seed
#from tabulate import tabulate
#from beautifultable import BeautifulTable
#from prettytable import PrettyTable

import pandas as pd
from pandas import DataFrame

from decimal import Decimal

import tensorflow
from tensorflow import set_random_seed

#from concurrent.futures import ProcessPoolExecutor
#from multiprocessing import cpu_count

#from hyperopt import Trials, STATUS_OK, tpe
#from keras.layers.convolutional import Conv1D
from keras.layers.core import Dense, Dropout, Activation
#from keras.layers import Flatten, BatchNormalization
from keras.models import Sequential, load_model
from keras.utils import np_utils, to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

#from hyperas import optim
#from hyperas.distributions import choice, uniform, conditional

from sklearn.externals import joblib
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import matthews_corrcoef, classification_report, precision_recall_fscore_support, confusion_matrix

import sys


seed(913824)
set_random_seed(913824)


def decode_categories(data, class_dict):
	'''
	Given a set of bin numbers (data), and a set of classes (class_dict),
	translates the bins into classes.
	Eg. goes from [0,1,2] into ['<=1,2,>=4']
	'''
	arry = np.array([])
	for item in data:
		arry = np.append(arry,class_dict[item])
	return arry


def encode_categories(data, class_dict):
	'''
	Given a set of bin numbers (data), and a set of classes (class_dict),
	translates the classes into bins.
	Eg. goes from ['<=1,2,>=4'] into [0,1,2] 
	'''
	arry = np.array([], dtype = 'i4')
	for item in data:
		temp = str(item)
		temp = int(''.join(filter(str.isdigit, temp)))
		for index in range(len(class_dict)):
			check = class_dict[index]
			check = int(''.join(filter(str.isdigit, check)))
			if temp == check:
				temp = index
		arry = np.append(arry,temp)
	return arry


def eval_model(model, test_data, test_names):
    '''
    Takes a model (neural net), a set of test data, and a set of test names.
    Returns perc: the precent of correct guesses by the model using a windown of size 1.
    Returns mcc: the matthews correlation coefficient.
    Returns prediction and actual.
    '''
    # Create and save the prediction from the model
    prediction = model.predict_classes(test_data)
    #np.save('prediction.npy', prediction)

    # Reformat the true test data into the same format as the predicted data
    actual = []
    for row in range(test_names.shape[0]):
        for col in range(test_names.shape[1]):
            if(test_names[row,col]!=0):
                actual = np.append(actual,col)

    # Sum the number of correct guesses using a window: if the bin is one to either
    # side of the true bin, it is considered correct
    total_count = 0
    correct_count = 0
    for i in range(len(prediction)):
        total_count +=1
        pred = prediction[i]
        act = actual[i]
        if pred==act or pred==act+1 or pred==act-1:
            correct_count+=1
    # Calculate the percent of correct guesses
    perc = (correct_count*100)/total_count
    perc = Decimal(perc)
    perc = round(perc,2)

    #print("When allowing the model to guess MIC values that are next to the correct value:")
    #print("This model correctly predicted mic values for {} out of {} genomes ({}%).".format(correct_count,total_count,perc))
    #print("\nMCC: ", matthews_corrcoef(np.argmax(to_categorical(actual),axis=1),(prediction)))

    # Find the matthew's coefficient
    mcc = matthews_corrcoef(np.argmax(to_categorical(actual),axis=1),(prediction))
    return (perc, mcc, prediction, actual)


if __name__ == "__main__":
	df = joblib.load("amr_data/mic_class_dataframe.pkl") # Matrix of experimental MIC values
	mic_class_dict = joblib.load("amr_data/mic_class_order_dict.pkl") # Matrix of classes for each drug

	#	drug="AMP"
	# Perform the prediction for each drug
	df_cols = df.columns
	for drug in df_cols:
		print("\n********************",drug,"*******************")
		num_classes = len(mic_class_dict[drug])

		matrix = np.load('amr_data/'+drug+'/kmer_matrix.npy')
		rows_mic = np.load('amr_data/'+drug+'/kmer_rows_mic.npy')

		X = SelectKBest(f_classif, k=270).fit_transform(matrix, rows_mic)
		Y = rows_mic

		cv = StratifiedKFold(n_splits=5, random_state=913824)

		cvscores = []
		window_scores = []
		mcc_scores = []
		report_scores = []
		conf_scores = []

		for train,test in cv.split(X,Y):
			Y[train] = encode_categories(Y[train], mic_class_dict[drug])
			Y[test]  = encode_categories(Y[test], mic_class_dict[drug])
			y_train = to_categorical(Y[train], num_classes)
			y_test  = to_categorical(Y[test], num_classes)
			x_train = X[train]
			x_test =  X[test]

			patience = 16
			early_stop = EarlyStopping(monitor='loss', patience=patience, verbose=0, min_delta=0.005, mode='auto')
			model_save = ModelCheckpoint("best_model.hdf5",monitor='loss', verbose = 0, save_best_only =True, save_weights_only = False, mode ='auto', period =1)
			reduce_LR = ReduceLROnPlateau(monitor='loss', factor= 0.1, patience=(patience/2), verbose = 0, min_delta=0.005,mode = 'auto', cooldown=0, min_lr=0)

			model = Sequential()
			model.add(Dense(270,activation='relu',input_dim=(270)))
			#model.add(Dropout(0.5))
			#model.add(Dense(int((270+num_classes)/2), activation='relu', kernel_initializer='uniform'))
			#model.add(Dense(40, activation='relu', kernel_initializer='uniform'))
			#model.add(Dropout(0.5))
			model.add(Dense(num_classes, kernel_initializer='uniform', activation='softmax'))
			model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
			model.fit(x_train, y_train, epochs=50, verbose=0, callbacks=[early_stop, reduce_LR])

			scores = model.evaluate(x_test, y_test, verbose=0)

			#print('Test accuracy:', acc)
			results = eval_model(model, x_test, y_test)
			window_scores.append(results[0])
			mcc_scores.append(results[1])

			labels = np.arange(0,num_classes)
			report = precision_recall_fscore_support(results[3], results[2], average=None, labels=labels)
			report_scores.append(report)

			conf = confusion_matrix(results[3], results[2], labels=labels)
			conf_scores.append(conf)
			#print(conf)

			#print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
			cvscores.append(scores[1] * 100)
		print("Avg base acc:   %.2f%%   (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
		print("Avg window acc: %.2f%%   (+/- %.2f%%)" % (np.mean(window_scores), np.std(window_scores)))
		print("Avg base mcc:   %f (+/- %f)" % (np.mean(mcc_scores), np.std(mcc_scores)))
		
		#########################################################################
		## Average the classification reports from the 5 splits into one report.
		np.set_printoptions(suppress=True) # Prevent printing in scientific notation
		avg_reports = np.mean(report_scores,axis=0) # Average the results

		# This makes a transpose of what we want to display, so we will flip it:
		#   Make a dict of key = column names and value = column contents
		#   Then load it into a pandas dataframe (df)
		col_headers = ["precision", "recall", "f1-score", "support"] 
		cols = {}
		for i in range(len(avg_reports)):
			cols[col_headers[i]]=avg_reports[i]
		report_df = DataFrame(cols, index=mic_class_dict[drug])

		# Add a new row that contains the average for each column
		new_row = []
		for header in col_headers:
			new_row.append(report_df[header].mean())
		report_df.loc["avg"] = new_row

		report_df = np.round(report_df,decimals=2)
		print("\n Classification Report (avg)")
		with pd.option_context('display.max_rows', 15, 'display.max_columns', 15): print(report_df)
		#########################################################################

		#########################################################################
		## Average the confusion matrices from the 5 splits into one matrix
		avg_confs = np.mean(conf_scores, axis=0) # Average the results
		conf_df = DataFrame(avg_confs, index=mic_class_dict[drug]) # Turn the results into a pandas dataframe (df)
		conf_df.set_axis(mic_class_dict[drug], axis='columns', inplace=True) # Label the axis
		print("\n Confusion Matrix (avg)")
		with pd.option_context('display.max_rows', 15, 'display.max_columns', 15): print(conf_df) # Print all columns of the df
		#########################################################################

		#########################################################################
		## Save all of the results because we can't do plots on panther.
		## Everything is saved into a single numpy for each drug; loading and
		## plotting is done with plot_results.py.
		all_results = []
		all_results.append([np.mean(cvscores), np.std(cvscores)])
		all_results.append([np.mean(window_scores), np.std(window_scores)])
		all_results.append([np.mean(mcc_scores), np.std(mcc_scores)])
		all_results.append([mic_class_dict[drug]])
		all_results.append([avg_reports])
		all_results.append([avg_confs])
		np.save('amr_data/'+drug+'/all_results.npy', all_results)
		#########################################################################



#	best_run, best_model = optim.minimize(model=create_model, data=data, algo=tpe.suggest, max_evals=100, trials=Trials())#
#	score = best_model.evaluate(test_data, test_names)
#	print(sys.argv[1],"features. Evaluation of best performing model:", score)
#	print(sys.argv[1],"features. Best performing model chosen hyper-parameters:", best_run)
