#!/usr/bin/env python

import numpy as np
from numpy.random import seed

import pandas as pd
from pandas import DataFrame
from decimal import Decimal
from xgboost import XGBClassifier
import sys
import os
import pickle

import tensorflow
from tensorflow import set_random_seed

#from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential#, load_model
from keras.utils import np_utils, to_categorical
#from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.externals import joblib
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, RandomizedSearchCV, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import matthews_corrcoef, classification_report, precision_recall_fscore_support, confusion_matrix

from hpsklearn import HyperoptEstimator, svc, xgboost_classification
from hyperopt import tpe

import warnings
warnings.filterwarnings('ignore')


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
	prediction = model.predict(test_data)
	prediction = [int(round(float(value))) for value in prediction]
	#np.save('prediction.npy', prediction)

	actual = test_names
	actual = [int(float(value)) for value in actual]
    # Sum the number of correct guesses using a window: if the bin is one to either
    # side of the true bin, it is considered correct
	total_count = 0
	correct_count = 0
	for i in range(len(prediction)):
		total_count +=1
		pred = prediction[i]
		act = actual[i]
		if pred==act:
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

def eval_modelOBO(model, test_data, test_names):
	'''
	Takes a model (neural net), a set of test data, and a set of test names.
	Returns perc: the precent of correct guesses by the model using a windown of size 1.
	Returns mcc: the matthews correlation coefficient.
	Returns prediction and actual.
	'''
	# Create and save the prediction from the model
	prediction = model.predict(test_data)
	prediction = [int(round(float(value))) for value in prediction]
	#np.save('prediction.npy', prediction)

	actual = test_names
	actual = [int(float(value)) for value in actual]
    # Sum the number of correct guesses using a window: if the bin is one to either
    # side of the true bin, it is considered correct
	#print("actual: ", actual)
	#print("predict: ", prediction)
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
	#mcc = 1
	return (perc, mcc, prediction, actual)

def find_major(pred, act, drug, mic_class_dict):
	class_dict = mic_class_dict[drug]
	pred = class_dict[pred]
	act  = class_dict[int(act)]
	pred = (str(pred).split("=")[-1])
	pred = ((pred).split(">")[-1])
	pred = ((pred).split("<")[-1])
	pred = int(round(float(pred)))
	act = (str(act).split("=")[-1])
	act = ((act).split(">")[-1])
	act = ((act).split("<")[-1])
	act = int(round(float(act)))

	if(drug =='AMC' or drug == 'AMP' or drug =='CHL' or drug =='FOX'):
		susc = 8
		resist = 32
	if(drug == 'AZM' or drug == 'NAL'):
		susc = 16
	if(drug == 'CIP'):
		susc = 0.06
		resist = 1
	if(drug == 'CRO'):
		susc = 1
	if(drug == 'FIS'):
		susc = 256
		resist = 512
	if(drug == 'GEN' or drug =='TET'):
		susc = 4
		resist = 16
	if(drug == 'SXT' or drug =='TIO'):
		susc = 2

	if(drug == 'AZM' or drug == 'NAL'):
		resist = 32
	if(drug == 'CRO' or drug == 'SXT'):
		resist = 4
	if(drug == 'TIO'):
		resist = 8

	if(pred <= susc and act >= resist):
		return "VeryMajorError"
	if(pred >= resist and act <= susc):
		return "MajorError"
	return "NonMajor"


def find_errors(model, test_data, test_names, genome_names, class_dict, drug, mic_class_dict):
	if not os.path.exists('./amr_data/errors'):
		os.mkdir('./amr_data/errors')
	err_file = open('./amr_data/errors/'+str(sys.argv[1])+'_feats_svm_errors.txt', 'a+')

	prediction = model.predict(test_data)
	prediction = [int(round(float(value))) for value in prediction]
	actual = [int(float(value)) for value in test_names]

	total_count = 0
	wrong_count = 0
	close_count = 0
	off_by_one = False
	for i in range(len(prediction)):
		total_count +=1
		pred = prediction[i]
		act = actual[i]
		if (pred == act):
			continue
		else:
			if (pred==act+1 or pred==act-1):
				close_count+=1
				off_by_one = True
			else:
				off_by_one = False
			wrong_count+=1
			err_file.write("Drug:{} Genome:{} Predicted:{} Actual:{} OBO:{} Major?:{}\n".format(drug, genome_names[i].decode('utf-8'), class_dict[pred], class_dict[int(act)], off_by_one, find_major(pred,act,drug,mic_class_dict)))


if __name__ == "__main__":
	##################################################################
	# call with
	#	time python svm_test2.py <feats> <drug> <fold>
	# to do all folds
	#	for i in {1..5}; do python svm_test2.py <numfeats> <drug> "$i"; done
	#	sbatch -c 16 --mem 80G --wrap='for i in {1..5}; do python svm_test2.py <numfeats> <drug> "$i"; done'
	##################################################################

	num_feats = sys.argv[1]
	drug = sys.argv[2]
	fold = sys.argv[3]

	df = joblib.load("amr_data/mic_class_dataframe.pkl") # Matrix of experimental MIC values
	mic_class_dict = joblib.load("amr_data/mic_class_order_dict.pkl") # Matrix of classes for each drug

	print("\n****************",drug,"***************")
	num_classes = len(mic_class_dict[drug])

	matrix = np.load('amr_data/'+drug+'/kmer_matrix.npy')
	rows_mic = np.load('amr_data/'+drug+'/kmer_rows_mic.npy')
	rows_gen = np.load('amr_data/'+drug+'/kmer_rows_genomes.npy')

	filepath = './amr_data/'+drug+'/'+str(num_feats)+'feats/fold'+str(fold)+'/'
	x_train = np.load(filepath+'x_train.npy')
	x_test  = np.load(filepath+'x_test.npy')
	y_train = np.load(filepath+'y_train.npy')
	y_test  = np.load(filepath+'y_test.npy')



	model = HyperoptEstimator(classifier=svc("mySVC"), preprocessing=[], algo=tpe.suggest, max_evals=100, trial_timeout=120)
	model.fit(x_train, y_train)


	best_model = model.best_model()


	score = eval_model(model, x_test, y_test)
	wind_score = eval_modelOBO(model, x_test, y_test)
	y_true = wind_score[3]
	y_pred = wind_score[2]
	conf = confusion_matrix(y_true, y_pred)
	report = classification_report(y_true, y_pred)

	filepath = './amr_data/'+drug+'/'+str(num_feats)+'feats/fold'+str(fold)+'/'
	pickle.dump(model, open(filepath+'xgb_model.dat', 'wb'))

	mic_class_dict = joblib.load("amr_data/mic_class_order_dict.pkl")
	class_dict = mic_class_dict[drug]
	genome_names = np.load('./amr_data/'+drug+'/'+str(num_feats)+'feats/fold'+fold+'/genome_test.npy')
	find_errors(model, x_test, y_test, genome_names, class_dict, drug, mic_class_dict)

	# make the confusion matrix pretty for printing
	#conf_df = DataFrame(conf, index=mic_class_dict[drug]) # Turn the results into a pandas dataframe (df)
	#conf_df.set_axis(mic_class_dict[drug], axis='columns', inplace=True) # Label the axis

	with open(filepath+'svm_out.txt','w') as f:
		f.write("\nBase acc: {0}%\n".format(score[0]))
		f.write("Window acc: {0}%\n".format(wind_score[0]))
		f.write("MCC: {0}\n".format(round(wind_score[1],4)))
		f.write("\nConfusion Matrix\n{0}\n".format(conf))
		#f.write("\nConfusion Matrix\n{0}\n".format(conf_df))		
		f.write("\nClassification Report\n{0}\n".format(report))
		f.write("Best performing model chosen hyper-parameters:\n{0}".format(best_model))

