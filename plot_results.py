

import numpy as np
from decimal import Decimal

import itertools

import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots_adjust

from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib

import matplotlib
# Make sure that we are using QT5
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets, QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


class ScrollableWindow(QtWidgets.QMainWindow):
    def __init__(self, fig):
        self.qapp = QtWidgets.QApplication([])

        QtWidgets.QMainWindow.__init__(self)
        self.widget = QtWidgets.QWidget()
        self.setCentralWidget(self.widget)
        self.widget.setLayout(QtWidgets.QGridLayout())
        self.widget.layout().setContentsMargins(0,0,0,0)
        self.widget.layout().setSpacing(0)

        self.fig = fig
        self.canvas = FigureCanvas(self.fig)
        self.canvas.draw()
        self.scroll = QtWidgets.QScrollArea(self.widget)
        self.scroll.setWidget(self.canvas)

        self.nav = NavigationToolbar(self.canvas, self.widget)
        self.widget.layout().addWidget(self.nav)
        self.widget.layout().addWidget(self.scroll)

        self.fig.subplots_adjust(0.125, 0.1, 0.9, 0.9) # left,bottom,right,top 

        self.show()
        exit(self.qapp.exec_()) 


def plot_confusion_matrix(row,col,ind, cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix. Normalization can be applied by setting `normalize=True`.
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    ax = fig.add_subplot(row,col,ind)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar(im,fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' #if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout(h_pad=5)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


if __name__ == "__main__":

	#drug = "AMP"

	df = joblib.load("amr_data/mic_class_dataframe.pkl") # Matrix of experimental MIC values
	mic_class_dict = joblib.load("amr_data/mic_class_order_dict.pkl") # Matrix of classes for each drug

	num_rows  = 7
	num_cols = 4
	plt_index = 1

	fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, sharex=True, sharey=True, figsize=(50,50))

	#plt.subplots_adjust(left=0.125, bottom=None, right=None, top=None, wspace=0.7, hspace=0.5)

	df_cols = df.columns
	for drug in df_cols:
		all_results = np.load('amr_data/'+drug+'/all_results.npy')

		base_acc = all_results[0][0]
		base_std = all_results[0][1]

		wind_acc = all_results[1][0]
		wind_std = all_results[1][1]

		mcc_acc  = all_results[2][0]
		mcc_std  = all_results[2][1]

		classes  = all_results[3]
		classes = tuple(item for item in classes[0])

		avg_reports = all_results[4]
		avg_confs   = all_results[5]


		np.set_printoptions(suppress=True)

		#print(base_acc, base_std)
		#print(wind_acc, wind_std)
		#print(mcc_acc, mcc_std)
		#print(classes)
		#print(avg_reports)
		#print(avg_confs[0])

		# Plot non-normalized confusion matrix
		plt.subplot(num_rows,num_cols,plt_index)
		plot_confusion_matrix(num_rows,num_cols,plt_index, avg_confs[0], classes=classes, title=drug+': Confusion matrix, without normalization')
		plt_index+=1

		# Plot normalized confusion matrix
		plt.subplot(num_rows,num_cols,plt_index)
		plot_confusion_matrix(num_rows,num_cols,plt_index, avg_confs[0], classes=classes, normalize=True, title=drug+': Normalized confusion matrix')
		plt_index+=1

		#plt.subplots_adjust(left=0.125, bottom=None, right=None, top=None, wspace=0.7, hspace=0.5)

		#plt.hold(True)

		if drug=="TIO":break

	#plt.show()
	#plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
	#plt.tight_layout()
	#plt.show()
	a = ScrollableWindow(fig)