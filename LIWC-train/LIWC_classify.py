# -*- coding: utf-8 -*-
# __author__: Yixuan LI
# __email__: yl2363@cornell.edu

import numpy as np
import os
from sklearn import svm
import matplotlib.pyplot as plt
import pickle
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
import pylab as pl
import seaborn as sns





class Trainer:
	def __init__(self,inputDir_geo,inputDir_non_geo,outputDir):
		self.inputDir_geo = inputDir_geo
		self.inputDir_non_geo = inputDir_non_geo
		self.outputDir = outputDir

		self.lineCount  = 1
		self.IDcount = 0
		self.id = []

		# number of users in each group
		self.maxLine = 545
		#self.maxLine = 803
		# number of LIWC features
		self.LIWC_attribute = 68
		# initialization of feature matrix of geo-locators
		self.data_geo = np.zeros((self.maxLine,self.LIWC_attribute))
		# initialize the labels of geo locators as 0s
		self.label_geo = np.zeros(self.maxLine,dtype='int')
		# initialization of feature matrix of non-locators
		self.data_non_geo = np.zeros((self.maxLine,self.LIWC_attribute))
		# initialize the labels of non-locators as 1s
		self.label_non_geo = np.ones(self.maxLine,dtype='int')
		
		self.LIWC = None
		self.clf_SVM = None
		self.clf_SGD = None
		self.clf_tree = None
		self.clf_GNB = None
		self.clf_KNN = None

		# the percentage of data kept for training. Remaining kept for testing.
		self.portion = 0.75
		self.portion = 0.9
		#self.portion = 0.5


	def read_data_geo(self):
		with open(self.inputDir_geo,'r') as fin:
			indexLine = 0
			for line in fin:
				if self.lineCount == 1:
					self.LIWC = line.split('\t')[1:][:-1]
				if self.lineCount > 1:    # skip the header
					values = line.split('\t')[1:][:-1]
					values = [float(s) for s in values]   # convert string to float				
					self.data_geo[indexLine,:] = values
					indexLine += 1
					if indexLine == self.maxLine:
						break
				self.lineCount += 1
		print "Finish reading geo locators' data"

	
	def read_data_non_geo(self):
		self.lineCount = 1
		with open(self.inputDir_non_geo,'r') as fin:
			indexLine = 0
			for line in fin:
				if self.lineCount > 1:    # skip the header
					values = line.split('\t')[1:][:-1]
					values = [float(s) for s in values]   # convert string to float
					
					self.data_non_geo[indexLine,:] = values
					indexLine += 1
					if indexLine == self.maxLine:
						break
				self.lineCount += 1
		print "Finish reading non-locators' data"


	def train_data(self):
		np.random.seed(40)
		#np.random.seed(35)
		order = np.random.permutation(self.maxLine * 2)
		X = np.concatenate((self.data_geo, self.data_non_geo), axis=0)
		Y = np.concatenate((self.label_geo,self.label_non_geo))		

		#fig = plt.figure(1, figsize=(8, 6))
		#X_reduced = PCA(n_components=2).fit_transform(X[:,2:])
		#plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=Y, cmap=plt.cm.Paired)
		#plt.show()

		# random permute the dataset
		# http://scikit-learn.org/stable/_downloads/plot_iris_exercise.py
		X = X[order]
		Y = Y[order]

		# select the best k features
		# http://scikit-learn.org/stable/modules/feature_selection.html#feature-selection
		X = SelectKBest(chi2, k=45).fit_transform(X, Y)
		

		# http://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html
		# PCA (3-dimension) visualization
		fig = plt.figure(1, figsize=(8, 6))
		ax = Axes3D(fig, elev=-150, azim=110)
		X_reduced = PCA(n_components=3).fit_transform(X[:,:])
		ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=Y, cmap=plt.cm.Paired)
		ax.set_title("First three PCA directions of LIWC feature space")
		ax.set_xlabel("1st eigenvector")
		ax.w_xaxis.set_ticklabels([])
		ax.set_ylabel("2nd eigenvector")
		ax.w_yaxis.set_ticklabels([])
		ax.set_zlabel("3rd eigenvector")
		ax.w_zaxis.set_ticklabels([])

		# show plot
		plt.show()
		
		# total number of data samples
		n_sample = len(X)

		# 10-fold cross validation
		# split the dataset into training set and test set. 
		X_train = X[:self.portion * n_sample]
		Y_train = Y[:self.portion * n_sample]
		
		X_test = X[self.portion * n_sample:]
		Y_test = Y[self.portion * n_sample:]


		# standardization the dataset into Gaussians with zero means and unit variances.
		# http://scikit-learn.org/stable/modules/preprocessing.html#preprocessing
		scaler = StandardScaler()
		scaler.fit(X_train)  # Don't cheat - fit only on training data
		X_train = scaler.transform(X_train)
		X_test = scaler.transform(X_test)  # apply same transformation to test data

		# fit the model using SVM with rbf kernel
		self.clf_SVM = svm.SVC(gamma=0.1, probability = True, kernel='rbf',C=2.)
		#self.clf_SVM = svm.SVC()
		self.clf_SVM.fit(X_train, Y_train)
		probas_SVM = self.clf_SVM.fit(X_train, Y_train).predict_proba(X_test)
		#print probas_SVM
	
		# prediction
		# http://scikit-learn.org/stable/tutorial/basic/tutorial.html#learning-and-predicting
		s = pickle.dumps(self.clf_SVM)
		clf2 = pickle.loads(s)
	
		diff_train_SVM = clf2.predict(X_train) - Y_train
		diff_test_SVM = clf2.predict(X_test) - Y_test
		print "----------------------------------------------------\n"
		print "training accuracy with SVM:",  1 - np.nonzero(diff_train_SVM)[0].size / (self.portion * n_sample)
		print "testing accuracy with SVM:", 1 - np.nonzero(diff_test_SVM)[0].size / ((1-self.portion) * n_sample)
		
		# Compute ROC curve and area the curve
		fpr_SVM, tpr_SVM, thresholds_SVM = roc_curve(Y_test, probas_SVM[:,1])
		roc_auc_SVM = auc(fpr_SVM, tpr_SVM)
		print "Area under the ROC curve : %f" % roc_auc_SVM

		
		
		# fit the model using k nearest neighbor
		# http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier
		self.clf_KNN = KNeighborsClassifier(n_neighbors=3)
		self.clf_KNN = self.clf_KNN.fit(X_train, Y_train)
		probas_KNN = self.clf_KNN.fit(X_train, Y_train).predict_proba(X_test)
		
		# prediction
		diff_train_KNN = self.clf_KNN.predict(X_train) - Y_train
		diff_test_KNN = self.clf_KNN.predict(X_test) - Y_test
		print "----------------------------------------------------\n"
		print "training accuracy with k nearest neighbors:",  1 - np.nonzero(diff_train_KNN)[0].size / (self.portion * n_sample)
		print "testing accuracy with k nearest neighbors:", 1 - np.nonzero(diff_test_KNN)[0].size / ((1-self.portion) * n_sample)

		# Compute ROC curve and area the curve
		fpr_KNN, tpr_KNN, thresholds_KNN = roc_curve(Y_test, probas_KNN[:,1])
		roc_auc_KNN = auc(fpr_KNN, tpr_KNN)
		print "Area under the ROC curve : %f" % roc_auc_KNN

		# fit the model using stochastic gradient descent
		# http://scikit-learn.org/stable/modules/sgd.html#classification
		self.clf_SGD = SGDClassifier(loss="modified_huber", penalty="l2")
		self.clf_SGD.fit(X_train, Y_train)

		# prediction using SGD model
		diff_train_SGD = self.clf_SGD.predict(X_train) - Y_train
		diff_test_SGD = self.clf_SGD.predict(X_test) - Y_test
		print "----------------------------------------------------\n"
		print "training accuracy with SGD:",  1 - np.nonzero(diff_train_SGD)[0].size / (self.portion * n_sample)
		print "testing accuracy with SGD:", 1 - np.nonzero(diff_test_SGD)[0].size / ((1-self.portion) * n_sample)
		

		# fit the model using decision tree
		# http://scikit-learn.org/stable/modules/tree.html#classification
		self.clf_tree = tree.DecisionTreeClassifier()
		self.clf_tree = self.clf_tree.fit(X_train, Y_train)
		probas_tree = self.clf_tree.fit(X_train, Y_train).predict_proba(X_test)
		#print probas_tree

		# prediction
		diff_train_tree = self.clf_tree.predict(X_train) - Y_train
		diff_test_tree = self.clf_tree.predict(X_test) - Y_test
		print "----------------------------------------------------\n"
		print "training accuracy with decision tree:",  1 - np.nonzero(diff_train_tree)[0].size / (self.portion * n_sample)
		print "testing accuracy with tree:", 1 - np.nonzero(diff_test_tree)[0].size / ((1-self.portion) * n_sample)
	
		# Compute ROC curve and area the curve
		fpr_tree, tpr_tree, thresholds_tree = roc_curve(Y_test, probas_tree[:,1])
		roc_auc_tree = auc(fpr_tree, tpr_tree)
		print "Area under the ROC curve : %f" % roc_auc_tree

		# fit the model using Gaussian Naive Bayes
		# http://scikit-learn.org/stable/modules/naive_bayes.html#gaussian-naive-bayes
		self.clf_GNB = GaussianNB()
		self.clf_GNB = self.clf_GNB.fit(X_train, Y_train)
		probas_GNB = self.clf_GNB.fit(X_train, Y_train).predict_proba(X_test)
		# prediction
		diff_train_GNB = self.clf_GNB.predict(X_train) - Y_train
		diff_test_GNB = self.clf_GNB.predict(X_test) - Y_test
		print "----------------------------------------------------\n"
		print "training accuracy with naive bayes:",  1 - np.nonzero(diff_train_GNB)[0].size / (self.portion * n_sample)
		print "testing accuracy with naive bayes:", 1 - np.nonzero(diff_test_GNB)[0].size / ((1-self.portion) * n_sample)
		
		# Compute ROC curve and area the curve
		fpr_GNB, tpr_GNB, thresholds_GNB = roc_curve(Y_test, probas_GNB[:,1])
		roc_auc_GNB = auc(fpr_GNB, tpr_GNB)
		print "Area under the ROC curve : %f" % roc_auc_GNB

		# Plot ROC curve
		
		rc={'font.size': 26, 'axes.labelsize': 26, 'legend.fontsize': 26, 
    'axes.titlesize': 26, 'xtick.labelsize': 26, 'ytick.labelsize': 26}
		sns.set(rc=rc)
		sns.set_palette("hls",4)
		pl.clf()
		plt.plot(fpr_SVM, tpr_SVM, label='SVM ROC curve (area = %0.2f)' % roc_auc_SVM)
		plt.plot(fpr_KNN, tpr_KNN, label='KNN ROC curve (area = %0.2f)' % roc_auc_KNN)
		plt.plot(fpr_GNB, tpr_GNB, label='GNB ROC curve (area = %0.2f)' % roc_auc_GNB)
		plt.plot(fpr_tree, tpr_tree, label='DT ROC curve (area = %0.2f)' % roc_auc_tree)		
		plt.plot([0, 1], [0, 1], 'k--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.0])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		#plt.title('Receiver operating characteristic (ROC) curves')
		plt.legend(loc="lower right")
		plt.show()


if __name__=='__main__':


#########################################################################################
	inputDir_geo = '../../LIWC/txt/geo/3_13_threshold600_complete_803users.txt'
	inputDir_geo = '../../LIWC/txt/geo/3_18_threshold600_manual_filtered_765users.txt'
	inputDir_non_geo = '../../LIWC/txt/non_geo/3_12_865users.txt'
	inputDir_non_geo = '../../LIWC/txt/non_geo/3_25_545users.txt'
	outputDir = 'out.txt'
	trainer = Trainer(inputDir_geo,inputDir_non_geo,outputDir)

	# read geo data
	trainer.read_data_geo()
	# read non-geo data
	trainer.read_data_non_geo()
	# train the classifier
	trainer.train_data()




