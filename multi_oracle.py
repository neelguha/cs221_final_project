# Runs the oracle for multiclass

from collections import defaultdict
import numpy as np
from sklearn.linear_model import LogisticRegression
import random
from feature_generator import *
from sklearn.metrics import precision_recall_fscore_support
from sklearn import svm
from sklearn.multiclass import OneVsOneClassifier
from sklearn.naive_bayes import GaussianNB

verbose = True

user_dict = defaultdict(list)
for line in open("sampled_users_features.tsv"):
	user,feature_text = line.strip().split("\t")
	feature_vector = feature_text.strip().split(",")
	feature_vector = [float(x) for x in feature_vector if len(x) > 0]
	user_dict[user].append(feature_vector)

train_X = []
train_Y = []
#i = len(user_dict.keys())
i = len(user_dict)
users_to_test = np.random.choice(user_dict.keys(),i,replace=False)
for user in users_to_test:
	num_comments = len(user_dict[user])
	features = user_dict[user]
	random.shuffle(features)
	for x in features:
		train_X.append(x)
		train_Y.append(user)

print "Number of training examples:",len(train_X)

logreg = LogisticRegression()
logreg.fit(train_X,train_Y)
predictions = logreg.predict(train_X)
#clf = svm.SVC(decision_function_shape='ovo')
#clf.fit(train_X, train_Y)
#predictions = clf.predict(test_X)
#gnb = GaussianNB()
#gnb.fit(train_X,train_Y)
#predictions = gnb.predict(test_X)

micro_precision,micro_recall,micro_fbeta_score,micro_support = precision_recall_fscore_support(train_Y, predictions, average='micro')
if verbose:
	print "Micro Precision:%f\nMicro Recall:%f" % (micro_precision,micro_recall)
macro_precision,macro_recall,macro_fbeta_score,macro_support = precision_recall_fscore_support(train_Y, predictions, average='macro')
if verbose:
	print "Macro Precision:%f\nMacro Recall:%f" % (macro_precision,macro_recall)
output_file = open("multi_user_results.csv","w")
output_file.write("%f,%f,%f,%f\n" %(micro_precision,micro_recall,macro_precision,macro_recall))