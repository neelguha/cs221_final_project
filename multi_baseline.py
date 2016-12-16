# Runs the baseline for multiclass

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
all_comments = []
all_users  = []
index = 0
for line in open("data/sampled_users.tsv"):
	if line[0] == "#": continue
	author_id,comment_id,timestamp,text = line.strip().split("\t")
	comment_dict = {'author_id':author_id,'index':index,'timestamp':timestamp,'text':text}
	user_dict[author_id].append(comment_dict)
	all_users.append(author_id)	
	all_comments.append(comment_dict)
	index += 1

train_X = []
train_Y = []
test_X = []
test_Y = []
#i = len(user_dict.keys())
i = len(user_dict)
users_to_test = np.random.choice(user_dict.keys(),i,replace=False)
for user in users_to_test:
	num_comments = len(user_dict[user])
	comments = user_dict[user]
	random.shuffle(comments)
	train,test = np.array_split(comments,2)
	for comment in train:
		train_X.append([len(comment['text'].split(" "))])
		train_Y.append(user)
	for comment in test:
		test_X.append([len(comment['text'].split(" "))])
		test_Y.append(user)

print "Number of training examples:",len(train_X)
print "Number of test examples:",len(test_X)

logreg = LogisticRegression()
logreg.fit(train_X,train_Y)
predictions = logreg.predict(test_X)
#clf = svm.SVC(decision_function_shape='ovo')
#clf.fit(train_X, train_Y)
#predictions = clf.predict(test_X)
#gnb = GaussianNB()
#gnb.fit(train_X,train_Y)
#predictions = gnb.predict(test_X)

micro_precision,micro_recall,micro_fbeta_score,micro_support = precision_recall_fscore_support(test_Y, predictions, average='micro')
if verbose:
	print "Micro Precision:%f\nMicro Recall:%f" % (micro_precision,micro_recall)
macro_precision,macro_recall,macro_fbeta_score,macro_support = precision_recall_fscore_support(test_Y, predictions, average='macro')
if verbose:
	print "Macro Precision:%f\nMacro Recall:%f" % (macro_precision,macro_recall)
output_file = open("data/multi_user_results.csv","w")
output_file.write("%f,%f,%f,%f\n" %(micro_precision,micro_recall,macro_precision,macro_recall))