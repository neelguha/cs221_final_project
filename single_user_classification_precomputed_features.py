# Single User Classification 
# Constructs a unique classifier for each user and evaluates a sample of comments 
# to determine which came from the user and which didn't

from collections import defaultdict
import numpy as np
import random
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
import tfidf_util
from feature_generator import *

USERS_TO_TRACK = 97 # Number of users to build classifier for
USER_TRAIN_TEST_SIZE = 50 # Number of target user's comments to train 
OTHER_TRAIN_TEST_SIZE = 50 # Number of comments to include from users in train/test
verbose = False

user_dict = defaultdict(list)
all_features = []
all_users  = []
index = 0
for line in open("sampled_users_features.tsv"):
	if line[0] == "#": continue
	user,feature_text = line.strip().split("\t")
	features = feature_text.split(",")
	features = [float(x) for x in features if len(x) > 0]
	user_dict[user].append(features)
	all_users.append(user)	
	all_features.append(features)
	index += 1

print "Read in features."
users_to_test = np.random.choice(user_dict.keys(),USERS_TO_TRACK,replace = False)
output_file = open("single_user_results.csv","w")


precisionScore = 0.0
recallScore = 0.0

for user in users_to_test:
	if verbose: print "User:",user
	other_features_indices = np.random.choice([i for i in range(len(all_features)) if not all_users[i] == user],OTHER_TRAIN_TEST_SIZE) 
	other_features = []
	for index in other_features_indices:
		other_features.append(all_features[index])
	user_train,user_test = np.array_split(user_dict[user],2)
	other_train,other_test = np.array_split(other_features,2)

	all_train = list(user_train) + list(other_train)
	train_vals = [True]*len(user_train) + [False]*len(other_train)
	all_test = list(user_test) + list(other_test)
	test_vals = [True]*len(user_test) + [False]*len(other_test)
	
	#logreg = LogisticRegression()
	#logreg.fit(all_train,train_vals)
	#predictions = logreg.predict(all_test)
	#clf = svm.SVC()
	#clf.fit(all_train, train_vals)  
	#predictions = clf.predict(all_test)
	gnb = GaussianNB()
	gnb.fit(all_train,train_vals)
	predictions = gnb.predict(all_test)

	precision,recall,fbeta_score,support = precision_recall_fscore_support(test_vals, predictions, average='binary')
	if verbose:
		print "\tPrecision:%f\n\tRecall:%f" % (precision,recall)
	output_file.write("%f,%f\n" %(precision,recall))
	precisionScore += precision
	recallScore += recall


print "Avg Precision:%f\nAvg Recall:%f" % (precisionScore/len(users_to_test), recallScore/len(users_to_test))
output_file.write("Avg:%f,%f" % (precisionScore/len(users_to_test), recallScore/len(users_to_test)))

