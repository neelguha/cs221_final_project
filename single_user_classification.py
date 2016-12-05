# Single User Classification 
# Constructs a unique classifier for each user and evaluates a sample of comments 
# to determine which came from the user and which didn't

from collections import defaultdict
import numpy as np
import random
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import tfidf_util
from feature_generator import *

USERS_TO_TRACK = 20 # Number of users to build classifier for
USER_TRAIN_TEST_SIZE = 40 # Number of target user's comments to train 
OTHER_TRAIN_TEST_SIZE = 40 # Number of comments to include from users in train/test
verbose = True

user_dict = defaultdict(list)
all_comments = []
all_users  = []
index = 0
for line in open("sampled_users.tsv"):
	if line[0] == "#": continue
	author_id,comment_id,timestamp,text = line.strip().split("\t")
	comment_dict = {'author_id':author_id,'index':index,'timestamp':timestamp,'text':text}
	user_dict[author_id].append(comment_dict)
	all_users.append(author_id)	
	all_comments.append(comment_dict)
	index += 1

users_to_test = np.random.choice(user_dict.keys(),USERS_TO_TRACK,replace = False)
output_file = open("single_user_results.csv","w")
tfidf = tfidf_util.TFIDF(comment['text'] for comment in all_comments)

for user in users_to_test:
	if verbose: print "User:",user
	user_features = []
	user_comments = np.random.choice(user_dict[user],USER_TRAIN_TEST_SIZE) 
	for comment in user_comments:
		#user_features.append(extract_tfidf(comment, tfidf))
		user_features.append(extract_features(comment, tfidf))
	other_comments = np.random.choice([comment for comment in all_comments if comment['author_id'] != user],OTHER_TRAIN_TEST_SIZE) 
	other_features = []
	for comment in other_comments:
		#other_features.append(extract_tfidf(comment, tfidf))
		other_features.append(extract_features(comment, tfidf))
	user_train,user_test = np.array_split(user_features,2)
	other_train,other_test = np.array_split(other_features,2)

	logreg = LogisticRegression()
	dv = DictVectorizer(sparse=True)

	all_train = list(user_train) + list(other_train)
	all_train = dv.fit_transform(all_train)
	train_vals = [True]*len(user_train) + [False]*len(other_train)
	logreg.fit(all_train,train_vals)

	all_test = list(user_test) + list(other_test)
	all_test = dv.transform(all_test) #ONLY DO TRANSFORM AND NOT FIT FOR TEST VALS
	test_vals = [True]*len(user_test) + [False]*len(other_test)
	predictions = logreg.predict(all_test)
	precision,recall,fbeta_score,support = precision_recall_fscore_support(test_vals, predictions, average='binary')
	if verbose:
		print "\tPrecision:%f\n\tRecall:%f" % (precision,recall)
	output_file.write("%f,%f\n" %(precision,recall))
