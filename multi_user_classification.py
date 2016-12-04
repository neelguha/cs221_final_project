# Single User Classification 
# Constructs a unique classifier for each user and evaluates a sample of comments 
# to determine which came from the user and which didn't

from collections import defaultdict
import numpy as np
from sklearn.linear_model import LogisticRegression
import random
from feature_generator import *
from sklearn.metrics import precision_recall_fscore_support

USER_TRAIN_TEST_SIZE = 100
verbose = True

user_dict = defaultdict(list)
all_comments = []
all_users  = []
for line in open("sampled_users.tsv"):
	if line[0] == "#": continue
	author_id,comment_id,timestamp,text = line.strip().split("\t")
	comment_dict = {'author_id':author_id,'comment_id':comment_id,'timestamp':timestamp,'text':text}
	user_dict[author_id].append(comment_dict)
	all_users.append(author_id)	
	all_comments.append(text)

user_mappings = list(set(all_users))

output_file = open("multi_user_results.csv","w")

train_comment_features = []
train_comment_users = []
test_comment_features = []
test_comment_users = []
i = 0
user_sample = np.random.choice(user_dict.keys(),10)
for user in user_sample:
	i += 1
	print i
	user_comments = np.random.choice(user_dict[user],USER_TRAIN_TEST_SIZE) 
	train_comments,test_comments = np.array_split(user_comments,2)
	combined_comment = ""
	for comment in train_comments:
		combined_comment = combined_comment + " " + comment['text']
	#print combined_comment
	user_features = extract_features(combined_comment)
	train_comment_features.append(user_features)
	train_comment_users.append(user)
	combined_comment = ""
	for comment in test_comments:
		combined_comment += " " + comment['text']
	user_features = extract_features(combined_comment)
	test_comment_features.append(user_features)
	test_comment_users.append(user)
	if i == 10:
		break


logreg = LogisticRegression()
logreg.fit(train_comment_features,train_comment_users)
predictions = logreg.predict(test_comment_features)
micro_precision,micro_recall,micro_fbeta_score,micro_support = precision_recall_fscore_support(test_comment_users, predictions, average='micro')
if verbose:
	print "\tPrecision:%f\n\tRecall:%f" % (micro_precision,micro_recall)
macro_precision,macro_recall,macro_fbeta_score,macro_support = precision_recall_fscore_support(test_comment_users, predictions, average='macro')
if verbose:
	print "\tPrecision:%f\n\tRecall:%f" % (macro_precision,macro_recall)
output_file.write("%f,%f,%f,%f\n" %(micro_precision,micro_recall,macro_precision,macro_recall))