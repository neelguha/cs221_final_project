from collections import defaultdict
import numpy as np
from sklearn.linear_model import LogisticRegression
import random
from feature_generator import *
from sklearn.metrics import precision_recall_fscore_support

USER_COMMENT_SUBSETS = 10 # How many distinct sets we want to split a single user's comments into 
COMMENT_WORD_THRESHOLD = 50 # Ignore comments with fewer than this many words
USERS_TO_TRACK = 20 # Number of users to build classifier for 

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

users_to_test = np.random.choice(user_dict.keys(),USERS_TO_TRACK,replace = False)
print users_to_test
for user in users_to_test:
	user_features = []
	user_comments = np.random.choice(user_dict[user],40) 
	for comment in user_comments:
		user_features.append(extract_features(comment))
	other_comments = np.random.choice(all_comments,40) 
	other_features = []
	for comment in other_comments:
		other_features.append(extract_features(comment))
	print len(user_features),len(other_features)
	user_train,user_test = np.array_split(user_features,2)
	other_train,other_test = np.array_split(other_features,2)
	print len(other_train),len(user_train)
	all_train = list(user_train) + list(other_train)
	train_vals = [True]*len(user_train) + [False]*len(other_train)
	logreg = LogisticRegression()
	logreg.fit(all_train,train_vals)

	all_test = list(user_test) + list(other_test)
	test_vals = [True]*len(user_test) + [False]*len(other_test)
	predictions = logreg.predict(all_test)
	print precision_recall_fscore_support(test_vals, predictions, average='binary')



# Filter comments which don't meet length threshold and split comments per user into subset
'''filtered_comments = {}
for user in all_users:
	user_comments = []
	for comment in all_users[user]:
		if len(comment['text'].split(" ")) > COMMENT_WORD_THRESHOLD:
			user_comments.append(comment['text'])
	random.shuffle(user_comments)
	if len(user_comments) == 0:
		continue
	comments = np.array_split(user_comments,USER_COMMENT_SUBSETS)
	features = [extract_features(comment) for comment in comments]
	filtered_comments[user] = features








# pick sample of users to build a classifier for 
users_to_track = np.random.choice(filtered_comments.keys(),USERS_TO_TRACK,replace = False)

# Iterate through this set of users 
for track_user in users_to_track:
	train_keys = set()
	test_keys = set()
	train_comment_features = []
	train_values = []
	test_comment_features = []
	test_values = []
	for user in filtered_comments:
		comment_features = filtered_comments[user]
		train_indices = np.random.choice(range(len(comments)),USER_COMMENT_SUBSETS / 2,replace = False)
		for i in range(len(comment_features)):
			if i in train_indices:
				train_comment_features.append(comment_features[i])
				train_values.append(user == track_user)
			else:
				test_comment_features.append(comment_features[i])
				test_values.append(user == track_user)
	print len(test_values), sum(test_values)
	logreg = LogisticRegression()
	logreg.fit(train_comment_features,train_values)
	predictions = logreg.predict(test_comment_features)
	#test_values = [1,1,1,0,0,0]
	#predictions = [1,1,0,0,0,0]
	print precision_recall_fscore_support(test_values, predictions, average='binary')'''
	




