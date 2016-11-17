from collections import defaultdict
import numpy as np
from sklearn.linear_model import LogisticRegression
import random
from feature_generator import *


USER_COMMENT_SUBSETS = 100 # How many distinct sets we want to split a single user's comments into 
COMMENT_WORD_THRESHOLD = 50 # Ignore comments with fewer than this many words
USERS_TO_TRACK = 20 # Number of users to build classifier for 


all_users = defaultdict(list)
for line in open("sampled_users.tsv"):
	if line[0] == "#": continue
	author_id,comment_id,timestamp,text = line.strip().split("\t")
	comment_dict = {'author_id':author_id,'comment_id':comment_id,'timestamp':timestamp,'text':text}
	all_users[author_id].append(comment_dict)	

# Filter comments which don't meet length threshold and split comments per user into subset
filtered_comments = {}
for user in all_users:
	user_comments = []
	for comment in all_users[user]:
		if len(comment['text']) > COMMENT_WORD_THRESHOLD:
			user_comments.append(comment['text'])
	random.shuffle(user_comments)
	if len(user_comments) == 0:
		continue
	filtered_comments[user] = np.array_split(user_comments,USER_COMMENT_SUBSETS)

# pick sample of users to build a classifier for 
users_to_track = np.random.choice(filtered_comments.keys(),USERS_TO_TRACK,replace = False)

# Iterate through this set of users 
for track_user in users_to_track:
	train_comment_features = []
	train_values = []
	test_comment_features = []
	test_values = []
	for user in filtered_comments:
		comments = filtered_comments[user]
		train_indices = np.random.choice(range(len(comments)),USER_COMMENT_SUBSETS / 2,replace = False)
		for i in range(len(comments)):
			if i in train_indices:
				train_comment_features.append(extract_features(comments[i]))
				train_values.append(user == track_user)
			else: 
				test_comment_features.append(extract_features(comments[i]))
				test_values.append(user == track_user)
	
	#convert train_comment_features and test_comment_features from list of dicts to list of lists

	logreg = LogisticRegression()
	#print len(train_comment_features)
	print sum(test_values)
	logreg.fit(train_comment_features,train_values)
	predictions = logreg.predict(test_comment_features)
	num_correct = 0
	#print predictions
	for i in range(len(predictions)):
		if predictions[i] == test_values[i]:
			num_correct += 1
	print num_correct,len(predictions)




