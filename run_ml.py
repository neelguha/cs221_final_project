from collections import defaultdict
import numpy as np
from sklearn.linear_model import SGDClassifier
import random


USER_COMMENT_SUBSETS = 4 # How many distinct sets we want to split a single user's comments into 


all_users = defaultdict(list)
for line in open("sampled_users.tsv"):
	if line[0] == "#": continue
	
	author_id,comment_id,timestamp,text = line.strip().split("\t")
	comment_dict = {'author_id':author_id,'comment_id':comment_id,'timestamp':timestamp,'text':text}
	all_users[author_id].append(comment_dict)

for user in all_users:
	num_words = 0
	for comment in all_users[user]:
		num_words += len(comment['text'].split(" ")) 
	print user,len(all_users[user]),num_words




# pick sample of users to build a classifier for 

# Iterate through this set of users 
	# For this user, select a subset of their comments 


