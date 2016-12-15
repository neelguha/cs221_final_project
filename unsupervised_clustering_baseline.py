# Single User Classification 
# Constructs a unique classifier for each user and evaluates a sample of comments 
# to determine which came from the user and which didn't
from sklearn import metrics
from collections import defaultdict
import numpy as np
import random
from feature_generator import *
from sklearn.metrics import precision_recall_fscore_support
from sklearn import svm
from sklearn.cluster import KMeans

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

all_X = []
all_Y= []
#i = len(user_dict.keys())
i = 100
users_to_test = np.random.choice(user_dict.keys(),i)
for user in users_to_test:
	num_comments = len(user_dict[user])
	comments = user_dict[user]
	for comment in comments:
		all_X.append([len(comment['text'].split(" "))])
		all_Y.append(user)

kmeans = KMeans(n_clusters=i, random_state=0)
kmeans.fit(all_X)
predictions = kmeans.labels_
print metrics.adjusted_rand_score(all_Y,predictions)

