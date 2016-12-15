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


verbose = True

user_dict = defaultdict(list)
for line in open("sampled_users_features.tsv"):
	user,feature_text = line.strip().split("\t")
	feature_vector = feature_text.strip().split(",")
	feature_vector = [float(x) for x in feature_vector if len(x) > 0]
	user_dict[user].append(feature_vector)

all_X = []
all_Y= []
#i = len(user_dict.keys())
i = 100
users_to_test = np.random.choice(user_dict.keys(),i)
for user in users_to_test:
	num_comments = len(user_dict[user])
	features = user_dict[user]
	for x in features:
		all_X.append(x)
		all_Y.append(user)

kmeans = KMeans(n_clusters=i, random_state=0)
kmeans.fit(all_X)
predictions = kmeans.labels_
print metrics.adjusted_rand_score(all_Y,predictions)

