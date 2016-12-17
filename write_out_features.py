# Writes out feature vectors for all users to csv file to prevent unnecessary computation.

from feature_generator import *
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
import numpy as np
import tfidf_util
# Writes out a feature vector for each comment

def standardize(X):
	for col in range(6): #num of non entity features
		x = X[:, col]
		if np.std(x) != 0:
			X[:, col] = (x - np.mean(x))/np.std(x)
		else:
			X[:, col] = (x - np.mean(x))

def output_features():
	user_dict = defaultdict(list)
	all_comments = []
	all_users  = []
	all_features = []
	v = DictVectorizer(sparse=False)
	i = 0
	for line in open("data/sampled_users.tsv"):
		i += 1
		if line[0] == "#": continue
		author_id,comment_id,timestamp,text = line.strip().split("\t")
		comment_dict = {'author_id':author_id,'index':index,'timestamp':timestamp,'text':text}
		user_dict[author_id].append(comment_dict)
		all_users.append(author_id)	
		all_comments.append(comment_dict)
		index += 1

	tfidf = tfidf_util.TFIDF(comment['text'] for comment in all_comments)
	all_features = []
	for comment in all_comments:
		feature_dict = extract_features(comment, tfidf)
		all_features.append(feature_dict)

	v = DictVectorizer(sparse=False)
	X = v.fit_transform(all_features)
	standardize(X)
	#output_file = open("sampled_users_features.tsv","w")
	output_file = open("sampled_users_features_standardized.tsv","w")
	for i in range(len(all_comments)):
		output_file.write("%s\t" % all_users[i])
		for var in X[i]:
			output_file.write("%f," % var)
		output_file.write("\n")

output_features()
