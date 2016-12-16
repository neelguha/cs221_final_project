# Writes out feature vectors for all users to csv file to prevent unnecessary computation.

from feature_generator import *
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
import numpy as np
# Writes out a feature vector for each comment
def write_out_comment_features():
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
		feature_dict = extract_features(text)
		all_features.append(feature_dict)
		all_users.append(author_id)
		if i % 100 == 0:
			print i
	X = v.fit_transform(all_features)
	output_file = open("data/sampled_users_features.tsv","w")
	for i in range(len(X)):
		output_file.write("%s," % all_users[i])
		for var in X[i]:
			output_file.write("%f," % var)
		output_file.write("\n")
	print v.get_feature_names()

write_out_comment_features()

		