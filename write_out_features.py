# Writes out feature vectors for all users to csv file to prevent unnecessary computation.

from feature_generator import *
from collections import defaultdict
import numpy as np
# Writes out a feature vector for each comment
def write_out_comment_features():
	user_dict = defaultdict(list)
	all_comments = []
	all_users  = []
	all_features = []
	output_file = open("sampled_users_features.tsv","w")
	for line in open("sampled_users.tsv"):
		if line[0] == "#": continue
		author_id,comment_id,timestamp,text = line.strip().split("\t")
		feature = extract_features(text)
		output_file.write(line.strip()+"\t")
		for var in feature:
			output_file.write("%f," % var)
		output_file.write("\n")

# Takes a user's comments, splits them into two sets and writes out a feature vector for each set. 
def write_out_user_split_features():
	output_file = open("sampled_users_split_features.tsv","w")
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
	for user in user_dict:
		user_comments = np.random.choice(user_dict[user],len(user_dict[user])) 
		train_comments,test_comments = np.array_split(user_comments,2)
		combined_comment = ""
		for comment in train_comments:
			combined_comment = combined_comment + " " + comment['text']
		train_features = extract_features(combined_comment)
		output_file.write(user + "\t")
		for var in train_features:
			output_file.write("%f," % var)
		output_file.write("\n")

		combined_comment = ""
		for comment in test_comments:
			combined_comment += " " + comment['text']
		test_features = extract_features(combined_comment)
		output_file.write(user + "\t")
		for var in test_features:
			output_file.write("%f," % var)
		output_file.write("\n")

#write_out_comment_features()
write_out_user_split_features()
		