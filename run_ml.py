import sklearn 
from collections import defaultdict
import numpy as np
all_users = defaultdict(list)


for line in open("sampled_users.tsv"):
	if line[0] == "#": continue
	
	author_id,comment_id,timestamp,text = line.strip().split("\t")
	comment_dict = {'author_id':author_id,'comment_id':comment_id,'timestamp':timestamp,'text':text}
	all_users[author_id].append(comment_dict)

train_comments = defaultdict(list)
test_comments = defaultdict(list)

for user in all_users:
	comments = all_users[user]
	train_indices = np.random.choice(range(len(comments)),len(comments)/ 2,replace=False)
	for i in range(len(comments)):
		if i in train_indices:
			train_comments[user].append(comments[i]['text'])
		else:
			test_comments[user].append(comments[i]['text'])

train_features = []
test_features = []

for user in all_users:
	num_train_words = sum(len(comment.split()) for comment in train_comments[user]) / len(train_comments[user])
	num_test_words = sum(len(comment.split()) for comment in test_comments[user]) / len(test_comments[user])
	train_features.append(num_train_words)
	test_features.append(num_test_words)




