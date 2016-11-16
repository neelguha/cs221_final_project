from collections import defaultdict
import numpy as np
from sklearn.linear_model import SGDClassifier
import random
all_users = defaultdict(list)


for line in open("sampled_users.tsv"):
	if line[0] == "#": continue
	
	author_id,comment_id,timestamp,text = line.strip().split("\t")
	comment_dict = {'author_id':author_id,'comment_id':comment_id,'timestamp':timestamp,'text':text}
	all_users[author_id].append(comment_dict)

# How to model this problem 
# The goal is to identify whether two comment streams originate from the same user 
# We'd like to build some sort of classifier where given two streams of comments, it outputs a 1 if they are sufficiently similar 
# and a 0 if they are not

train_users = np.random.choice(all_users.keys(),50,replace = False)
test_users = [user for user in all_users if not user in train_users]

user_split_comments = {}

train_comments = []
test_comments = []
for user in all_users:
	comments = all_users[user]
	batch1 = ' '.join([comment['text'] for comment in comments[:len(comments)/2]])
	batch2 = ' '.join([comment['text'] for comment in comments[:len(comments)/2]])
	if user in train_users:
		train_comments.append((user,batch1))
		train_comments.append((user,batch2))
	else:
		test_comments.append((user,batch1))
		test_comments.append((user,batch2))

random.shuffle(train_comments)
random.shuffle(test_comments)
train_features = []
train_values = []
print "Creating train"
for i in range(len(train_comments)):
	for j in range(i):
		user1,comment1 = train_comments[i]
		user2,comment2 = train_comments[j]
		value = (user1 == user2)
		# Calculate feature vector - right now we just look at the number of words in the intersection
	#	print len(comment1)
		split1 = comment1.split(" ")
		split2 = comment2.split(" ")
		feature = len([x for x in split1 if x in split2])
		train_features.append(feature)
		train_values.append(value)
print "creating test"
test_features = []
tests_values = []
for i in range(len(test_comments)):
	for j in range(i):
		user1,comment1 = test_comments[i]
		user2,comment2 = test_comments[j]
		value = (user1 == user2)
		# Calculate feature vector - right now we just look at the number of words in the intersection
		split1 = comment1.split(" ")
		split2 = comment2.split(" ")
		feature = len([x for x in split1 if x in split2])
		test_features.append(feature)
		test_values.append(value) 

print "fitting"
clf = SGDClassifier(loss="hinge", penalty="l2")
clf.fit(train_features, train_values)

predictions = clf.predict(test_features)
correct = 0
for i in range(len(predictions)):
	if predictions[i] == test_values[i]:
		correct += 1
print correct,len(predictions)









