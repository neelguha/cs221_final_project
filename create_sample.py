# Reads in politics.tsv, sorts posts by users and picks a random sample of users
from collections import defaultdict
import numpy as np

NUM_USERS = 100
MIN_COMMENTS = 100
NUM_COMMENTS = 50
MIN_WORDS_PER_COMMENT = 50

user_list = defaultdict(list)

for line in open("politics_subreddit/politics.tsv"):
	items = line.strip().split("\t")
	timestamp = items[1]
	comment_id = items[3]
	author_id = items[5]
	text = items[9]
	text= text.replace("<EOS>",".")
	if len(text.split(" ")) < MIN_WORDS_PER_COMMENT:
		continue
	user_dict = {"timestamp":timestamp,"comment_id":comment_id,"author_id":author_id,"text":text}
	user_list[author_id].append(user_dict)
print len(user_list)

users_to_sample_from = []
for user in user_list:
	if len(user_list[user]) >= MIN_COMMENTS: 
		users_to_sample_from.append(user)

sampled_users = np.random.choice(users_to_sample_from,NUM_USERS)
output_file = open("sampled_users.tsv","w")
print len(sampled_users)
for user in sampled_users:
	sample_comments = np.random.choice(user_list[user],NUM_COMMENTS)
	for comment in sample_comments:
		to_write = "%s\t%s\t%s\t%s\n" % (comment['author_id'],comment['comment_id'],comment['timestamp'],comment['text'])
		output_file.write(to_write)

