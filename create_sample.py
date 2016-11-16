# Reads in politics.tsv, sorts posts by users and picks a random sample of users
from collections import defaultdict
import numpy as np

SAMPLE_SIZE = 100


user_list = defaultdict(list)

for line in open("politics_subreddit/politics.tsv"):
	items = line.strip().split("\t")
	timestamp = items[1]
	comment_id = items[3]
	author_id = items[5]
	text = items[9]
	text= text.replace("<EOS>",".")
	user_dict = {"timestamp":timestamp,"comment_id":comment_id,"author_id":author_id,"text":text}
	user_list[author_id].append(user_dict)

users_to_sample_from = []
for user in user_list:
	if len(user_list[user]) >= 100: 
		users_to_sample_from.append(user)

sampled_users = np.random.choice(users_to_sample_from,100)
output_file = open("sampled_users.tsv","w")

for user in sampled_users:
	for comment in user_list[user]:
		to_write = "%s\t%s\t%s\t%s\n" % (comment['author_id'],comment['comment_id'],comment['timestamp'],comment['text'])
		output_file.write(to_write)

