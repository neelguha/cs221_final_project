# Get statistics about the sample of data we are using 
from collections import defaultdict
import numpy as np

user_dict = defaultdict(list)
all_comments = []
all_users = []

for line in open("sampled_users.tsv"):
	if line[0] == "#": continue
	author_id,comment_id,timestamp,text = line.strip().split("\t")
	comment_dict = {'author_id':author_id,'comment_id':comment_id,'timestamp':timestamp,'text':text}
	user_dict[author_id].append(comment_dict)
	all_users.append(author_id)	
	all_comments.append(text)



# TODO: Add statistics about how many posts these comments span
print "Number of users: %d" % len(user_dict)
print "Number of comments: %d" % len(all_comments)
print "Average number of comments per user: %f" % float(len(all_comments)/len(user_dict))
print "Average number of words per comment: %f" % np.mean([len(comment.split(" ")) for comment in all_comments])