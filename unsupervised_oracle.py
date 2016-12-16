# Take a sample of comments from a sample of users, see how well a human can sort them.

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


users_to_test = np.random.choice(user_dict.keys(),USERS_TO_TRACK,replace = False)
