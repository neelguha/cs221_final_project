from fg_util import *
from fg_util_extra import *

# Given a set of comments, returns a feature vector representing those comments
def extract_features(self, comment):
	result = {}

	# stats = TextStats(comment)
	# entity_names = extract_entity_names(stats[])
	stats = TextStatsExt(comment)
	stats.sentiment_score()
	return result

for line in open("sampled_users.tsv"):
	extract_features(line)