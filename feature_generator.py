from fg_util import *

# Given a set of comments, returns a feature vector representing those comments
def extract_features(self, comment):
	result = {}

	stats = TextStats(comment)
	entity_names = extract_entity_names(stats[])

	return result