from fg_util import *

# Given a set of comments, returns a feature vector representing those comments
def extract_features(self, comment):

	stats = TextStats(comment)
	logWordCount = stats.log_word_count()
	maxWordLen = stats.max_word_len()
	readability = stats.SMOG_readability()
	numSpellingErrors = stats.num_spelling_errors()
	entityNames = stats.extract_entity_names()
	
	result = {
		'log_word_count': logWordCount,
		'max_word_len': maxWordLen,
		'readability': readability, # might have to take the log of this
		'num_spelling_errs': numSpellingErrors
	}
	for name in entityNames:
		result[('entity', name)] = 1

	return result