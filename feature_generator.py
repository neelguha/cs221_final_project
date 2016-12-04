from fg_util import *
import random
# Given a set of comments, returns a feature vector representing those comments




def extract_features(comment):
	stats = TextStats(comment)
	logWordCount = stats.log_word_count()
	maxWordLen = stats.max_word_len()
	readability = stats.SMOG_readability()
	numSpellingErrors = stats.num_spelling_errors()
	#entityNames = stats.extract_entity_names()
	punctScore = stats.punctuation_score()
	profanityScore = stats.profanity_score()
	#sentimentScore = stats.sentiment_score()
	
	result = {
		'log_word_count': logWordCount,
		'max_word_len': maxWordLen,
		'readability': readability, # might have to take the log of this
		'num_spelling_errs': numSpellingErrors,
		'punctuation_measure': punctScore,
		'profanity': profanityScore,
		#'sentiment': sentimentScore
	}
	#for name in entityNames:
		#result[('entity', name)] = 1

	return [v for k,v in result.items()]

