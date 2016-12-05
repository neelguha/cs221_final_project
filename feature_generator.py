from fg_util import *
import random
import tfidf_util
import json
# Given a set of comments, returns a feature vector representing those comments

def extract_features(comment, tfidf):
	stats = TextStats(comment['text'])
	logWordCount = stats.log_word_count()
	maxWordLen = stats.max_word_len()
	readability = stats.SMOG_readability()
	numSpellingErrors = stats.num_spelling_errors()
	punctScore = stats.punctuation_score()
	profanityScore = stats.profanity_score()
	#sentimentScore = stats.sentiment_score()
	entityNames = tfidf.entities(comment['index'])
	
	result = {
		'log_word_count': logWordCount,
		'max_word_len': maxWordLen,
		'readability': readability, # might have to take the log of this
		'num_spelling_errs': numSpellingErrors,
		'punctuation_measure': punctScore,
		'profanity': profanityScore,
		#'sentiment': sentimentScore
	}
	for name in entityNames:
		result[('entity', name)] = 1

	return result

