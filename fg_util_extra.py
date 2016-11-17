import nltk
from nltk.tokenize import RegexpTokenizer
from string import punctuation
import requests
import json
import profanity

TOKENIZER = RegexpTokenizer('(?u)\W+|\$[\d\.]+|\S+')

class TextStatsExt:
	def __init__(self, text):
		self.text = text

	# copied from fg.util, merge at some point
	def get_words(self):
	    words = []
	    words = TOKENIZER.tokenize(self.text)
	    filtered_words = []
	    for word in words:
	        if word in punctuation or word == " ":
	            pass
	        else:
	            new_word = word.replace(",","").replace(".","")
	            new_word = new_word.replace("!","").replace("?","")
	            filtered_words.append(new_word)
	    return filtered_words

	# calculates num punctuation symbols / num words
	def punctuation_score(self):
		nPunct = 0
		punct = set(punctuation)
		for ch in self.stats.text:
			if ch in punct:
				nPunct += 1
		return float(nPunct / len(self.get_words()))

	# returns number of curse words 
	def profanity_score(self):
		return len(profanity.censored_words(self.text))

	# ok this is hella jank but itll do for now
	def sentiment_score(self):
		url = 'http://text-processing.com/api/sentiment/'
		response = requests.post(url, data={'text':self.text})
		result = response.json()
		label = result['label']
		score = result['probability'][label]
		if label == 'pos':
			score += 1
		elif label == 'neg':
			score *= -1
		else:
			score -= 0.5
		return score