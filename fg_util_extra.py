# TODO:
# Sentiment (A)

import nltk
import requests
import fg_util as util
from string import punctuation

class TextStatsExt:
	def __init__(self, text):
		self.stats = util.TextStats(text)

	# calculates num punctuation symbols / num words
	def punctuation_score(self):
		nPunct = 0
		punct = set(punctuation)
		for ch in self.stats.text:
			if ch in punct:
				nPunct += 1
		return float(nPunct / self.stats.word_count)

	# returns number of curse words 
	def profanity_score(self):
		pass

	def sentiment_score(self):
		response = requests.post("http://text-processing.com/api/sentiment/", data={'text':self.stats.text})
		print response, "text:",self.stats.text