from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

class TFIDF:
	def __init__(self, comments):
		self.tf = TfidfVectorizer(input='content', analyzer='word', min_df = 1, stop_words = 'english', ngram_range = (1,2))
		self.matrix = self.tf.fit_transform(comments)

	def entities(self, comment):
		# for i in range(len(comment) - 1): # unigrams and bigrams

		# get last word
		names = self.tf.get_feature_names()
		values = dict(zip(names, self.matrix.data))
		print values
		return values
