from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class TFIDF:
	def __init__(self, comments):
		self.tf = TfidfVectorizer(input='content', analyzer='word', min_df = 1, stop_words = 'english', ngram_range = (1,2))
		self.matrix = self.tf.fit_transform(comments)
		self.features = self.tf.get_feature_names()
		self.threshold = 0.2

	def entities(self, comment_index):
		entities = []

		comment_features = self.matrix[comment_index,:].nonzero()[1]
		for cf in comment_features:
			if self.matrix[comment_index, cf] > self.threshold:
				entities.append(self.features[cf])

		return entities
