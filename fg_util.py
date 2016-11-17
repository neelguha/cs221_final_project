import nltk
from nltk.tokenize import RegexpTokenizer
import math
import syllables_en
import spelling
from string import punctuation
import requests
import json
import profanity

TOKENIZER = RegexpTokenizer('(?u)\W+|\$[\d\.]+|\S+')
SPECIAL_CHARS = ['.', ',', '!', '?']

class TextStats:
	def __init__(self, text):
		self.text = text
		self.words = self.get_words()
		self.sentences = self.get_sentences()
	 	self.char_count = float(self.get_char_count())
	 	self.word_count = float(len(words))
	 	self.sentence_count = float(len(sentences))
	 	self.syllable_count = float(self.count_syllables())
	 	self.complex_word_count = float(self.count_complex_words())
	 	self.avg_words_p_sentence = float(word_count/sentence_count)
		self.word_count = float(len(self.words))
		self.char_count = float(self.get_char_count())
		self.sentences = self.get_sentences()
		self.sentence_count = float(len(self.sentences))

		self.spelling_err_count = 0
		self.correct_spelling_errors()

		self.complex_word_count = float(self.count_complex_words())
		self.avg_words_p_sentence = float(self.word_count)/self.sentence_count
	    
	def correct_spelling_errors(self):
		correctedWords = []
		for word in self.words:
			cap = word[0].isupper()
			correction = spelling.correction(word.lower())
			if cap:
				correction = correction[0].upper() + correction[1:]
			correctedWords.append(correction)
			if word != correction:
				self.spelling_err_count += 1
		self.words = correctedWords

	def log_word_count(self):
		return math.log(self.word_count)

	def max_word_len(self):
		return max(len(word) for word in self.words)

	# SMOG grade for readability
	def SMOG_readability(self):
	    score = 0.0 
	    if self.word_count > 0.0:
	        score = (math.sqrt(self.complex_word_count*(30/self.sentence_count)) + 3)
	    return score

	def num_spelling_errors(self):
		return self.spelling_err_count

	def extract_entity_names(self):
		def entity_names_helper(t):
		    entity_names = []
		    if hasattr(t, 'label') and t.label:
		        if t.label() == 'NE':
		            entity_names.append(' '.join([child[0] for child in t]))
		        else:
		            for child in t:
		                entity_names.extend(entity_names_helper(child))
		    return entity_names 

		tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in self.sentences]
		tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
		chunked_sentences = nltk.ne_chunk_sents(tagged_sentences, binary=True)
		entity_names = []
		for tree in chunked_sentences:
		    entity_names.extend(entity_names_helper(tree))
		return entity_names

	# nltk/contrib
	def get_char_count(self):
	    characters = 0
	    for word in self.words:
	        characters += len(word.decode("utf-8"))
	    return characters
	    
	# nltk/contrib
	def get_words(self):
	    words = []
	    words = TOKENIZER.tokenize(self.text)
	    filtered_words = []
	    for word in words:
	        if word in SPECIAL_CHARS or word == " ":
	            pass
	        else:
	            new_word = word.replace(",","").replace(".","")
	            new_word = new_word.replace("!","").replace("?","")
	            filtered_words.append(new_word)
	    return filtered_words

	# nltk/contrib
	def get_sentences(self):
	    return nltk.sent_tokenize(self.text)

	# nltk/contrib
	def count_syllables(self, word):
	    return syllables_en.count(word)

	# nltk/contrib
	# might be sketchy
	def count_complex_words(self):
	    complex_words = 0
	    found = False
	    
	    for word in self.words:          
	        if self.count_syllables(word)>= 3:
	            
	            #Checking proper nouns. If a word starts with a capital letter
	            #and is NOT at the beginning of a sentence we don't add it
	            #as a complex word.
	            if not(word[0].isupper()):
	                complex_words += 1
	            else:
	                for sentence in self.sentences:
	                    if str(sentence).startswith(word):
	                        found = True
	                        break
	                if found: 
	                    complex_words += 1
	                    found = False
	                
	    return complex_words

	def punctuation_score(self):
		nPunct = 0
		punct = set(punctuation)
		for ch in self.text:
			if ch in punct:
				nPunct += 1
		return float(nPunct / self.word_count)

	def profanity_score(self):
		return len(profanity.censored_words(self.text))

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