import nltk
import math
import syllables_en

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
	    
	# SMOG grade for readability
	def SMOG_readability(self, stats):
	    score = 0.0 
	    if self.word_count > 0.0:
	        score = (math.sqrt(self.complex_word_count*(30/self.sentence_count)) + 3)
	    return score

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
	def count_syllables(self):
	    syllableCount = 0
	    for word in self.words:
	        syllableCount += syllables_en.count(word)
	    return syllableCount

	# nltk/contrib
	def count_complex_words(self):
	    complex_words = 0
	    found = False
	    cur_word = []
	    
	    for word in self.words:          
	        cur_word.append(word)
	        if count_syllables(cur_word)>= 3:
	            
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
	                
	        cur_word.remove(word)
	    return complex_words