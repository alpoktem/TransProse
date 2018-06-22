import sys
import os
import re
import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import time
import datetime
import math
import csv
import nltk.data
from collections import defaultdict
import numpy as np
import collections
import torch
from nltk.tokenize.toktok import ToktokTokenizer
toktok = ToktokTokenizer()

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

#SPECIAL TOKENS
UNK_TOKEN = 'UNKNOWN'
EMP_TOKEN = 'EMPTY'
EOS_TOKEN = 'END'
SWT_TOKEN = 'SWITCH'  

#Tokenization tools
tokenizer_es = nltk.data.load('tokenizers/punkt/spanish.pickle')	#TEMPORARY COMMENT OUT FOR NLTK PUNKT BUG
tokenizer_en = nltk.data.load('tokenizers/punkt/english.pickle')	#TEMPORARY COMMENT OUT FOR NLTK PUNKT BUG

def tokenize_nltk(string, tokenizer=None, lang_code=None, to_lower = False):
	tokens = []
	if tokenizer == None:
		if lang_code == 'en':
			tokenizer = tokenizer_en
		elif lang_code == 'es':
			tokenizer = tokenizer_es
	if to_lower:
		string = string.lower()
	for sent in tokenizer.tokenize(normalize(string)):
		tokens.extend(toktok.tokenize(sent))

	return tokens

def normalize(line):
	normalized = re.sub('-', ' - ', line)		#remove dash at the beginning
	normalized = normalized.strip()
	return normalized

def normalize_string(string, tokenizer=None, lang_code=None):
	return ' '.join(tokenize_nltk(string, tokenizer, lang_code))

def indexes_from_sentence(lang, sentence):
	tokens = lang.tokenize(sentence, to_lower = True)
	return indexes_from_tokens(lang, tokens)

def indexes_from_proscript(lang, proscript):
	tokens = read_tokens_from_proscript(proscript)
	return indexes_from_tokens(lang, tokens)

# return index list from token list. adds EOS at the end
def indexes_from_tokens(lang, tokens):
	return [lang.word2index(token) for token in tokens] + [lang.SPECIAL_TOKEN2INDEX[EOS_TOKEN]]

# Creates dummy prosody vectors for given sentence tokens. adds extra vector for EOS
def prosody_from_tokens(tokens, n_prosody_params):
	return [[0.0] * n_prosody_params for token in tokens] + [[0.0] * n_prosody_params]

def pad_seq(lang, seq, max_length):
	seq += [lang.SPECIAL_TOKEN2INDEX[EMP_TOKEN] for i in range(max_length - len(seq))]
	return seq

def pad_prosody_seq(seq, max_length, n_prosody_params):
	seq += [[0.0] * n_prosody_params for i in range(max_length - len(seq))]
	return seq

def remove_punc_tokens(tokens):
	return [token for token in tokens if not re.search(r"\w+|'", token) == None]

def readable_from_tokens(tokens):
	return ' '.join(tokens)

'''
limits a sequence to max_length size. 
Assumes that there is an END token at the end.
'''
def limit_seqs_to_max(sequences, max_length):
	new_sequences = []
	for seq in sequences:
		if not len(seq) <= max_length:
			seq = seq[0:max_length - 1] + [seq[-1]]
		new_sequences.append(seq)
	return new_sequences

'''
Audio dataset reader
'''
def read_audio_dataset_file(audio_data_file, shuffle=False):
	audio_data = []
	with open(audio_data_file,'r') as inputfile:
		for line in inputfile:
			seg_data = [column.strip() for column in line.split('\t')]
			audio_data.append(seg_data)
	if shuffle:
		random.shuffle(audio_data)
	return audio_data

'''
Functions to save and load models
'''
def save_model(encoder_state, decoder_state, losses, model_name, models_path, checkpoint=False):
	"""Save checkpoint if a new best is achieved"""
	if checkpoint:
		print ("=> Saving checkpoint")
		encoder_model_path = os.path.join(models_path, model_name + '_checkpoint_encoder' + '.model')
		decoder_model_path = os.path.join(models_path, model_name + '_checkpoint_decoder' + '.model')	
	else:
		print("==> Saving new best model")
		encoder_model_path = os.path.join(models_path, model_name + '_encoder' + '.model')
		decoder_model_path = os.path.join(models_path, model_name + '_decoder' + '.model')	
	
	torch.save(encoder_state, encoder_model_path)  # save checkpoint
	torch.save(decoder_state, decoder_model_path)  # save checkpoint

def load_model(encoder, decoder, encoder_model_path, decoder_model_path):
	loaded_encoder_state_dict = torch.load(encoder_model_path)
	loaded_decoder_state_dict = torch.load(decoder_model_path)

	encoder.load_state_dict(loaded_encoder_state_dict)
	decoder.load_state_dict(loaded_decoder_state_dict)

'''
Reads contents of a text file and returns as string
'''
def read_text_file(file):
	with open(file,'r') as f:
		return f.read()

'''
reads proscript format transcription as a dictionary. 
n_prosody_params has to be larger than len(prosody_params)
'''
def read_data_from_proscript(filename, lang, n_prosody_params, prosody_params):
	tokens = []
	prosody_vectors = []
	
	with open(filename) as f:
		reader = csv.DictReader(f, delimiter='|') # read rows into a dictionary format
		for row in reader: # read a row as {column1: value1, column2: value2,...}
			#word_tokens = lang.tokenize(row['word'])
			word_tokens = re.split("(')", row['word'])	#TEMPORARY NLTK PUNKT BUG SOLUTION
			tokens += word_tokens
			
			prosody_vector_token = [0.0] * n_prosody_params
			if not prosody_params == None:
				for index, prosody_param in enumerate(prosody_params):
					prosody_vector_token[index] = float(row[prosody_param])

			prosody_vectors += [prosody_vector_token] * len(word_tokens)

	prosody_vectors += [[0.0] * n_prosody_params]  #for end token
	return tokens, prosody_vectors

'''
Reads tokens (words and punctuation) from a proscript format transcription
TEMPORARY NLTK PUNKT BUG SOLUTION
'''
def read_tokens_from_proscript(filename):
	tokens = []
	with open(filename) as f:
		reader = csv.DictReader(f, delimiter='|') # read rows into a dictionary format
		for row in reader: # read a row as {column1: value1, column2: value2,...}
			if not row['punctuation_before'] == '':
				tokens.append(row['punctuation_before'])
			word_tokens = re.split("(')", row['word'].lower())	#FOR ENGLISH TOKENS LIKE She's is three tokens "she" "'" "s"
			tokens += word_tokens
			if not row['punctuation_after'] == '':
				tokens.append(row['punctuation_after'])
	return tokens

class Lang:
	def __init__(self, lang_code, w2v_model_path, base_vocabulary_size, omit_punctuation=False):
		self.lang_code = lang_code
		self.w2v_model = gensim.models.Word2Vec.load(w2v_model_path)
		self.word_vectors = self.w2v_model.wv
		self.base_vocabulary_size = base_vocabulary_size
		self.vocabulary_size = base_vocabulary_size + 4
		self.omit_punctuation = omit_punctuation

		self.SPECIAL_TOKEN2INDEX = {EOS_TOKEN: base_vocabulary_size, 
							   EMP_TOKEN: base_vocabulary_size + 1, 
							   UNK_TOKEN: base_vocabulary_size + 2, 
							   SWT_TOKEN: base_vocabulary_size + 3} 

		self.SPECIAL_INDEX2TOKEN = {v: k for k, v in self.SPECIAL_TOKEN2INDEX.items()}

	def index2word(self, word_index):
		if word_index in self.SPECIAL_INDEX2TOKEN:
			return self.SPECIAL_INDEX2TOKEN[word_index]
		else:
			return self.word_vectors.index2word[word_index]

	def word2index(self, word):
		if word in self.SPECIAL_TOKEN2INDEX:
			return self.SPECIAL_TOKEN2INDEX[word]
		else:
			word = word.lower()
			if word not in self.word_vectors.vocab or self.word_vectors.vocab[word].index > self.base_vocabulary_size:
				eprint("Unknown (%s): %s"%(self.lang_code, word))
				return self.SPECIAL_TOKEN2INDEX[UNK_TOKEN]
			else:
				index = self.word_vectors.vocab[word].index
				if index >= self.base_vocabulary_size:
					return self.SPECIAL_TOKEN2INDEX[UNK_TOKEN]
				else:
					return index

	def word2vec(self, word):
		if word in self.SPECIAL_TOKEN2INDEX:
			return self.word_vectors[word]
		else:
			word = word.lower()
			if word not in self.word_vectors.vocab or self.word_vectors.vocab[word].index > self.base_vocabulary_size:
				return self.word_vectors[UNK_TOKEN]
			else:
				index = self.word_vectors.vocab[word].index
				if index > self.base_vocabulary_size:
					return self.word_vectors[UNK_TOKEN]
				else:
					return self.word_vectors[word]

	def normalize_string(self, string):
		if self.lang_code == 'en':
			return normalize_string(string, tokenizer_en)
		elif self.lang_code == 'es':
			return normalize_string(string, tokenizer_es)

	def tokenize(self, string, to_lower = False):
		if self.lang_code == 'en':
			return tokenize_nltk(string, tokenizer_en, to_lower = to_lower)
		elif self.lang_code == 'es':
			return tokenize_nltk(string, tokenizer_es, to_lower = to_lower)

	def get_weights_matrix(self):
		weights_matrix = np.zeros((self.vocabulary_size, self.word_vectors.vector_size))

		for index in range(self.vocabulary_size):
			try: 
				weights_matrix[index] = self.word_vectors.vectors[index]
			except KeyError:
				print("keyerr")
				weights_matrix[index] = np.random.normal(scale=0.6, size=(self.word_vectors.vector_size, ))

		for index in self.SPECIAL_INDEX2TOKEN:
			weights_matrix[index] = self.word_vectors[self.SPECIAL_INDEX2TOKEN[index]]

		self.weights_matrix = weights_matrix
		return weights_matrix

def checkArgument(argname, isFile=False, isDir=False, createDir=False):
	if not argname:
		return False
	else:
		if isFile and not os.path.isfile(argname):
			return False
		if isDir:
			if not os.path.isdir(argname):
				if createDir:
					print("Creating directory %s"%(argname))
					os.makedirs(argname)
				else:
					return False
	return True

'''
to print warnings
'''
def eprint(*args, **kwargs):
	print(*args, file=sys.stderr, **kwargs)


'''
timing assistants
'''
def as_minutes(s):
	m = math.floor(s / 60)
	s -= m * 60
	return '%dm %ds' % (m, s)

def time_since(since, percent):
	now = time.time()
	s = now - since
	es = s / (percent)
	rs = es - s
	return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

'''
BLEU scoring. Source: tensorflow
'''

def _get_ngrams(segment, max_order):
	'''
	Extracts all n-grams upto a given maximum order from an input segment.
	Args:
		segment: text segment from which n-grams will be extracted.
		max_order: maximum length in tokens of the n-grams returned by this
			methods.
	Returns:
		The Counter containing all n-grams upto max_order in segment
		with a count of how many times each n-gram occurred.
	'''
	ngram_counts = collections.Counter()
	for order in range(1, max_order + 1):
		for i in range(0, len(segment) - order + 1):
			ngram = tuple(segment[i:i+order])
			ngram_counts[ngram] += 1
	return ngram_counts

def compute_bleu(test_results_generator, max_order=4, smooth=False):
	'''
	Computes BLEU score of translated segments against one or more references.

	Args:
	reference_corpus: list of lists of references for each translation. Each
		reference should be tokenized into a list of tokens.
	translation_corpus: list of translations to score. Each translation
		should be tokenized into a list of tokens.
	max_order: Maximum n-gram order to use when computing BLEU score.
	smooth: Whether or not to apply Lin et al. 2004 smoothing.

	Returns:
	3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
	precisions and brevity penalty.
	'''
	matches_by_order = [0] * max_order
	possible_matches_by_order = [0] * max_order
	reference_length = 0
	translation_length = 0
	sentence_count = 0
	for (references, translation) in test_results_generator:
		sentence_count += 1
		reference_length += min(len(r) for r in references)
		translation_length += len(translation)

		merged_ref_ngram_counts = collections.Counter()
		for reference in references:
			merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
		translation_ngram_counts = _get_ngrams(translation, max_order)
		overlap = translation_ngram_counts & merged_ref_ngram_counts
		for ngram in overlap:
			matches_by_order[len(ngram)-1] += overlap[ngram]
		for order in range(1, max_order+1):
			possible_matches = len(translation) - order + 1
			if possible_matches > 0:
				possible_matches_by_order[order-1] += possible_matches

	precisions = [0] * max_order
	for i in range(0, max_order):
		if smooth:
			precisions[i] = ((matches_by_order[i] + 1.) /
					   (possible_matches_by_order[i] + 1.))
		else:
			if possible_matches_by_order[i] > 0:
				precisions[i] = (float(matches_by_order[i]) /
						 possible_matches_by_order[i])
			else:
				precisions[i] = 0.0

	if min(precisions) > 0:
		p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
		geo_mean = math.exp(p_log_sum)
	else:
		geo_mean = 0

	ratio = float(translation_length) / reference_length

	if ratio > 1.0:
		bp = 1.
	else:
		try:
			bp = math.exp(1 - 1. / ratio)
		except ZeroDivisionError:
			bp = 0.0

	bleu = geo_mean * bp
	
	#return (bleu, precisions, bp, ratio, translation_length, reference_length)
	return bleu, sentence_count