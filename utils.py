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
from gensim.corpora import Dictionary
from nltk.tokenize.toktok import ToktokTokenizer
toktok = ToktokTokenizer()

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

#SPECIAL TOKENS
UNK_TOKEN = 'UNKNOWN'
EMP_TOKEN = 'EMPTY'
EOS_TOKEN = 'END'
SWT_TOKEN = 'SWITCH'  

#Tokenization tools
tokenizer_es = nltk.data.load('tokenizers/punkt/spanish.pickle')	
tokenizer_en = nltk.data.load('tokenizers/punkt/english.pickle')	

ENGLISH_ENCLITICS = ['s', 'll', 't', 'd', 've', 're', 'm', 'clock', 'am', 'er', 'em', 'cuz']

DEFAULT_N_PROSODY_PARAMS = 3

MAIN_PUNCS = ['.', ',', '?']

def tokenize_es(string, to_lower = False):
	tokens = []
	if to_lower:
		string = string.lower()
	for sent in tokenizer_es.tokenize(normalize(string)):
		tokens.extend(toktok.tokenize(sent))
	return tokens

def tokenize_en(string, to_lower = False):
	tokens = []
	if to_lower:
		string = string.lower()
		
	for sent in tokenizer_en.tokenize(normalize(string)):
		tokens.extend(toktok.tokenize(sent))
		
	#fix apostrophe
	tokens_fix = []
	tokens_index = 0
	while tokens_index < len(tokens):
		token =  tokens[tokens_index]
		if token == "'":
			if tokens_index + 1 < len(tokens):
				next_token = tokens[tokens_index + 1]
				if next_token.lower() in ENGLISH_ENCLITICS:
					tokens_fix.append("'" + next_token)
					tokens_index += 1
				else:
					tokens_fix.append("'")
			else:
				tokens_fix.append("'")
		else:
			tokens_fix.append(tokens[tokens_index])
		tokens_index += 1
	return tokens_fix

def normalize(line):
	normalized = re.sub('-', ' - ', line)
	normalized = normalized.strip()
	return normalized

def indexes_from_sentence(lang, sentence):
	tokens = lang.tokenize(sentence, to_lower = True)
	return indexes_from_tokens(lang, tokens)

def indexes_from_proscript(lang, proscript):
	tokens = read_tokens_from_proscript(proscript)
	return indexes_from_tokens(lang, tokens)

# return index list from token list. adds EOS at the end
def indexes_from_tokens(lang, tokens):
	return [lang.token2index(token) for token in tokens] + [lang.token2index(EOS_TOKEN)]

# Creates dummy prosody vectors for given sentence tokens. adds extra vector for EOS
def prosody_from_tokens(tokens, n_prosody_params=DEFAULT_N_PROSODY_PARAMS):
	return [[0.0] * n_prosody_params for token in tokens] + [[0.0] * n_prosody_params]

def flags_from_prosody(prosody_seq):
	return [[1 if not feature_value == 0.0  else 0 for feature_index, feature_value in enumerate(prosody_token)] for prosody_token in prosody_seq]

def pad_seq(lang, seq, max_length, pad_value=EMP_TOKEN):
	seq += [lang.token2index(pad_value) for i in range(max_length - len(seq))]
	return seq

def pad_prosody_seq(seq, max_length, pad_values):
	assert seq.shape[1] == len(pad_values)
	padding = [pad_values] * (max_length - len(seq))
	if padding:
		padded_seq = np.append(seq, padding, axis=0)
		return padded_seq
	else:
		return seq

def pad_flag_seq(seq, max_length, n_prosody_params=DEFAULT_N_PROSODY_PARAMS, pad_value = 0):
	empty_flags = [pad_value] * n_prosody_params
	seq += [empty_flags for i in range(max_length - len(seq))]
	return seq

def remove_punc_tokens(tokens, keep_main_puncs = False):
	if keep_main_puncs:
		return [token for token in tokens if not re.search(r"\w+|'|\.|\?|,", token) == None]
	else:
		return [token for token in tokens if not re.search(r"\w+|'", token) == None]

def readable_from_tokens(tokens):
	return ' '.join(tokens)

def print_prosody(prosody):
	print(np.array(prosody).transpose())

def normalize_prosody(prosody_seq, min_values, max_values, norm_values=None, flag_seq = None):
	assert len(min_values) == len(max_values)
	if norm_values is not None:
		assert len(norm_values) == len(max_values)
	for token_index, prosody_token in enumerate(prosody_seq):
		for feature_index, feature_value in enumerate(prosody_token):
			if not feature_index >= len(max_values):
				if not flag_seq == None and not flag_seq[token_index][feature_index]:
					prosody_token[feature_index] = normalize_value(norm_values[feature_index], min_values[feature_index], max_values[feature_index])
				else:
					prosody_token[feature_index] = normalize_value(prosody_token[feature_index], min_values[feature_index], max_values[feature_index])
	return prosody_seq

def normalize_value(value, min_value, max_value):
	if value <= min_value:
		return 0.0
	elif value >= max_value:
		return 1.0
	return (value - min_value) / (max_value - min_value)

def unnormalize_value(value, min_value, max_value):
	return value * (max_value - min_value) + min_value

'''
limits a sequence to max_length size. 
Assumes that there is an END token at the end.
'''
def limit_seqs_to_max(sequences, max_length):
	new_sequences = []
	for seq in sequences:
		if not len(seq) <= max_length:
			trimmed = seq[0:max_length - 1]
			#print(len(trimmed))
			end_token = seq[-1]
			#print(len([end_token]))
			new_seq = np.concatenate((trimmed, [end_token]))
			#print(len(new_seq))
		else:
			new_seq = seq
		new_sequences.append(new_seq)
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

def load_audio_dataset(audio_data_file, input_lang, output_lang, n_prosody_params, input_prosody_params, output_prosody_params, dummyfy_input_prosody=False, dummyfy_output_prosody=False, shuffle=False):
	audio_data = read_audio_dataset_file(audio_data_file, shuffle=False)

	input_data = []
	output_data = []

	for sample in audio_data:
		es_txt = sample[0]
		es_csv = sample[1]
		en_txt = sample[2]
		en_csv = sample[3]

		if input_lang.lang_code == 'en' and output_lang.lang_code == 'es':
			input_csv = en_csv
			output_csv = es_csv
		elif input_lang.lang_code == 'es' and output_lang.lang_code == 'en':
			input_csv = es_csv
			output_csv = en_csv
			
		if input_lang.punctuation_level == 0:
			input_punc = False
			input_only_main_punc = False
		elif input_lang.punctuation_level == 1:
			input_punc = True
			input_only_main_punc = True
		elif input_lang.punctuation_level == 2:
			input_punc = True
			input_only_main_punc = False
			
		if output_lang.punctuation_level == 0:
			output_punc = False
			output_only_main_punc = False
		elif output_lang.punctuation_level == 1:
			output_punc = True
			output_only_main_punc = True
		elif output_lang.punctuation_level == 2:
			output_punc = True
			output_only_main_punc = False

		input_tokens, input_prosody = read_data_from_proscript(input_csv, input_lang, n_prosody_params, input_prosody_params, punctuation_as_tokens = input_punc, keep_only_main_puncs = input_only_main_punc)
		output_tokens, output_prosody = read_data_from_proscript(output_csv, output_lang, n_prosody_params, output_prosody_params, punctuation_as_tokens = output_punc, keep_only_main_puncs = output_only_main_punc)

		if dummyfy_input_prosody:
			input_prosody = np.zeros_like(input_prosody)
		if dummyfy_output_prosody:
			output_prosody = np.zeros_like(output_prosody)

		input_data.append((input_tokens, input_prosody, input_csv))
		output_data.append((output_tokens, output_prosody, output_csv))

	return input_data, output_data

'''
Functions to save and load models
'''
def save_model(encoder_state, decoder_state, model_name, models_path, checkpoint=False):
	"""Save checkpoint if a new best is achieved"""
	if checkpoint:
		print ("=> Saving checkpoint")
		encoder_model_path = os.path.join(models_path, model_name + '_encoder_checkpoint' + '.model')
		decoder_model_path = os.path.join(models_path, model_name + '_decoder_checkpoint' + '.model')	
	else:
		print("==> Saving new best model")
		encoder_model_path = os.path.join(models_path, model_name + '_encoder' + '.model')
		decoder_model_path = os.path.join(models_path, model_name + '_decoder' + '.model')	
	
	torch.save(encoder_state, encoder_model_path)  # save checkpoint
	torch.save(decoder_state, decoder_model_path)  # save checkpoint

def log_loss(loss_file, row):
	with open(loss_file, 'a') as f:
		wr = csv.writer(f)
		wr.writerow(row)

def initialize_log_loss(loss_file, header):
	with open(loss_file, 'w') as f:
		wr = csv.writer(f)
		wr.writerow(header)


def load_model(encoder, decoder, encoder_model_path, decoder_model_path, gpu_to_cpu=False):
	print('gpu2cpu:', gpu_to_cpu)
	if gpu_to_cpu:
		loaded_encoder_state_dict = torch.load(encoder_model_path, map_location='cpu')
		loaded_decoder_state_dict = torch.load(decoder_model_path, map_location='cpu')
	else:
		loaded_encoder_state_dict = torch.load(encoder_model_path)
		loaded_decoder_state_dict = torch.load(decoder_model_path)

	encoder.load_state_dict(loaded_encoder_state_dict, strict=False)
	decoder.load_state_dict(loaded_decoder_state_dict, strict=False)

'''
Reads contents of a text file and returns as string
'''
def read_text_file(file):
	with open(file,'r') as f:
		return f.read()

'''
Reads dictionary file saved as csv. Each line has tab separated w2v_index and token.
Returns index2token and token2index dictionaries
'''
def dict_from_file(filename):
	idx2token = {}
	#idx2w2v = {}
	idx = 0
	with open(filename, encoding='utf-8') as f:
		reader = csv.reader(f, delimiter='\t')
		for row in reader:
			idx2token[idx] = row[0]
			#idx2w2v[idx] = int(row[1])
			idx += 1

	token2idx = {v: k for k, v in idx2token.items()}
	return idx2token, token2idx

'''
reads proscript format transcription as a dictionary. 
n_prosody_params has to be larger than len(prosody_params)
'''
def read_data_from_proscript(filename, lang, n_prosody_params, prosody_params = [], punctuation_as_tokens=False, keep_only_main_puncs=False):
	token_sequence = []
	prosody_sequence = np.empty((0,n_prosody_params), float)
	punctuation_sequence = []
	last_word_index = 0

	with open(filename) as f:
		reader = csv.DictReader(f, delimiter='|') # read rows into a dictionary format
		for row_index, row in enumerate(reader): # read a row as {column1: value1, column2: value2,...}
			if punctuation_as_tokens and row['punctuation_before']:
				punctuation_tokens = lang.tokenize(row['punctuation_before'])
				if keep_only_main_puncs:
					punctuation_tokens = [punc for punc in punctuation_tokens if punc in MAIN_PUNCS]
				if len(punctuation_tokens) > 0:
					token_sequence += punctuation_tokens
					punc_prosody_vector = np.zeros((len(punctuation_tokens), n_prosody_params ))
				else:
					punc_prosody_vector = None
			else:
				punc_prosody_vector = None
			
			#prosody vector for each word token
			word_tokens = lang.tokenize(row['word'])   
			token_sequence += word_tokens
			prosody_vector = np.zeros((len(word_tokens), n_prosody_params ))
			#if not len(prosody_params) == 0:
			for param_index in (param_index for param_index in range(n_prosody_params) if param_index < len(prosody_params)):
				prosody_param = prosody_params[param_index]
				for token_index, token in enumerate(word_tokens):
					if "mean" in prosody_param: 
						prosody_vector[token_index][param_index] = float(row[prosody_param])
					elif prosody_param == "pause_before" and token_index == 0 and not row_index == 0:
						prosody_vector[token_index][param_index] = float(row[prosody_param])
					elif prosody_param == "pause_after" and token_index == len(word_tokens) - 1:
						prosody_vector[token_index][param_index] = float(row[prosody_param])

			if punc_prosody_vector is not None:
				for param_index in (param_index for param_index in range(n_prosody_params) if param_index < len(prosody_params)):
					prosody_param = prosody_params[param_index]
					if "mean" in prosody_param: 
						#insert the mean value of the word it's attached to
						punc_prosody_vector[:,param_index] = prosody_vector[-1][prosody_params.index(prosody_param)]
				prosody_sequence = np.append(prosody_sequence, punc_prosody_vector, axis=0)
			prosody_sequence = np.append(prosody_sequence, prosody_vector, axis=0)
			last_word_index = len(prosody_sequence) - 1
			
			if punctuation_as_tokens and row['punctuation_after']:
				punctuation_tokens = lang.tokenize(row['punctuation_after'])
				if keep_only_main_puncs:
					punctuation_tokens = [punc for punc in punctuation_tokens if punc in MAIN_PUNCS]
				if len(punctuation_tokens) > 0:
					token_sequence += punctuation_tokens
					punc_prosody_vector = np.zeros((len(punctuation_tokens), n_prosody_params ))
					if not len(prosody_params) == 0:
						for param_index in (param_index for param_index in range(n_prosody_params) if param_index < len(prosody_params)):
							prosody_param = prosody_params[param_index]
							if "mean" in prosody_param: 
								#insert the mean value of the word it's attached to
								punc_prosody_vector[:,param_index] = prosody_sequence[-1][prosody_params.index(prosody_param)]
					prosody_sequence = np.append(prosody_sequence, punc_prosody_vector, axis=0)
				
	#Last word shouldn't have a pause_after
	if 'pause_after' in prosody_params and prosody_params.index('pause_after') < n_prosody_params:
		prosody_sequence[last_word_index][prosody_params.index('pause_after')] = 0.0
		
	return token_sequence, prosody_sequence


'''
adds a dummy prosody token at the end of a sequence to match with a finalized word sequence
'''
def finalize_prosody_sequence(prosody_sequence):
	n_prosody_params = len(prosody_sequence[0])
	prosody_sequence = np.vstack([prosody_sequence, [0.0] * n_prosody_params])
	return prosody_sequence

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
	def __init__(self, lang_code, w2v_model_path, lookup_table_path, punctuation_level = 2):
		self.lang_code = lang_code
		self.w2v_model = gensim.models.Word2Vec.load(w2v_model_path)
		self.word_vectors = self.w2v_model.wv
		self.idx2token, self.token2idx = dict_from_file(lookup_table_path)
		self.vocabulary_size = len(self.idx2token)
		self.punctuation_level = punctuation_level

		print("%s Vocabulary size: %i"%(self.lang_code, self.vocabulary_size))

	def index2token(self, word_index):
		try:
			return self.idx2token[word_index]
		except KeyError:
			return UNK_TOKEN

	def token2index(self, word):
		try:
			return self.token2idx[word]
		except KeyError:
			eprint("Unknown (%s): %s"%(self.lang_code, word))
			return self.token2idx[UNK_TOKEN]

	def word2vec(self, word):
		try:
			idx = self.token2idx[word]
			return self.word_vectors[word]
		except KeyError:
			return self.word_vectors[UNK_TOKEN]

	def tokenize(self, string, to_lower = False):
		if self.lang_code == 'en':
			return tokenize_en(string, to_lower = to_lower)
		elif self.lang_code == 'es':
			return tokenize_es(string, to_lower = to_lower)

	def get_weights_matrix(self):
		weights_matrix = np.zeros((self.vocabulary_size, self.word_vectors.vector_size))
		for idx, token in self.idx2token.items():
			# try: 
			weights_matrix[idx] = self.word_vectors[token]
			# except KeyError:
			# 	print("keyerr")
			# 	weights_matrix[index] = np.random.normal(scale=0.6, size=(self.word_vectors.vector_size, ))

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