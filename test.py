from optparse import OptionParser
import string
import re
import os
import sys
import math

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from utils import *
from models import *
import yaml

def text_test_results_generator(data_path, encoder, decoder, input_lang, output_lang, n_prosody_params, max_seq_length, log_file=None, stop_at = -1, USE_CUDA=False):
	count = 0
	if not log_file == None:
		out_log_file = open(log_file, 'w')
	with open(data_path,'r') as inputfile:
		for line in inputfile:
			if not stop_at == -1 and count >= stop_at:
				break
			pair = [sentence.strip() for sentence in line.split('\t')]
			if input_lang.lang_code == 'en':
				in_sentence = pair[0]
				out_sentence = pair[1]
			elif input_lang.lang_code == 'es':
				in_sentence = pair[1]
				out_sentence = pair[0]

			in_sentence_tokens = in_sentence.lower().split()
			out_sentence_tokens = out_sentence.lower().split()

			if input_lang.omit_punctuation:
				in_sentence_tokens = remove_punc_tokens(in_sentence_tokens)
			if output_lang.omit_punctuation:
				out_sentence_tokens = remove_punc_tokens(out_sentence_tokens)
				
			translation_tokens, attentions = evaluate(input_seq_tokens = in_sentence_tokens, 
													  input_prosody_seq = None, 
													  input_lang = input_lang, 
													  output_lang = output_lang, 
													  encoder = encoder, 
													  decoder = decoder, 
													  n_prosody_params = n_prosody_params, 
													  max_length = max_seq_length,
													  USE_CUDA = USE_CUDA)
			count += 1

			#log translations
			if not log_file == None:
				out_log_file.write("> %s (%s)\n"%(readable_from_tokens(in_sentence_tokens), in_sentence))
				out_log_file.write("= %s (%s)\n"%(readable_from_tokens(out_sentence_tokens), out_sentence))
				out_log_file.write("< %s\n"%readable_from_tokens(translation_tokens[:-1]))
				out_log_file.write("---\n")

			yield [out_sentence_tokens], translation_tokens[:-1]
	if not log_file == None:
		out_log_file.close()

def audio_test_results_generator(data_path, encoder, decoder, input_lang, output_lang, n_prosody_params, input_prosody_params, max_seq_length, log_file=None, stop_at = -1, USE_CUDA=False):
	assert not input_lang == output_lang
	audio_data = read_audio_dataset_file(data_path, shuffle=False)
	
	if not log_file == None:
		out_log_file = open(log_file, 'w')

	#start generating samples from the proscript links in the data file
	count = 0
	for segment_data in audio_data:
		if not stop_at == -1 and count >= stop_at:
			break

		es_txt = segment_data[0]
		es_csv = segment_data[1]
		en_txt = segment_data[2]
		en_csv = segment_data[3]

		if input_lang.lang_code == 'en' and output_lang.lang_code == 'es':
			input_proscript = en_csv
			output_proscript = es_csv
			#input_transcript = read_text_file(en_txt)
			#output_transcript = read_text_file(es_txt)
		elif input_lang.lang_code == 'es' and output_lang.lang_code == 'en':
			input_proscript = es_csv
			output_proscript = en_csv
			#input_transcript = read_text_file(es_txt)
			#output_transcript = read_text_file(en_txt)

		in_sentence_tokens, in_prosody_tokens = read_data_from_proscript(input_proscript, input_lang, n_prosody_params, input_prosody_params)
		##input_prosody_tokens = finalize_prosody_sequence(input_prosody_tokens)
		out_sentence_tokens = read_tokens_from_proscript(output_proscript) 	

		if output_lang.omit_punctuation:
			out_sentence_tokens = remove_punc_tokens(out_sentence_tokens)

		translation_tokens, attentions = evaluate(input_seq_tokens = in_sentence_tokens, 
												  input_prosody_seq = in_prosody_tokens, 
												  input_lang = input_lang, 
												  output_lang = output_lang, 
												  encoder = encoder, 
												  decoder = decoder, 
												  n_prosody_params = n_prosody_params, 
												  max_length = max_seq_length, 
												  USE_CUDA = USE_CUDA)
		count += 1

		#log translations
		if not log_file == None:
			out_log_file.write("> %s\n"%readable_from_tokens(in_sentence_tokens))
			out_log_file.write("= %s\n"%readable_from_tokens(out_sentence_tokens))
			out_log_file.write("< %s\n"%readable_from_tokens(translation_tokens[:-1]))
			out_log_file.write("---\n")

		yield [out_sentence_tokens], translation_tokens[:-1]

	if not log_file == None:
		out_log_file.close()

def evaluate(input_seq_tokens, input_prosody_seq, input_lang, output_lang, encoder, decoder, n_prosody_params, max_length, USE_CUDA=False):
	input_word_seqs = [indexes_from_tokens(input_lang, input_seq_tokens)]
	if input_prosody_seq == None:
		input_prosody_seqs = [prosody_from_tokens(input_seq_tokens, n_prosody_params)]
	else:
		input_prosody_seqs = [finalize_prosody_sequence(input_prosody_seq)]

	#make sure sequences are below max_length. 
	input_word_seqs = limit_seqs_to_max(input_word_seqs, max_length)
	input_prosody_seqs = limit_seqs_to_max(input_prosody_seqs, max_length)

	input_lengths = [len(input_word_seqs[0])]
	input_word_batches = Variable(torch.LongTensor(input_word_seqs)).transpose(0, 1)
	input_prosody_batches = Variable(torch.FloatTensor(input_prosody_seqs)).transpose(0, 1)

	if USE_CUDA:
		input_batches = input_batches.cuda()
		
	# Set to not-training mode to disable dropout
	encoder.train(False)
	decoder.train(False)
	
	# Run through encoder
	encoder_outputs, encoder_hidden = encoder(input_word_batches, input_prosody_batches, input_lengths, None)

	# Create starting vectors for decoder
	decoder_input = Variable(torch.LongTensor([output_lang.token2index(SWT_TOKEN)])) # SOS
	decoder_hidden = encoder_hidden[:decoder.n_layers] # Use last (forward) hidden state from encoder
	
	if USE_CUDA:
		decoder_input = decoder_input.cuda()

	# Store output words and attention states
	decoded_words = []
	decoder_attentions = torch.zeros(max_length + 1, max_length + 1)
	
	# Run through decoder
	for di in range(max_length):
		decoder_output, decoder_hidden, decoder_attention = decoder(
			decoder_input, decoder_hidden, encoder_outputs
		)
		decoder_attentions[di,:decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data

		# Choose top word from output
		topv, topi = decoder_output.data.topk(1)
		#ni = topi[0][0]  #old code
		ni = topi.item()
		if ni == output_lang.token2index(EOS_TOKEN):
			decoded_words.append(EOS_TOKEN)
			break
		else:
			decoded_words.append(output_lang.index2word(ni))
			
		# Next input is chosen word
		decoder_input = Variable(torch.LongTensor([ni]))
		if USE_CUDA: decoder_input = decoder_input.cuda()

	# Set back to training mode
	encoder.train(True)
	decoder.train(True)
	
	return decoded_words, decoder_attentions[:di+1, :len(encoder_outputs)]


def main(options):
	#Testing on either audio data or text data
	assert (options.use_audio_data or options.use_text_data) and not (options.use_audio_data and options.use_text_data)
	
	USE_CUDA = options.use_cuda
	print("Use cuda: %s" %USE_CUDA)

	try:
		with open(options.params_file, 'r') as ymlfile:
			config = yaml.load(ymlfile)
	except:
		sys.exit("Parameters file missing")

	#Select the right dataset (file)
	if options.use_text_data:	
		if options.use_validation:
			TEST_DATA_PATH = config["TEXT_VALIDATION_DATA_PATH"]
		else:
			TEST_DATA_PATH = config["TEXT_TEST_DATA_PATH"]
	elif options.use_audio_data:	
		if options.use_validation:
			TEST_DATA_PATH = config["AUDIO_VALIDATION_DATA_FILE"]
		else:
			TEST_DATA_PATH = config["AUDIO_TEST_DATA_FILE"]

	#Setup languages
	INPUT_LANG_CODE = config['INPUT_LANG']
	OUTPUT_LANG_CODE = config['OUTPUT_LANG']

	if INPUT_LANG_CODE == 'en' and OUTPUT_LANG_CODE == 'es':
		lang_en = input_lang = Lang(INPUT_LANG_CODE, config["W2V_EN_PATH"], config["DICT_EN_PATH"], omit_punctuation=config["INPUT_LANG_OMIT_PUNC"])
		lang_es = output_lang = Lang(OUTPUT_LANG_CODE, config["W2V_ES_PATH"], config["DICT_ES_PATH"], omit_punctuation=config["OUTPUT_LANG_OMIT_PUNC"])
	elif INPUT_LANG_CODE == 'es' and OUTPUT_LANG_CODE == 'en':
		lang_es = input_lang = Lang(INPUT_LANG_CODE, config["W2V_ES_PATH"], config["DICT_ES_PATH"], omit_punctuation=config["INPUT_LANG_OMIT_PUNC"])
		lang_en = output_lang = Lang(OUTPUT_LANG_CODE, config["W2V_EN_PATH"], config["DICT_EN_PATH"], omit_punctuation=config["OUTPUT_LANG_OMIT_PUNC"])

	# Configure models
	MAX_SEQ_LENGTH = int(config['MAX_SEQ_LENGTH'])
	N_PROSODY_PARAMS = int(config['N_PROSODY_PARAMS'])
	input_prosody_params = config['INPUT_PROSODY']
	encoder_type = config['ENCODER_TYPE']
	attn_model = config['ATTN_MODEL']
	hidden_size = int(config['HIDDEN_SIZE'])
	n_layers = int(config['N_LAYERS'])

	# Initialize models
	if encoder_type == 'sum':
		encoder = EncoderRNN_sum(input_lang.vocabulary_size, N_PROSODY_PARAMS, hidden_size, input_lang.get_weights_matrix(), n_layers)
	elif encoder_type == 'parallel':
		encoder = EncoderRNN_parallel(input_lang.vocabulary_size, N_PROSODY_PARAMS, hidden_size, input_lang.get_weights_matrix(), n_layers)
	else:
		sys.exit("Unrecognized encoder type. Check params file. Exiting...")
	decoder = LuongAttnDecoderRNN(attn_model, hidden_size, output_lang.vocabulary_size, n_layers)

	# Load states from models
	load_model(encoder, decoder, options.encoder_model, options.decoder_model)

	#Initialize testing samples iterators
	if options.use_text_data:
		test_iterator = text_test_results_generator(TEST_DATA_PATH, encoder, decoder, input_lang, output_lang, N_PROSODY_PARAMS, MAX_SEQ_LENGTH, log_file=options.output_file, USE_CUDA=USE_CUDA)
	elif options.use_audio_data:
		test_iterator = audio_test_results_generator(TEST_DATA_PATH, encoder, decoder, input_lang, output_lang, N_PROSODY_PARAMS, input_prosody_params, MAX_SEQ_LENGTH, log_file=options.output_file, USE_CUDA=False)

	#Compute BLEU
	testing_set_bleu, sentence_count = compute_bleu(test_iterator, max_order=4, smooth=False)

	print("Evaluated %i samples."%sentence_count)
	print("BLEU: ", testing_set_bleu)

if __name__ == "__main__":
	usage = "usage: %prog [-s infile] [option]"
	parser = OptionParser(usage=usage)
	parser.add_option("-o", "--output", dest="output_file", default=None, help="test output filename", type="string")
	parser.add_option("-p", "--params", dest="params_file", default=None, help="params filename", type="string")
	parser.add_option("-e", "--encoder", dest="encoder_model", default=None, help="encoder model to load", type="string")
	parser.add_option("-d", "--decoder", dest="decoder_model", default=None, help="decoder model to load", type="string")
	parser.add_option("-c", "--usecuda", dest="use_cuda", default=False, help="run on gpu", action="store_true")
	parser.add_option("-v", "--validate", dest="use_validation", default=False, help="use validation set as testing set", action="store_true")
	parser.add_option("-t", "--text", dest="use_text_data", default=False, help="use text data", action="store_true")
	parser.add_option("-a", "--audio", dest="use_audio_data", default=False, help="use audio data", action="store_true")

	(options, args) = parser.parse_args()
	main(options)