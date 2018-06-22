from optparse import OptionParser
import unicodedata
import string
import re
import os
import sys
import random
import math

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from masked_cross_entropy import *

from utils import *
from models import *
from train import *
import yaml
import csv
import random

def audio_batch_generator(audio_data, batch_size, input_lang, output_lang, n_prosody_params, input_prosody_params, USE_CUDA=False):
	assert not input_lang == output_lang
	input_word_seqs = []
	input_prosody_seqs = []
	target_seqs = []

	for segment_data in audio_data:
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

		input_word_tokens, input_prosody_tokens = read_data_from_proscript(input_proscript, input_lang, n_prosody_params, input_prosody_params)	
		input_word_indexes = indexes_from_tokens(input_lang, input_word_tokens)
		input_word_seqs.append(input_word_indexes)
		input_prosody_seqs.append(input_prosody_tokens)
		
		#target_word_seq = indexes_from_sentence(output_lang, output_transcript)
		target_word_tokens = read_tokens_from_proscript(output_proscript)
		target_word_indexes = indexes_from_tokens(output_lang, target_word_tokens)
		target_seqs.append(target_word_indexes)

		if len(input_word_seqs) == batch_size:
			# Zip into pairs, sort by length (descending), unzip
			seq_pairs = sorted(zip(input_word_seqs, input_prosody_seqs, target_seqs), key=lambda p: len(p[0]), reverse=True)
			input_word_seqs, input_prosody_seqs, target_seqs = zip(*seq_pairs)

			# For input and target sequences, get array of lengths and pad with 0s to max length
			input_lengths = [len(s) for s in input_word_seqs]
			input_words_padded = [pad_seq(input_lang, s, max(input_lengths)) for s in input_word_seqs]
			input_prosody_padded = [pad_prosody_seq(s, max(input_lengths), n_prosody_params) for s in input_prosody_seqs]
			target_lengths = [len(s) for s in target_seqs]
			target_padded = [pad_seq(output_lang, s, max(target_lengths)) for s in target_seqs]

			# Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
			input_word_var = Variable(torch.LongTensor(input_words_padded)).transpose(0, 1)
			input_prosody_var = Variable(torch.FloatTensor(input_prosody_padded)).transpose(0, 1)
			target_var = Variable(torch.LongTensor(target_padded)).transpose(0, 1)

			if USE_CUDA:
				input_word_var = input_word_var.cuda()
				input_prosody_var = input_prosody_var.cuda()
				target_var = target_var.cuda()
			yield input_word_var, input_prosody_var, input_lengths, target_var, target_lengths
			input_word_seqs = []
			input_prosody_seqs = []
			target_seqs = []

def main(options):
	USE_CUDA = options.use_cuda
	print("Use cuda: %s" %USE_CUDA)

	try:
		with open(options.params_file, 'r') as ymlfile:
			config = yaml.load(ymlfile)
	except:
		sys.exit("Parameters file missing")

	if not checkArgument(options.resume_encoder, isFile=True):
		sys.exit("encoder model missing or not specified")
	if not checkArgument(options.resume_decoder, isFile=True):
		sys.exit("decoder model missing or not specified")

	AUDIO_TRAIN_DATA_FILE = config['AUDIO_TRAIN_DATA_FILE']
	AUDIO_VALIDATION_DATA_FILE = config['AUDIO_TRAIN_DATA_FILE']

	audio_train_data = read_audio_dataset_file(AUDIO_TRAIN_DATA_FILE, shuffle=False)
	audio_validation_data = read_audio_dataset_file(AUDIO_TRAIN_DATA_FILE, shuffle=False)

	w2v_model_es = gensim.models.Word2Vec.load(config["W2V_ES_PATH"])
	w2v_model_en = gensim.models.Word2Vec.load(config["W2V_EN_PATH"])

	BASE_VOCABULARY_SIZE_EN = config["BASE_VOCABULARY_SIZE_EN"]
	BASE_VOCABULARY_SIZE_ES = config["BASE_VOCABULARY_SIZE_ES"]

	INPUT_LANG_CODE = config['INPUT_LANG']
	OUTPUT_LANG_CODE = config['OUTPUT_LANG']

	if INPUT_LANG_CODE == 'en' and OUTPUT_LANG_CODE == 'es':
		lang_en = input_lang = Lang(INPUT_LANG_CODE, config["W2V_EN_PATH"], BASE_VOCABULARY_SIZE_EN, omit_punctuation=config["INPUT_LANG_OMIT_PUNC"])
		lang_es = output_lang = Lang(OUTPUT_LANG_CODE, config["W2V_ES_PATH"], BASE_VOCABULARY_SIZE_ES, omit_punctuation=config["OUTPUT_LANG_OMIT_PUNC"])
	elif INPUT_LANG_CODE == 'es' and OUTPUT_LANG_CODE == 'en':
		lang_es = input_lang = Lang(INPUT_LANG_CODE, config["W2V_ES_PATH"], BASE_VOCABULARY_SIZE_ES, omit_punctuation=config["INPUT_LANG_OMIT_PUNC"])
		lang_en = output_lang = Lang(OUTPUT_LANG_CODE, config["W2V_EN_PATH"], BASE_VOCABULARY_SIZE_EN, omit_punctuation=config["OUTPUT_LANG_OMIT_PUNC"])

	MAX_SEQ_LENGTH = int(config['MAX_SEQ_LENGTH'])
	TRAINING_BATCH_SIZE = int(config['AUDIO_TRAINING_BATCH_SIZE'])
	N_PROSODY_PARAMS = int(config['N_PROSODY_PARAMS'])
	input_prosody_params = config['INPUT_PROSODY']

	# Configure models
	attn_model = config['ATTN_MODEL']
	hidden_size = int(config['HIDDEN_SIZE'])
	n_layers = int(config['N_LAYERS'])
	dropout = float(config['DROPOUT'])
	encoder_type = config['ENCODER_TYPE']

	# Configure training/optimization
	clip = float(config['AUDIO_CLIP'])
	learning_rate = float(config['AUDIO_LEARNING_RATE'])
	decoder_learning_ratio = float(config['AUDIO_DECODER_LEARNING_RATIO'])
	n_epochs = int(config['AUDIO_N_EPOCHS'])
	patience_epochs = int(config['AUDIO_PATIENCE_EPOCHS'])
	print_every_batch = int(config['AUDIO_PRINT_EVERY_BATCH'])
	save_every_batch = int(config['AUDIO_SAVE_EVERY_BATCH'])
	training_data_size = len(audio_train_data)
	no_of_batches_in_epoch = training_data_size/TRAINING_BATCH_SIZE  #TODO

	# Initialize models
	if encoder_type == 'sum':
		encoder = EncoderRNN(input_lang.vocabulary_size, N_PROSODY_PARAMS, hidden_size, input_lang.get_weights_matrix(), n_layers, dropout=dropout)
	elif encoder_type == 'parallel':
		encoder = EncoderRNN_parallel(input_lang.vocabulary_size, N_PROSODY_PARAMS, hidden_size, input_lang.get_weights_matrix(), n_layers, dropout=dropout)
	else:
		sys.exit("Unrecognized encoder type. Check params file. Exiting...")
	decoder = LuongAttnDecoderRNN(attn_model, hidden_size, output_lang.vocabulary_size, n_layers, dropout=dropout)

	# Load states from models if given
	load_model(encoder, decoder, options.resume_encoder, options.resume_decoder)

	# Initialize optimizers and criterion
	encoder_optimizer = optim.Adam(filter(lambda p: p.requires_grad,encoder.parameters()), lr=learning_rate)
	decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
	# criterion = nn.CrossEntropyLoss()

	# Move models to GPU
	if USE_CUDA:
		encoder.cuda()
		decoder.cuda()

	# Keep track of time elapsed and running averages
	start = time.time()
	plot_losses = []
	validation_losses = []
	print_loss_total = 0 # Reset every print_every
	plot_loss_total = 0 # Reset every plot_every

	# Begin!
	ecs = []
	dcs = []
	eca = 0
	dca = 0

	epoch = 1
	
	while epoch <= n_epochs:
		training_batch = 0
		validation_batch = 0

		# Train 
		for input_word_batches, input_prosody_batches, input_lengths, target_batches, target_lengths in audio_batch_generator(audio_train_data, TRAINING_BATCH_SIZE, input_lang, output_lang, N_PROSODY_PARAMS, input_prosody_params, USE_CUDA=USE_CUDA):
			# Run the train function
			loss, ec, dc = train(input_word_batches=input_word_batches,
								 input_prosody_batches=input_prosody_batches, 
								 input_lengths=input_lengths, 
								 target_batches=target_batches, 
								 target_lengths=target_lengths,
								 input_lang=input_lang, 
								 output_lang=output_lang,
								 batch_size=TRAINING_BATCH_SIZE,
								 encoder=encoder, 
								 decoder=decoder,
								 clip=clip,
								 encoder_optimizer=encoder_optimizer, 
								 decoder_optimizer=decoder_optimizer, 
								 USE_CUDA=USE_CUDA)

			# Keep track of loss
			print_loss_total += loss
			plot_loss_total += loss
			eca += ec
			dca += dc
			training_batch += 1

			if training_batch % print_every_batch == 0:
				print_loss_avg = print_loss_total / print_every_batch
				print_loss_total = 0
				print_summary = '%s (Batch:%d/%d %d%%) (Epoch: %d/%d) Loss:%.4f' % (time_since(start, training_batch / no_of_batches_in_epoch), training_batch, no_of_batches_in_epoch, training_batch / no_of_batches_in_epoch * 100, epoch, n_epochs, print_loss_avg)
				print(print_summary)

			if training_batch % save_every_batch == 0:
				plot_loss_avg = float(plot_loss_total / save_every_batch)
				plot_losses.append(plot_loss_avg)
				plot_loss_total = 0

				#if plot_loss_avg <= min(plot_losses):
				print("Average loss of last %i batches: %f"%(save_every_batch, plot_loss_avg))
				save_model(encoder.state_dict(), decoder.state_dict(), plot_losses, options.model_name, options.model_dir, checkpoint=True)
			
		# Validate model on validation set
		print("VALIDATION ", end='')
		validation_loss_total = 0
		validation_batch = 0
		for input_word_batches, input_prosody_batches, input_lengths, target_batches, target_lengths in audio_batch_generator(audio_validation_data, 1, input_lang, output_lang, N_PROSODY_PARAMS, input_prosody_params, USE_CUDA=USE_CUDA):
			loss = validate( input_word_batches=input_word_batches,
							 input_prosody_batches=input_prosody_batches, 
							 input_lengths=input_lengths, 
							 target_batches=target_batches, 
							 target_lengths=target_lengths,
							 input_lang=input_lang, 
							 output_lang=output_lang,
							 batch_size=1,
							 encoder=encoder, 
							 decoder=decoder,
							 USE_CUDA=USE_CUDA)

			validation_loss_total += loss
			validation_batch += 1

		validation_loss_avg = float(validation_loss_total / validation_batch)
		validation_summary = 'at Epoch: %d/%d Average loss:%.4f' % (epoch, n_epochs, validation_loss_avg)
		print(validation_summary)

		# Stopping criteria: stop if validation loss didn't get better in last PATIENCE_EPOCHS 
		if len(validation_losses) == 0 or any([validation_loss_avg < loss for loss in validation_losses[-patience_epochs:]]):
			#Keep on training
			if len(validation_losses) == 0 or validation_loss_avg < min(validation_losses):
				save_model(encoder.state_dict(), decoder.state_dict(), plot_losses, options.model_name, options.model_dir)
			validation_losses.append(validation_loss_avg)
		else:
			print("Finished!")
			print("Best validation loss: %f"%min(validation_losses))
			break

		epoch += 1

if __name__ == "__main__":
	usage = "usage: %prog [-s infile] [option]"
	parser = OptionParser(usage=usage)
	parser.add_option("-m", "--modelname", dest="model_name", default=None, help="output model filename", type="string")
	parser.add_option("-d", "--modeldir", dest="model_dir", default=None, help="directory to store model", type="string")
	parser.add_option("-c", "--usecuda", dest="use_cuda", default=False, help="train on gpu", action="store_true")
	parser.add_option("-p", "--params", dest="params_file", default=None, help="params filename", type="string")
	parser.add_option("-e", "--resumeencoder", dest="resume_encoder", default=None, help="encoder model to resume training from", type="string")
	parser.add_option("-r", "--resumedecoder", dest="resume_decoder", default=None, help="decoder model to resume training from", type="string")

	(options, args) = parser.parse_args()
	main(options)