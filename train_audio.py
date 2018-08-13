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
from train_text import *
import yaml
import csv
import random

PROSODY_FEATURE_INDEXES = {'pause_after':0, 'f0_mean':1, 'i0_mean':2}

WORD_LOSS_WEIGHT_INITIAL = 1.0
PAUSEFLAG_LOSS_WEIGHT_INITIAL = 10.0
PAUSEVALUE_LOSS_WEIGHT_INITIAL = 10.0

NO_OF_LOSSES = 4

TRAINING_LOSS_LOG_COLUMNS = ["LOSS", "W-word", "L-word", "W-pauseflag", "L-pauseflag", "W-pausevalue", "L-pausevalue"]
VALIDATION_LOSS_LOG_COLUMNS = ["LOSS", "L-word", "L-pauseflag", "L-pausevalue"]

EARLY_STOP = True

#Debug flags
DO_TRAIN = True  #FOR DEBUGGING: If false, no training is performed dry run on batches
AUDIO_ENCODE_ONLY = False
DUMMY_PROSODY_INPUT = False	
DUMMY_PROSODY_OUTPUT = False
DEBUG_PRINT_SAMPLES = False
DEBUG_PRINT_LOSSES = False

def audio_batch_generator(audio_input_data, audio_output_data, batch_size, input_lang, output_lang, input_prosody_params, output_prosody_params, prosody_mins, prosody_maxs, input_prosody_norms, output_prosody_norms, n_prosody_params, USE_CUDA=False):
	assert not input_lang == output_lang
	input_word_seqs = []
	input_prosody_seqs = []
	target_word_seqs = []
	target_prosody_seqs = []
	target_flag_seqs = []

	input_prosody_mins = [prosody_mins[prosody_param] for prosody_param in input_prosody_params]
	input_prosody_maxs = [prosody_maxs[prosody_param] for prosody_param in input_prosody_params]
	input_prosody_norms = [input_prosody_norms[prosody_param] for prosody_param in input_prosody_params]
	output_prosody_mins = [prosody_mins[prosody_param] for prosody_param in output_prosody_params]
	output_prosody_maxs = [prosody_maxs[prosody_param] for prosody_param in output_prosody_params]
	output_prosody_norms = [output_prosody_norms[prosody_param] for prosody_param in output_prosody_params]

	for (input_word_tokens, input_prosody_tokens, input_csv), (target_word_tokens, target_prosody_tokens, target_csv) in zip(audio_input_data, audio_output_data):
		input_word_indexes = indexes_from_tokens(input_lang, input_word_tokens)	#comes with END token
		input_prosody_tokens = finalize_prosody_sequence(input_prosody_tokens) #adds the END token
		#input_prosody_tokens_normalized = normalize_prosody(input_prosody_tokens, input_prosody_mins, input_prosody_maxs)  #no conversion of zeros to norm
		
		target_word_indexes = indexes_from_tokens(output_lang, target_word_tokens) #comes with END token
		target_prosody_tokens = finalize_prosody_sequence(target_prosody_tokens) #adds the END token
		target_flag_tokens = flags_from_prosody(target_prosody_tokens)
		#target_prosody_tokens_normalized = normalize_prosody(target_prosody_tokens, output_prosody_mins, output_prosody_maxs, output_prosody_norms, target_flag_tokens)

		if DEBUG_PRINT_SAMPLES:
			print('input csv: ', input_csv)
			print(input_word_tokens)
			print_prosody(input_prosody_tokens)
			print('target csv: ', target_csv)
			print(target_word_tokens)
			print_prosody(target_prosody_tokens)
			print_prosody(target_flag_tokens)

		input_word_seqs.append(input_word_indexes)
		input_prosody_seqs.append(input_prosody_tokens)
		target_word_seqs.append(target_word_indexes)
		target_prosody_seqs.append(target_prosody_tokens)
		target_flag_seqs.append(target_flag_tokens)

		if len(input_word_seqs) == batch_size:
			if DEBUG_PRINT_SAMPLES:
				exit = input('...')
				if exit == 'q':
					break

			# Zip into pairs, sort by length (descending), unzip
			seq_pairs = sorted(zip(input_word_seqs, input_prosody_seqs, target_word_seqs, target_prosody_seqs, target_flag_seqs), key=lambda p: len(p[0]), reverse=True)
			input_word_seqs, input_prosody_seqs, target_word_seqs, target_prosody_seqs, target_flag_seqs = zip(*seq_pairs)

			# For input and target sequences, get array of lengths and pad with 0s to max length
			input_lengths = [len(s) for s in input_word_seqs]
			input_words_padded = [pad_seq(input_lang, s, max(input_lengths)) for s in input_word_seqs]
			input_prosody_padded = [pad_prosody_seq(s, max(input_lengths), input_prosody_norms) for s in input_prosody_seqs]
			target_lengths = [len(s) for s in target_word_seqs]
			target_words_padded = [pad_seq(output_lang, s, max(target_lengths)) for s in target_word_seqs]
			target_prosody_padded = [pad_prosody_seq(s, max(target_lengths), output_prosody_norms) for s in target_prosody_seqs]
			target_flag_padded = [pad_flag_seq(s, max(target_lengths), n_prosody_params) for s in target_flag_seqs]

			if DEBUG_PRINT_SAMPLES:
				print("padded input")
				print(input_prosody_padded)
				print("padded target")
				print(target_prosody_padded)

			#normalize continuous values
			input_prosody_padded_normalized = [normalize_prosody(s, input_prosody_mins, input_prosody_maxs) for s in input_prosody_padded] #no conversion of zeros to norm
			target_prosody_padded_normalized = [normalize_prosody(s, output_prosody_mins, output_prosody_maxs, output_prosody_norms, f) for s,f in zip(target_prosody_padded, target_flag_seqs)]

			if DEBUG_PRINT_SAMPLES:
				print("normalized input")
				print(input_prosody_padded_normalized)
				print("normalized target")
				print(target_prosody_padded_normalized)

			# Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
			input_word_var = Variable(torch.LongTensor(input_words_padded)).transpose(0, 1)
			input_prosody_var = Variable(torch.FloatTensor(input_prosody_padded_normalized)).transpose(0, 1)
			target_word_var = Variable(torch.LongTensor(target_words_padded)).transpose(0, 1)
			target_prosody_var = Variable(torch.FloatTensor(target_prosody_padded_normalized)).transpose(0, 1)
			target_flag_var = Variable(torch.LongTensor(target_flag_padded)).transpose(0, 1)

			if DEBUG_PRINT_SAMPLES:
				print(input_word_var.shape)
				print(input_prosody_var.shape)

			if USE_CUDA:
				input_word_var = input_word_var.cuda()
				input_prosody_var = input_prosody_var.cuda()
				target_word_var = target_word_var.cuda()
				target_prosody_var = target_prosody_var.cuda()
				target_flag_var = target_flag_var.cuda()

			yield input_word_var, input_prosody_var, input_lengths, target_word_var, target_prosody_var, target_flag_var, target_lengths
			input_word_seqs = []
			input_prosody_seqs = []
			target_word_seqs = []
			target_prosody_seqs = []
			target_flag_seqs = []

def run_forward_audio(input_word_batch, input_prosody_batch, input_lengths, target_word_batch, target_prosody_batch, target_flag_batch, target_lengths, input_lang, output_lang, batch_size, encoder, decoder, mse_criterion, word_loss_weight, pauseflag_loss_weight, pausevalue_loss_weight, audio_encode_only, USE_CUDA):
	loss_word = 0 # Added onto for each word
	target_pauseflag_batch = 0
	loss_pauseflag = 0
	loss_total = 0

	# print('input_word, ', input_word_batch)
	# print('target word, ', target_word_batch)

	# Run words and prosody through encoder
	encoder_outputs, encoder_hidden = encoder(input_word_batch, input_prosody_batch, input_lengths, None)

	# Prepare input and output variables
	decoder_word_input = Variable(torch.LongTensor([output_lang.token2index(SWT_TOKEN)] * batch_size))
	decoder_hidden = encoder_hidden[:decoder.n_layers] # Use last (forward) hidden state from encoder
	decoder_context = Variable(torch.zeros(batch_size, decoder.hidden_size))

	max_target_length = max(target_lengths)
	all_decoder_outputs_word = Variable(torch.zeros(max_target_length, batch_size, decoder.output_size))
	all_decoder_outputs_pauseflag = Variable(torch.zeros(max_target_length, batch_size, 2))
	all_decoder_outputs_pausevalue = Variable(torch.zeros(max_target_length, batch_size, 1))

	# Move new Variables to CUDA
	if USE_CUDA:
		decoder_word_input = decoder_word_input.cuda()
		decoder_context = decoder_context.cuda()
		all_decoder_outputs_word = all_decoder_outputs_word.cuda()
		all_decoder_outputs_pauseflag = all_decoder_outputs_pauseflag.cuda()
		all_decoder_outputs_pausevalue = all_decoder_outputs_pausevalue.cuda()

	# Run through decoder one time step at a time
	for t in range(max_target_length):
		if audio_encode_only:
			decoder_output_word, decoder_context, decoder_hidden, decoder_attn = decoder(
				decoder_word_input, decoder_context, decoder_hidden, encoder_outputs)
		else:
			decoder_output_word, decoder_output_pauseflag, decoder_output_pausevalue, decoder_context, decoder_hidden, decoder_attn = decoder(
				decoder_word_input, decoder_context, decoder_hidden, encoder_outputs
			)
		
		all_decoder_outputs_word[t] = decoder_output_word # Store this step's outputs
		if not audio_encode_only:
			all_decoder_outputs_pauseflag[t] = decoder_output_pauseflag
			all_decoder_outputs_pausevalue[t] = decoder_output_pausevalue

		decoder_word_input = target_word_batch[t] # Next input is current target

	# Loss calculation and backpropagation
	loss_word = masked_cross_entropy(
		all_decoder_outputs_word.transpose(0, 1).contiguous(),
		target_word_batch.transpose(0, 1).contiguous(),
		target_lengths, 
		use_cuda=USE_CUDA
	)
	if audio_encode_only:
		loss_total = loss_word
		loss_word = loss_word
		loss_pauseflag = None
		loss_pausevalue = None
	else:
		target_pauseflag_batch = target_flag_batch[:,:,0]
		# print('pauseflag pred,\n', all_decoder_outputs_pauseflag)
		# print('pauseflag gold,\n', target_pauseflag_batch)

		loss_pauseflag = masked_cross_entropy(
			all_decoder_outputs_pauseflag.transpose(0, 1).contiguous(),
			target_pauseflag_batch.transpose(0, 1).contiguous(),
			target_lengths, 
			use_cuda=USE_CUDA
		)
		# print('pauseflag loss\n', loss_pauseflag)

		loss_pausevalue = mse_criterion(all_decoder_outputs_pausevalue.squeeze(-1), target_prosody_batch[:,:,PROSODY_FEATURE_INDEXES['pause_after']])

		loss_total, loss_word, loss_pauseflag, loss_pausevalue = calculate_loss_combination([word_loss_weight, pauseflag_loss_weight, pausevalue_loss_weight], [loss_word, loss_pauseflag, loss_pausevalue])

		if DEBUG_PRINT_LOSSES:
			print('pausevalue pred,\n', all_decoder_outputs_pausevalue.squeeze(-1))
			print('pausevalue gold,\n', target_prosody_batch[:,:,PROSODY_FEATURE_INDEXES['pause_after']])
			print('pausevalue loss,\n',loss_pausevalue)
			print('pauseflag loss,\n',loss_pauseflag)
			print('loss_total,\n',loss_total)

			exit = input('...')
			if exit == 'q':
				sys.exit()

	return loss_total, loss_word, loss_pauseflag, loss_pausevalue

def calculate_loss_combination_regularized(weights, loss_values):
	assert len(weights) == len(loss_values)
	loss_comb = 0
	weighted_losses = []
	for w, l in zip(weights, loss_values):
		weighted_loss = (1/(2*w**2))*l + math.log(1 + w**2)
		loss_comb += weighted_loss
		weighted_losses.append(weighted_loss)
	return loss_comb, weighted_losses[0], weighted_losses[1], weighted_losses[2]

def calculate_loss_combination(weights, loss_values):
	assert len(weights) == len(loss_values)
	loss_comb = 0
	weighted_losses = []
	for w, l in zip(weights, loss_values):
		weighted_loss = w*l
		loss_comb += weighted_loss
		weighted_losses.append(weighted_loss)
	return loss_comb, weighted_losses[0], weighted_losses[1], weighted_losses[2]

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
	AUDIO_VALIDATION_DATA_FILE = config['AUDIO_VALIDATION_DATA_FILE']

	INPUT_LANG_CODE = config['INPUT_LANG']
	OUTPUT_LANG_CODE = config['OUTPUT_LANG']

	MAX_SEQ_LENGTH = int(config['MAX_SEQ_LENGTH'])
	TRAINING_BATCH_SIZE = int(config['AUDIO_TRAINING_BATCH_SIZE'])
	N_PROSODY_PARAMS = int(config['N_PROSODY_PARAMS'])
	input_prosody_params = config['INPUT_PROSODY']
	if input_prosody_params == None:
		input_prosody_params = []
	output_prosody_params = config['OUTPUT_PROSODY']
	if output_prosody_params == None:
		output_prosody_params = []

	print("IN Prosody: ", input_prosody_params)
	print("OUT Prosody: ", output_prosody_params)

	if INPUT_LANG_CODE == 'en' and OUTPUT_LANG_CODE == 'es':
		lang_en = input_lang = Lang(INPUT_LANG_CODE, config["W2V_EN_PATH"], config["DICT_EN_PATH"], punctuation_level=config["INPUT_LANG_PUNC_LEVEL"])
		lang_es = output_lang = Lang(OUTPUT_LANG_CODE, config["W2V_ES_PATH"], config["DICT_ES_PATH"], punctuation_level=config["OUTPUT_LANG_PUNC_LEVEL"])
		input_prosody_norms = config['PROSODY_FEATURE_NORMS_EN']
		output_prosody_norms = config['PROSODY_FEATURE_NORMS_ES']
	elif INPUT_LANG_CODE == 'es' and OUTPUT_LANG_CODE == 'en':
		lang_es = input_lang = Lang(INPUT_LANG_CODE, config["W2V_ES_PATH"], config["DICT_ES_PATH"], punctuation_level=config["INPUT_LANG_PUNC_LEVEL"])
		lang_en = output_lang = Lang(OUTPUT_LANG_CODE, config["W2V_EN_PATH"], config["DICT_EN_PATH"], punctuation_level=config["OUTPUT_LANG_PUNC_LEVEL"])
		input_prosody_norms = config['PROSODY_FEATURE_NORMS_ES']
		output_prosody_norms = config['PROSODY_FEATURE_NORMS_EN']

	print("Loading audio data...", end='')
	audio_train_input_data, audio_train_output_data = load_audio_dataset(AUDIO_TRAIN_DATA_FILE, input_lang, output_lang, N_PROSODY_PARAMS, input_prosody_params, output_prosody_params, dummyfy_input_prosody=DUMMY_PROSODY_INPUT, dummyfy_output_prosody=DUMMY_PROSODY_OUTPUT)
	audio_validation_input_data, audio_validation_output_data = load_audio_dataset(AUDIO_VALIDATION_DATA_FILE, input_lang, output_lang, N_PROSODY_PARAMS, input_prosody_params, output_prosody_params, dummyfy_input_prosody=DUMMY_PROSODY_INPUT, dummyfy_output_prosody=DUMMY_PROSODY_OUTPUT)
	print("DONE.")

	prosody_mins = config['PROSODY_FEATURE_MINS']
	prosody_maxs = config['PROSODY_FEATURE_MAXS']
	
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
	training_data_size = len(audio_train_input_data)
	no_of_batch_in_epoch = training_data_size/TRAINING_BATCH_SIZE  

	# Initialize models
	if encoder_type == 'sum':
		encoder = EncoderRNN_sum(input_lang.vocabulary_size, N_PROSODY_PARAMS, hidden_size, input_lang.get_weights_matrix(), n_layers, dropout=dropout)
	elif encoder_type == 'parallel':
		encoder = EncoderRNN_parallel(input_lang.vocabulary_size, N_PROSODY_PARAMS, hidden_size, input_lang.get_weights_matrix(), n_layers, dropout=dropout)
	else:
		sys.exit("Unrecognized encoder type. Check params file. Exiting...")
	if AUDIO_ENCODE_ONLY:
		decoder = LuongAttnDecoderRNN(attn_model, hidden_size, output_lang.get_weights_matrix(), output_lang.vocabulary_size, n_layers, dropout=dropout, USE_CUDA=USE_CUDA)
	else:
		decoder = ProsodicDecoderRNN(attn_model, hidden_size, output_lang.get_weights_matrix(), output_lang.vocabulary_size, n_layers, dropout=dropout, USE_CUDA=USE_CUDA)

	# Load states from models if given
	load_model(encoder, decoder, options.resume_encoder, options.resume_decoder, gpu_to_cpu=options.gpu2cpu)

	# Initialize loss weights
	word_loss_weight = torch.tensor(WORD_LOSS_WEIGHT_INITIAL, requires_grad=False)
	pauseflag_loss_weight = torch.tensor(PAUSEFLAG_LOSS_WEIGHT_INITIAL, requires_grad=False)
	pausevalue_loss_weight = torch.tensor(PAUSEVALUE_LOSS_WEIGHT_INITIAL, requires_grad=False)
	loss_weight_generator = (weight for weight in [word_loss_weight, pauseflag_loss_weight, pausevalue_loss_weight])

	# Initialize optimizers and criterion
	encoder_optimizer = optim.Adam(filter(lambda p: p.requires_grad,encoder.parameters()), lr=learning_rate)
	decoder_optimizer = optim.Adam(filter(lambda p: p.requires_grad,decoder.parameters()), lr=learning_rate * decoder_learning_ratio)
	#loss_weights_optimizer = optim.Adam(loss_weight_generator, lr=learning_rate)
	mse_criterion = nn.MSELoss(size_average=False)

	# Move models to GPU
	if USE_CUDA:
		encoder.cuda()
		decoder.cuda()

	if DO_TRAIN:
		# Keep track of time elapsed and running averages
		start = time.time()
		training_losses = []
		validation_losses = []
		print_loss_totals = [0.0] * NO_OF_LOSSES # Reset every print_every
		save_loss_totals = [0.0] * NO_OF_LOSSES # Reset every plot_every

		#Initialize the loss log file
		if options.train_log_file:
			initialize_log_loss(options.train_log_file, TRAINING_LOSS_LOG_COLUMNS)
		if options.validation_log_file:
			initialize_log_loss(options.validation_log_file, VALIDATION_LOSS_LOG_COLUMNS)

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
			for batch in audio_batch_generator(audio_train_input_data, audio_train_output_data, TRAINING_BATCH_SIZE, input_lang, output_lang, input_prosody_params, output_prosody_params, prosody_mins, prosody_maxs, input_prosody_norms, output_prosody_norms, n_prosody_params=N_PROSODY_PARAMS, USE_CUDA=USE_CUDA):
				input_word_batch, input_prosody_batch, input_lengths, target_word_batch, target_prosody_batch, target_flag_batch, target_lengths = batch
				
				# for i, token in enumerate(target_word_batch):
				# 	print("%s - %s - %s"%(token, target_prosody_batch[i], target_flag_batch[i]))


				# Run the train function
				loss_values, ec, dc = train( input_word_batch=input_word_batch,
											 input_prosody_batch=input_prosody_batch, 
											 input_lengths=input_lengths, 
											 target_word_batch=target_word_batch,
											 target_prosody_batch=target_prosody_batch,
											 target_flag_batch=target_flag_batch, 
											 target_lengths=target_lengths,
											 input_lang=input_lang, 
											 output_lang=output_lang,
											 batch_size=TRAINING_BATCH_SIZE,
											 encoder=encoder, 
											 decoder=decoder,
											 clip=clip,
											 encoder_optimizer=encoder_optimizer, 
											 decoder_optimizer=decoder_optimizer, 
											 run_forward_func=run_forward_audio,
											 mse_criterion=mse_criterion,
											 word_loss_weight=word_loss_weight,
											 pauseflag_loss_weight=pauseflag_loss_weight,
											 pausevalue_loss_weight=pausevalue_loss_weight,
											 loss_weights_optimizer=None,
											 audio_encode_only=AUDIO_ENCODE_ONLY,
											 USE_CUDA=USE_CUDA)

				# inp = input("...")
				# if inp == 'q':
				# 	sys.exit()

				# Keep track of losses
				print_loss_totals = [print_loss_totals[i] + loss_values[i] for i in range(len(loss_values))]
				save_loss_totals = [save_loss_totals[i] + loss_values[i] for i in range(len(loss_values))]
				eca += ec
				dca += dc
				training_batch += 1

				if training_batch % print_every_batch == 0:
					print_loss_avgs = np.array(print_loss_totals) / print_every_batch
					print_loss_totals = [0] * NO_OF_LOSSES
					if AUDIO_ENCODE_ONLY:
						print_summary = '%s (Batch:%d/%d %d%%) (Epoch: %d/%d) Loss:%.4f' % (time_since(start, training_batch / no_of_batch_in_epoch), training_batch, no_of_batch_in_epoch, training_batch / no_of_batch_in_epoch * 100, epoch, n_epochs, print_loss_avgs[0])
						loss_report = [print_loss_avgs[0], word_loss_weight.item()]
					else:
						print_summary = '%s (Batch:%d/%d %d%%) (Epoch: %d/%d) Loss:%.4f (word: %.4f, pflag: %.4f, pval: %.4f)' % (time_since(start, training_batch / no_of_batch_in_epoch), training_batch, no_of_batch_in_epoch, training_batch / no_of_batch_in_epoch * 100, epoch, n_epochs, print_loss_avgs[0], print_loss_avgs[1], print_loss_avgs[2], print_loss_avgs[3])
						loss_report = [print_loss_avgs[0], word_loss_weight.item(), print_loss_avgs[1], pauseflag_loss_weight.item(), print_loss_avgs[2], pausevalue_loss_weight.item(), print_loss_avgs[3]]
					
					print(print_summary)
					log_loss(options.train_log_file, loss_report)

				if training_batch % save_every_batch == 0:
					save_loss_avgs = np.array(save_loss_totals) / save_every_batch
					training_losses.append(save_loss_avgs)
					save_loss_totals = [0.0] * NO_OF_LOSSES

					#if plot_loss_avg <= min(training_losses):
					print("Average loss of last %i batches: %f"%(save_every_batch, save_loss_avgs[0]))
					save_model(encoder.state_dict(), decoder.state_dict(), options.model_name, options.model_dir, checkpoint=True)
				

			# Validate model on validation set
			print("VALIDATION ", end='')
			validation_loss_totals = [0] * NO_OF_LOSSES
			validation_batch = 0
			for batch in audio_batch_generator(audio_validation_input_data, audio_validation_output_data, 1, input_lang, output_lang, input_prosody_params, output_prosody_params, prosody_mins, prosody_maxs, input_prosody_norms, output_prosody_norms, n_prosody_params=N_PROSODY_PARAMS, USE_CUDA=USE_CUDA):
				input_word_batch, input_prosody_batch, input_lengths, target_word_batch, target_prosody_batch, target_flag_batch, target_lengths = batch
				
				loss_values = validate(  input_word_batch=input_word_batch,
										 input_prosody_batch=input_prosody_batch, 
										 input_lengths=input_lengths, 
										 target_word_batch=target_word_batch,
										 target_prosody_batch=target_prosody_batch,
										 target_flag_batch=target_flag_batch, 
										 target_lengths=target_lengths,
										 input_lang=input_lang, 
										 output_lang=output_lang,
										 batch_size=1,
										 encoder=encoder, 
										 decoder=decoder,
										 run_forward_func=run_forward_audio,
										 mse_criterion=mse_criterion,
										 word_loss_weight=word_loss_weight,
										 pauseflag_loss_weight=pauseflag_loss_weight,
										 pausevalue_loss_weight=pausevalue_loss_weight,
										 audio_encode_only=AUDIO_ENCODE_ONLY,
										 USE_CUDA=USE_CUDA)

				validation_loss_totals = [validation_loss_totals[i] + loss_values[i] for i in range(len(loss_values))]
				validation_batch += 1

			validation_loss_avgs = np.array(validation_loss_totals) / validation_batch
			if AUDIO_ENCODE_ONLY:
				validation_summary = 'at Epoch: %d/%d Average Loss:%.4f (word: %.4f)' % (epoch, n_epochs, validation_loss_avgs[0], validation_loss_avgs[1])
			else:
				validation_summary = 'at Epoch: %d/%d Average Loss:%.4f (word: %.4f, pauseflag: %.4f, pauseval:%.4f)' % (epoch, n_epochs, validation_loss_avgs[0], validation_loss_avgs[1], validation_loss_avgs[2], validation_loss_avgs[3])
			print(validation_summary)

			if (options.validation_log_file):
				if AUDIO_ENCODE_ONLY:
					loss_report = [validation_loss_avgs[0]]
				else:
					loss_report = [validation_loss_avgs[0], validation_loss_avgs[1], validation_loss_avgs[2], validation_loss_avgs[3]]

				log_loss(options.validation_log_file, loss_report)
			
			if EARLY_STOP:
				# Stopping criteria: stop if validation loss didn't get better in last PATIENCE_EPOCHS 
				if len(validation_losses) == 0 or any([validation_loss_avgs[0] < loss for loss in validation_losses[-patience_epochs:][0]]):
					#Keep on training
					#save model if it's better than the previous ones (in terms of general loss)
					validation_losses_np = np.array(validation_losses)
					if len(validation_losses) == 0 or validation_loss_avgs[0] < min(validation_losses_np[:,0]):
						save_model(encoder.state_dict(), decoder.state_dict(), options.model_name, options.model_dir)
					validation_losses.append(validation_loss_avgs)
				else:
					print("Finished!")
					validation_losses_np = np.array(validation_losses)
					print("Best validation loss: %f"%min(validation_losses_np[:,0]))
					break
			else:
				# Train forever
				validation_losses_np = np.array(validation_losses)
				save_model(encoder.state_dict(), decoder.state_dict(), options.model_name, options.model_dir)
				validation_losses.append(validation_loss_avgs)

			epoch += 1
	else:
		all_pause_values_en = []
		all_pause_values_es = []
		for batch in audio_batch_generator(audio_train_input_data, audio_train_output_data, TRAINING_BATCH_SIZE, input_lang, output_lang, input_prosody_params, output_prosody_params, prosody_mins, prosody_maxs, input_prosody_norms, output_prosody_norms, n_prosody_params=N_PROSODY_PARAMS, USE_CUDA=USE_CUDA):
				input_word_batch, input_prosody_batch, input_lengths, target_word_batch, target_prosody_batch, target_flag_batch, target_lengths = batch

				# print('input_word_batch', input_word_batch)
				# print('input_prosody_batch', input_prosody_batch)

				# print('target_word_batch', target_word_batch)
				# print('target_prosody_batch', target_prosody_batch)


				all_pause_values_en += input_prosody_batch.squeeze().tolist()
				all_pause_values_es += target_prosody_batch.squeeze().tolist()

				# inp = input("...")
				# if inp == 'q':
				# 	sys.exit()

		with open('all_pause_values_en.txt', 'w') as f:
			for item in all_pause_values_en:
				f.write("%s\n" % int(item*1000))

		with open('all_pause_values_es.txt', 'w') as f:
			for item in all_pause_values_es:
				f.write("%s\n" % int(item*1000))


if __name__ == "__main__":
	usage = "usage: %prog [-s infile] [option]"
	parser = OptionParser(usage=usage)
	parser.add_option("-m", "--modelname", dest="model_name", default=None, help="output model filename", type="string")
	parser.add_option("-d", "--modeldir", dest="model_dir", default=None, help="directory to store model", type="string")
	parser.add_option("-c", "--usecuda", dest="use_cuda", default=False, help="train on gpu", action="store_true")
	parser.add_option("-g", "--gpu2cpu", dest="gpu2cpu", default=False, help="load gpu model on cpu", action="store_true")
	parser.add_option("-p", "--params", dest="params_file", default=None, help="params filename", type="string")
	parser.add_option("-e", "--resumeencoder", dest="resume_encoder", default=None, help="encoder model to resume training from", type="string")
	parser.add_option("-r", "--resumedecoder", dest="resume_decoder", default=None, help="decoder model to resume training from", type="string")
	parser.add_option("-t", "--tlogfile", dest="train_log_file", default=None, help="Log file to output training losses", type="string")
	parser.add_option("-v", "--vlogfile", dest="validation_log_file", default=None, help="Log file to output validation losses", type="string")

	(options, args) = parser.parse_args()

	main(options)