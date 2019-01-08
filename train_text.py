from optparse import OptionParser
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
import yaml
import csv

LOSS_LOG_COLUMNS = ["Loss"]

#Debug flags
DO_TRAIN = True  #FOR DEBUGGING: If false, no training is performed

#generates batch from given parallel data file... 
#parallel data file has tab separated english and spanish sentence pairs (tokenized) at each line
def batch_generator(data_path, batch_size, input_lang, output_lang, tokenize=False, USE_CUDA=False):
	input_word_seqs = []
	target_word_seqs = []
	line_no = 0
	
	with open(data_path,'r', encoding='utf-8') as inputfile:
		for line in inputfile:
			line_no += 1

			pair = [sentence.strip() for sentence in line.split('\t')]
			sentence_en = pair[0]
			sentence_es = pair[1]


			if input_lang.lang_code == 'en' and output_lang.lang_code == 'es':
				input_sentence = sentence_en
				output_sentence = sentence_es
			elif input_lang.lang_code == 'es' and output_lang.lang_code == 'en':
				input_sentence = sentence_en
				output_sentence = sentence_es

			if tokenize:
				input_tokens = input_lang.tokenize(input_sentence, to_lower = True)
				output_tokens = output_lang.tokenize(output_sentence, to_lower = True)
			else:
				input_tokens = input_sentence.lower().split()
				output_tokens = output_sentence.lower().split()

			if input_lang.punctuation_level == 0:
				input_tokens = remove_punc_tokens(input_tokens)
			elif input_lang.punctuation_level == 1:
				input_tokens = remove_punc_tokens(input_tokens, keep_main_puncs=True)
			if output_lang.punctuation_level == 0:
				output_tokens = remove_punc_tokens(output_tokens)
			elif output_lang.punctuation_level == 1:
				output_tokens = remove_punc_tokens(output_tokens, keep_main_puncs=True)

			input_word_seqs.append(indexes_from_tokens(input_lang, input_tokens))
			target_word_seqs.append(indexes_from_tokens(output_lang, output_tokens))

			if len(input_word_seqs) == batch_size:
				# Zip into pairs, sort by length (descending), unzip
				seq_pairs = sorted(zip(input_word_seqs, target_word_seqs), key=lambda p: len(p[0]), reverse=True)
				input_word_seqs, target_word_seqs = zip(*seq_pairs)

				# For input and target sequences, get array of lengths and pad with 0s to max length
				input_lengths = [len(s) for s in input_word_seqs]
				input_words_padded = [pad_seq(input_lang, s, max(input_lengths)) for s in input_word_seqs]
				target_lengths = [len(s) for s in target_word_seqs]
				target_padded = [pad_seq(output_lang, s, max(target_lengths)) for s in target_word_seqs]

				# Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
				input_word_var = Variable(torch.LongTensor(input_words_padded)).transpose(0, 1)
				target_word_var = Variable(torch.LongTensor(target_padded)).transpose(0, 1)

				if USE_CUDA:
					input_word_var = input_word_var.cuda()
					target_word_var = target_word_var.cuda()

				#yield input_word_var, input_prosody_var, input_lengths, target_var, target_lengths, line_no
				yield input_word_var, input_lengths, target_word_var, target_lengths
				input_word_seqs = []
				target_word_seqs = []

#Uses generic encoder/decoder returns word loss
def run_forward_text(input_word_batch, input_prosody_batch, input_lengths, target_word_batch, target_prosody_batch, target_flag_batch, target_lengths, input_lang, output_lang, batch_size, encoder, decoder, mse_criterion, word_loss_weight=None, pauseflag_loss_weight=None, pausevalue_loss_weight=None, audio_encode_only=False, USE_CUDA=False):
	loss = 0 # Added onto for each word

	# Run words through encoder
	encoder_outputs, encoder_hidden = encoder(input_word_batch, input_lengths, None)
	
	# Prepare input and output variables
	decoder_input = Variable(torch.LongTensor([output_lang.token2index(SWT_TOKEN)] * batch_size))
	decoder_hidden = encoder_hidden[:decoder.n_layers] # Use last (forward) hidden state from encoder
	decoder_context = Variable(torch.zeros(batch_size, decoder.hidden_size))

	max_target_length = max(target_lengths)
	all_decoder_outputs = Variable(torch.zeros(max_target_length, batch_size, decoder.output_size))

	# Move new Variables to CUDA
	if USE_CUDA:
		decoder_input = decoder_input.cuda()
		all_decoder_outputs = all_decoder_outputs.cuda()
		decoder_context = decoder_context.cuda()

	# Run through decoder one time step at a time
	for t in range(max_target_length):
		decoder_output, decoder_context, decoder_hidden, decoder_attn = decoder(
		decoder_input, decoder_context, decoder_hidden, encoder_outputs)

		all_decoder_outputs[t] = decoder_output
		decoder_input = target_word_batch[t] # Next input is current target

	# Loss calculation and backpropagation
	loss = masked_cross_entropy(
		all_decoder_outputs.transpose(0, 1).contiguous(), # -> batch x seq
		target_word_batch.transpose(0, 1).contiguous(), # -> batch x seq
		target_lengths,
		use_cuda = USE_CUDA
	)

	return loss, None , None # , None #FLAGTEST

def train(input_word_batch, input_prosody_batch, input_lengths, target_word_batch, target_prosody_batch, target_flag_batch, target_lengths, input_lang, output_lang, batch_size, encoder, decoder, clip, encoder_optimizer, decoder_optimizer, run_forward_func, mse_criterion=None, word_loss_weight=None, pauseflag_loss_weight=None, pausevalue_loss_weight=None, loss_weights_optimizer=None, audio_encode_only=False, USE_CUDA=False):
	# Set models to train
	encoder.train()
	decoder.train()

	# Zero gradients of both optimizers
	encoder_optimizer.zero_grad()
	decoder_optimizer.zero_grad()
	if loss_weights_optimizer is not None:
		loss_weights_optimizer.zero_grad()

	losses = run_forward_func(input_word_batch, input_prosody_batch, input_lengths, target_word_batch, target_prosody_batch, target_flag_batch, target_lengths, input_lang, output_lang, batch_size, encoder, decoder, mse_criterion, word_loss_weight, pauseflag_loss_weight, pausevalue_loss_weight, audio_encode_only, USE_CUDA)
	#[loss_total, loss_word, loss_pauseflag, loss_pausevalue] = losses #FLAGTEST
	[loss_total, loss_word, loss_pauseflag] = losses

	# Backpropagate
	loss_total.backward()
	
	# Clip gradient norms
	ec = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
	dc = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)
	if loss_weights_optimizer is not None:
		loss_weight_generator = (weight for weight in [word_loss_weight, pauseflag_loss_weight, pausevalue_loss_weight])
		lc = torch.nn.utils.clip_grad_norm_(loss_weight_generator, clip) 

	# Update parameters with optimizers
	encoder_optimizer.step()
	decoder_optimizer.step()
	if loss_weights_optimizer is not None:
		loss_weights_optimizer.step()

	loss_values = [loss_type.item() for loss_type in losses if loss_type is not None]
	
	return loss_values, ec, dc

def validate(input_word_batch, input_prosody_batch, input_lengths, target_word_batch, target_prosody_batch, target_flag_batch, target_lengths, input_lang, output_lang, batch_size, encoder, decoder, run_forward_func, mse_criterion=None, word_loss_weight=None, pauseflag_loss_weight=None, pausevalue_loss_weight=None, audio_encode_only=False, USE_CUDA=False):
	# Disable training, set to evaluation
	encoder.eval()
	decoder.eval()

	# Zero gradients of both optimizers
	dev_loss = 0

	losses = run_forward_func(input_word_batch, input_prosody_batch, input_lengths, target_word_batch, target_prosody_batch, target_flag_batch, target_lengths, input_lang, output_lang, batch_size, encoder, decoder, mse_criterion, word_loss_weight, pauseflag_loss_weight, pausevalue_loss_weight, audio_encode_only, USE_CUDA)
	#[loss_total, loss_word, loss_pauseflag, loss_pausevalue] = losses #FLAGTEST
	[loss_total, loss_word, loss_pauseflag] = losses
	
	loss_values = [loss_type.item() for loss_type in losses if loss_type is not None]

	return loss_values

def main(options):
	USE_CUDA = options.use_cuda
	print("Use cuda: %s" %USE_CUDA)

	try:
		with open(options.params_file, 'r') as ymlfile:
			config = yaml.load(ymlfile)
	except:
		sys.exit("Parameters file missing")

	TRAIN_DATA_PATH = config["TEXT_TRAIN_DATA_PATH"]
	VALIDATION_DATA_PATH = config["TEXT_VALIDATION_DATA_PATH"]

	INPUT_LANG_CODE = config['INPUT_LANG']
	OUTPUT_LANG_CODE = config['OUTPUT_LANG']

	TEXT_DATA_TOKENIZED = config['TEXT_TOKENIZED']
	LOAD_PREEMBEDDINGS = config['LOAD_PREEMBEDDINGS']
	COMMON_DICT = config['COMMON_DICT']

	if LOAD_PREEMBEDDINGS:
		W2V_EN_PATH = config["W2V_EN_PATH"]
		W2V_ES_PATH = config["W2V_ES_PATH"]
	else:
		W2V_EN_PATH = None
		W2V_ES_PATH = None

	if COMMON_DICT:
		DICT_EN_PATH = config["DICT_PATH"]
		DICT_ES_PATH = config["DICT_PATH"]
	else:
		DICT_EN_PATH = config["DICT_EN_PATH"]
		DICT_ES_PATH = config["DICT_ES_PATH"]

	if INPUT_LANG_CODE == 'en' and OUTPUT_LANG_CODE == 'es':
		lang_en = input_lang = Lang(INPUT_LANG_CODE, W2V_EN_PATH, DICT_EN_PATH, punctuation_level=config["INPUT_LANG_PUNC_LEVEL"])
		lang_es = output_lang = Lang(OUTPUT_LANG_CODE, W2V_ES_PATH, DICT_ES_PATH, punctuation_level=config["OUTPUT_LANG_PUNC_LEVEL"])
	elif INPUT_LANG_CODE == 'es' and OUTPUT_LANG_CODE == 'en':
		lang_es = input_lang = Lang(INPUT_LANG_CODE, W2V_ES_PATH, DICT_ES_PATH, punctuation_level=config["INPUT_LANG_PUNC_LEVEL"])
		lang_en = output_lang = Lang(OUTPUT_LANG_CODE, W2V_EN_PATH, DICT_EN_PATH, punctuation_level=config["OUTPUT_LANG_PUNC_LEVEL"])

	MAX_SEQ_LENGTH = int(config['MAX_SEQ_LENGTH'])
	TRAINING_BATCH_SIZE = int(config['TEXT_TRAINING_BATCH_SIZE'])
	N_PROSODY_PARAMS = int(config['N_PROSODY_PARAMS'])

	# Configure models
	attn_model = config['ATTN_MODEL']
	hidden_size = int(config['HIDDEN_SIZE'])
	data_size = int(config['TEXT_TRAINING_DATA_SIZE'])
	n_layers = int(config['N_LAYERS'])
	dropout = float(config['DROPOUT'])
	encoder_type = config['ENCODER_TYPE']
	decoder_input_feed = config['DECODER_INPUT_FEED']

	# Configure training/optimization
	clip = float(config['TEXT_CLIP'])
	learning_rate = float(config['TEXT_LEARNING_RATE'])
	decoder_learning_ratio = float(config['TEXT_DECODER_LEARNING_RATIO'])
	n_epochs = int(config['TEXT_N_EPOCHS'])
	patience_epochs = int(config['TEXT_PATIENCE_EPOCHS'])
	print_every_batch = int(config['TEXT_PRINT_EVERY_BATCH'])
	save_every_batch = int(config['TEXT_SAVE_EVERY_BATCH'])
	no_of_batch_in_epoch = data_size/TRAINING_BATCH_SIZE

	# Initialize models
	encoder = GenericEncoder(input_lang.vocabulary_size, hidden_size, input_lang.get_weights_matrix(), n_layers, dropout=dropout)
	decoder = LuongAttnDecoderRNN(attn_model, hidden_size, output_lang.vocabulary_size, input_lang.get_weights_matrix(), n_layers, dropout=dropout, input_feed=decoder_input_feed)

	# Load states from models if given
	if not options.resume_encoder == None and not options.resume_decoder == None:
		load_model(encoder, decoder, options.resume_encoder, options.resume_decoder, gpu_to_cpu=options.gpu2cpu)

	# Initialize optimizers
	encoder_optimizer = optim.Adam(filter(lambda p: p.requires_grad,encoder.parameters()), lr=learning_rate)
	decoder_optimizer = optim.Adam(filter(lambda p: p.requires_grad,decoder.parameters()), lr=learning_rate * decoder_learning_ratio)
	# criterion = nn.CrossEntropyLoss()

	# Move models to GPU
	if USE_CUDA:
		encoder.cuda()
		decoder.cuda()

	if DO_TRAIN:
		# Keep track of time elapsed and running averages
		start = time.time()
		losses = []
		plot_losses = []
		validation_losses = []
		print_loss_total = 0 # Reset every print_every
		plot_loss_total = 0 # Reset every plot_every

		#Initialize the loss log file
		if options.log_file:
			initialize_log_loss(options.log_file, LOSS_LOG_COLUMNS)

		# Begin!
		ecs = []
		dcs = []
		eca = 0
		dca = 0

		epoch = 1
		
		while epoch <= n_epochs:
			training_batch = 0
			validation_batch = 0
			line_in_training_file = 0
			line_in_validation_file = 0
			# Train 
			for input_word_batch, input_lengths, target_word_batch, target_lengths in batch_generator(TRAIN_DATA_PATH, TRAINING_BATCH_SIZE, input_lang, output_lang, tokenize = not TEXT_DATA_TOKENIZED, USE_CUDA=USE_CUDA):
				# Run the train function
				loss_values, ec, dc = train(input_word_batch=input_word_batch,
									 input_prosody_batch=None, 
									 input_lengths=input_lengths, 
									 target_word_batch=target_word_batch,
									 target_prosody_batch=None,
									 target_flag_batch=None, 
									 target_lengths=target_lengths,
									 input_lang=input_lang, 
									 output_lang=output_lang,
									 batch_size=TRAINING_BATCH_SIZE,
									 encoder=encoder, 
									 decoder=decoder,
									 clip=clip,
									 encoder_optimizer=encoder_optimizer, 
									 decoder_optimizer=decoder_optimizer, 
									 run_forward_func=run_forward_text,
									 USE_CUDA=USE_CUDA)

				loss = loss_values[0] #first loss is word loss and only loss in text training
				losses.append(loss)
				# Keep track of loss
				print_loss_total += loss
				plot_loss_total += loss
				eca += ec
				dca += dc
				training_batch += 1

				if training_batch % print_every_batch == 0:
					print_loss_avg = print_loss_total / print_every_batch
					print_loss_total = 0
					print_summary = '%s (Batch:%d/%d %d%%) (Epoch: %d/%d) Loss:%.4f' % (time_since(start, training_batch / no_of_batch_in_epoch), training_batch, no_of_batch_in_epoch, training_batch / no_of_batch_in_epoch * 100, epoch, n_epochs, print_loss_avg)
					print(print_summary)
					if options.log_file:
						log_loss(options.log_file, [print_loss_avg])

				if training_batch % save_every_batch == 0:
					plot_loss_avg = float(plot_loss_total / save_every_batch)
					plot_losses.append(plot_loss_avg)
					plot_loss_total = 0

					#if plot_loss_avg <= min(plot_losses):
					print("Average loss of last %i batch: %f"%(save_every_batch, plot_loss_avg))
					save_model(encoder.state_dict(), decoder.state_dict(), options.model_name, options.model_dir, checkpoint=True)
		
			# Validate model on validation set
			print("VALIDATION ", end='')
			validation_loss_total = 0
			validation_batch = 0
			for input_word_batch, input_lengths, target_word_batch, target_lengths in batch_generator(VALIDATION_DATA_PATH, 1, input_lang, output_lang, tokenize = not TEXT_DATA_TOKENIZED, USE_CUDA=USE_CUDA):
				loss_values = validate( input_word_batch=input_word_batch,
								 input_prosody_batch=None, 
								 input_lengths=input_lengths, 
								 target_word_batch=target_word_batch, 
								 target_prosody_batch=None,
								 target_flag_batch=None,
								 target_lengths=target_lengths,
								 input_lang=input_lang, 
								 output_lang=output_lang,
								 batch_size=1,
								 encoder=encoder, 
								 decoder=decoder,
								 run_forward_func=run_forward_text,
								 USE_CUDA=USE_CUDA)
				
				loss = loss_values[0] #first loss is word loss and only loss in text training

				validation_loss_total += loss
				validation_batch += 1

			validation_loss_avg = float(validation_loss_total / validation_batch)
			validation_summary = 'at Epoch: %d/%d Average Loss:%.4f' % (epoch, n_epochs, validation_loss_avg)
			print(validation_summary)

			# Stopping criteria: stop if validation loss didn't get better in last PATIENCE_EPOCHS 
			if len(validation_losses) == 0 or any([validation_loss_avg < loss for loss in validation_losses[-patience_epochs:]]):
				#Keep on training
				if len(validation_losses) == 0 or validation_loss_avg < min(validation_losses):
					save_model(encoder.state_dict(), decoder.state_dict(), options.model_name, options.model_dir)
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
	parser.add_option("-g", "--gpu2cpu", dest="gpu2cpu", default=False, help="load gpu model on cpu", action="store_true")
	parser.add_option("-p", "--params", dest="params_file", default=None, help="params filename", type="string")
	parser.add_option("-e", "--resumeencoder", dest="resume_encoder", default=None, help="encoder model to resume training from", type="string")
	parser.add_option("-r", "--resumedecoder", dest="resume_decoder", default=None, help="decoder model to resume training from", type="string")
	parser.add_option("-l", "--logfile", dest="log_file", default=None, help="Log file to output losses", type="string")

	(options, args) = parser.parse_args()
	main(options)