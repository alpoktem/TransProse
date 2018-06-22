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

#generates batches from given parallel data file... 
#parallel data file has tab separated english and spanish sentence pairs (tokenized) at each line
def batch_generator(data_path, batch_size, input_lang, output_lang, start_reading_from = 0, n_prosody_params = 3, USE_CUDA=False):
	assert not input_lang == output_lang
	input_word_seqs = []
	input_prosody_seqs = []
	target_seqs = []
	line_no = 0
	
	with open(data_path,'r') as inputfile:
		for line in inputfile:
			line_no += 1
			if line_no < start_reading_from:
				continue
			
			pair = [sentence.strip() for sentence in line.split('\t')]
			tokens_en = pair[0].lower().split()	
			tokens_es = pair[1].lower().split()	

			if input_lang.lang_code == 'en' and output_lang.lang_code == 'es':
				if input_lang.omit_punctuation:
					tokens_en = remove_punc_tokens(tokens_en)
				if output_lang.omit_punctuation:
					tokens_es = remove_punc_tokens(tokens_es)
				input_word_seqs.append(indexes_from_tokens(input_lang, tokens_en))
				input_prosody_seqs.append(prosody_from_tokens(tokens_en, n_prosody_params))
				target_seqs.append(indexes_from_tokens(output_lang, tokens_es))
			elif input_lang.lang_code == 'es' and output_lang.lang_code == 'en':
				if input_lang.omit_punctuation:
					tokens_es = remove_punc_tokens(tokens_es)
				if output_lang.omit_punctuation:
					tokens_en = remove_punc_tokens(tokens_en)
				input_word_seqs.append(indexes_from_tokens(input_lang, tokens_es))
				input_prosody_seqs.append(prosody_from_tokens(tokens_es, n_prosody_params))
				target_seqs.append(indexes_from_tokens(output_lang, tokens_en))

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

				yield input_word_var, input_prosody_var, input_lengths, target_var, target_lengths, line_no
				input_word_seqs = []
				input_prosody_seqs = []
				target_seqs = []

def run_forward(input_word_batches, input_prosody_batches, input_lengths, target_batches, target_lengths, input_lang, output_lang, batch_size, encoder, decoder, USE_CUDA):
	# Run words through encoder
	encoder_outputs, encoder_hidden = encoder(input_word_batches, input_prosody_batches, input_lengths, None)
	
	# Prepare input and output variables
	decoder_input = Variable(torch.LongTensor([output_lang.SPECIAL_TOKEN2INDEX[SWT_TOKEN]] * batch_size))
	decoder_hidden = encoder_hidden[:decoder.n_layers] # Use last (forward) hidden state from encoder

	max_target_length = max(target_lengths)
	all_decoder_outputs = Variable(torch.zeros(max_target_length, batch_size, decoder.output_size))

	# Move new Variables to CUDA
	if USE_CUDA:
		decoder_input = decoder_input.cuda()
		all_decoder_outputs = all_decoder_outputs.cuda()

	# Run through decoder one time step at a time
	for t in range(max_target_length):
		decoder_output, decoder_hidden, decoder_attn = decoder(
			decoder_input, decoder_hidden, encoder_outputs
		)

		all_decoder_outputs[t] = decoder_output
		decoder_input = target_batches[t] # Next input is current target

	# Loss calculation and backpropagation
	loss = masked_cross_entropy(
		all_decoder_outputs.transpose(0, 1).contiguous(), # -> batch x seq
		target_batches.transpose(0, 1).contiguous(), # -> batch x seq
		target_lengths,
		use_cuda = USE_CUDA
	)

	return loss

def train(input_word_batches, input_prosody_batches, input_lengths, target_batches, target_lengths, input_lang, output_lang, batch_size, encoder, decoder, clip, encoder_optimizer, decoder_optimizer, USE_CUDA):
	# Set models to train
	encoder.train()
	decoder.train()

	# Zero gradients of both optimizers
	encoder_optimizer.zero_grad()
	decoder_optimizer.zero_grad()
	loss = 0 # Added onto for each word

	loss = run_forward(input_word_batches, input_prosody_batches, input_lengths, target_batches, target_lengths, input_lang, output_lang, batch_size, encoder, decoder, USE_CUDA)

	# Backpropagate
	loss.backward()
	
	# Clip gradient norms
	ec = torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
	dc = torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)

	# Update parameters with optimizers
	encoder_optimizer.step()
	decoder_optimizer.step()
	
	return loss.data[0], ec, dc

def validate(input_word_batches, input_prosody_batches, input_lengths, target_batches, target_lengths, input_lang, output_lang, batch_size, encoder, decoder, USE_CUDA):
	# Disable training, set to evaluation
	encoder.eval()
	decoder.eval()

	# Zero gradients of both optimizers
	dev_loss = 0

	loss = run_forward(input_word_batches, input_prosody_batches, input_lengths, target_batches, target_lengths, input_lang, output_lang, batch_size, encoder, decoder, USE_CUDA)

	return loss.data[0]

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
	TRAINING_BATCH_SIZE = int(config['TEXT_TRAINING_BATCH_SIZE'])
	N_PROSODY_PARAMS = int(config['N_PROSODY_PARAMS'])

	# Configure models
	attn_model = config['ATTN_MODEL']
	hidden_size = int(config['HIDDEN_SIZE'])
	data_size = int(config['TEXT_TRAINING_DATA_SIZE'])
	n_layers = int(config['N_LAYERS'])
	dropout = float(config['DROPOUT'])
	encoder_type = config['ENCODER_TYPE']

	# Configure training/optimization
	clip = float(config['TEXT_CLIP'])
	learning_rate = float(config['TEXT_LEARNING_RATE'])
	decoder_learning_ratio = float(config['TEXT_DECODER_LEARNING_RATIO'])
	n_epochs = int(config['TEXT_N_EPOCHS'])
	patience_epochs = int(config['TEXT_PATIENCE_EPOCHS'])
	print_every_batch = int(config['TEXT_PRINT_EVERY_BATCH'])
	save_every_batch = int(config['TEXT_SAVE_EVERY_BATCH'])
	no_of_batches_in_epoch = data_size/TRAINING_BATCH_SIZE

	# Initialize models
	if encoder_type == 'sum':
		encoder = EncoderRNN(input_lang.vocabulary_size, N_PROSODY_PARAMS, hidden_size, input_lang.get_weights_matrix(), n_layers, dropout=dropout)
	elif encoder_type == 'parallel':
		encoder = EncoderRNN_parallel(input_lang.vocabulary_size, N_PROSODY_PARAMS, hidden_size, input_lang.get_weights_matrix(), n_layers, dropout=dropout)
	else:
		sys.exit("Unrecognized encoder type. Check params file. Exiting...")
	decoder = LuongAttnDecoderRNN(attn_model, hidden_size, output_lang.vocabulary_size, n_layers, dropout=dropout)

	# Load states from models if given
	if not options.resume_encoder == None and not options.resume_decoder == None:
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
		line_in_training_file = 0
		line_in_validation_file = 0
		# Train 
		for input_word_batches, input_prosody_batches, input_lengths, target_batches, target_lengths, line_in_training_file in batch_generator(TRAIN_DATA_PATH, TRAINING_BATCH_SIZE, input_lang, output_lang, line_in_training_file, USE_CUDA=USE_CUDA):
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
		for input_word_batches, input_prosody_batches, input_lengths, target_batches, target_lengths, line_in_validation_file in batch_generator(VALIDATION_DATA_PATH, 1, input_lang, output_lang, line_in_validation_file, USE_CUDA=USE_CUDA):
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