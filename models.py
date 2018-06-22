import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence#, masked_cross_entropy
from masked_cross_entropy import *
import numpy as np


class EncoderRNN(nn.Module):
	def __init__(self, input_size, n_prosody_params, hidden_size, weights_matrix, n_layers=1, dropout=0.1):
		super(EncoderRNN, self).__init__()
		
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.n_layers = n_layers
		self.n_prosody_params = n_prosody_params
		self.dropout = dropout

		self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weights_matrix))
		self.embedding_linear = nn.Linear(100, hidden_size) #Linear layer to increase embedding vector size
		self.prosody = nn.Linear(n_prosody_params, hidden_size) #Linear layer
		self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True)
		
	def forward(self, input_word_seqs, input_pros_seqs, input_lengths, hidden=None):
		# Note: we run this all at once (over multiple batches of multiple sequences)
		embedded = self.embedding_linear(self.embedding(input_word_seqs))
		prosody_vec = self.prosody(input_pros_seqs)
		input_seq = embedded + prosody_vec   #sum word embedding and prosody vector
		packed = torch.nn.utils.rnn.pack_padded_sequence(input_seq, input_lengths)
		outputs, hidden = self.gru(packed, hidden)
		outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs) # unpack (back to padded)
		outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:] # Sum bidirectional outputs
		return outputs, hidden

class EncoderRNN_parallel(nn.Module):
	def __init__(self, input_size, n_prosody_params, hidden_size, weights_matrix, n_layers=1, dropout=0.1):
		super(EncoderRNN_parallel, self).__init__()
		
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.n_layers = n_layers
		self.n_prosody_params = n_prosody_params
		self.dropout = dropout

		self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weights_matrix))
		self.embedding_linear = nn.Linear(100, hidden_size) #Linear layer to increase embedding vector size
		self.gru_word = nn.GRU(hidden_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True)
		self.gru_prosody = nn.GRU(n_prosody_params, hidden_size, n_layers, dropout=self.dropout, bidirectional=True)
		
	def forward(self, input_word_seqs, input_pros_seqs, input_lengths, hidden_word=None, hidden_prosody=None):
		# Note: we run this all at once (over multiple batches of multiple sequences)
		word_embedded = self.embedding_linear(self.embedding(input_word_seqs))
		packed_word = torch.nn.utils.rnn.pack_padded_sequence(word_embedded, input_lengths)
		packed_prosody = torch.nn.utils.rnn.pack_padded_sequence(input_pros_seqs, input_lengths)
		outputs_word, hidden_word = self.gru_word(packed_word, hidden_word)
		outputs_prosody, hidden_prosody = self.gru_prosody(packed_prosody, hidden_prosody)
		outputs_word, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs_word) # unpack (back to padded)
		outputs_prosody, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs_prosody) # unpack (back to padded)

		outputs = outputs_word[:, :, :self.hidden_size] + outputs_word[:, : ,self.hidden_size:] + outputs_prosody[:, :, :self.hidden_size] + outputs_prosody[:, : ,self.hidden_size:]# Sum bidirectional outputs
		hidden = hidden_word + hidden_prosody
		return outputs, hidden

class Attn(nn.Module):
	def __init__(self, method, hidden_size):
		super(Attn, self).__init__()
		
		self.method = method
		self.hidden_size = hidden_size
		
		if self.method == 'general':
			self.attn = nn.Linear(self.hidden_size, hidden_size)

		elif self.method == 'concat':
			self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
			self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))
			
	def forward(self, hidden, encoder_outputs):
		energy = self.attn(encoder_outputs).transpose(0, 1)  # S X B X N -> B X S X N
		e = energy.bmm(hidden.transpose(0, 1).transpose(1, 2)) # B X S X 1
		attn_energies = e.squeeze(2)  # B X S
		return F.softmax(attn_energies, 1).unsqueeze(1)  # B X 1 X S

	def score(self, hidden, encoder_output):
		if self.method == 'dot':
			energy = torch.dot(hidden.view(-1), encoder_output.view(-1))
		elif self.method == 'general':
			energy = self.attn(encoder_output)
			energy = torch.dot(hidden.view(-1), energy.view(-1))
		elif self.method == 'concat':
			energy = self.attn(torch.cat((hidden, encoder_output), 1))
			energy = torch.dot(self.v.view(-1), energy.view(-1))
		return energy

class LuongAttnDecoderRNN(nn.Module):
	def __init__(self, attn_model, hidden_size, output_size, n_layers=1, dropout=0.1):
		super(LuongAttnDecoderRNN, self).__init__()

		# Keep for reference
		self.attn_model = attn_model
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.n_layers = n_layers
		self.dropout = dropout

		# Define layers
		self.embedding = nn.Embedding(output_size, hidden_size)
		self.embedding_dropout = nn.Dropout(dropout)
		self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout)
		self.concat = nn.Linear(hidden_size * 2, hidden_size)
		self.out = nn.Linear(hidden_size, output_size)
		
		# Choose attention model
		if attn_model != 'none':
			self.attn = Attn(attn_model, hidden_size)

	def forward(self, input_seq, last_hidden, encoder_outputs):
		# Note: we run this one step at a time

		# Get the embedding of the current input word (last output word)
		batch_size = input_seq.size(0)
		embedded = self.embedding(input_seq)
		embedded = self.embedding_dropout(embedded)
		embedded = embedded.view(1, batch_size, self.hidden_size) # S=1 x B x N

		# Get current hidden state from input word and last hidden state
		rnn_output, hidden = self.gru(embedded, last_hidden)

		# Calculate attention from current RNN state and all encoder outputs;
		# apply to encoder outputs to get weighted average
		attn_weights = self.attn(rnn_output, encoder_outputs)
		context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) # B x S=1 x N

		# Attentional vector using the RNN hidden state and context vector
		# concatenated together (Luong eq. 5)
		rnn_output = rnn_output.squeeze(0) # S=1 x B x N -> B x N
		context = context.squeeze(1)       # B x S=1 x N -> B x N
		concat_input = torch.cat((rnn_output, context), 1)
		concat_output = F.tanh(self.concat(concat_input))

		# Finally predict next token (Luong eq. 6, without softmax)
		output = self.out(concat_output)

		# Return final output, hidden state, and attention weights (for visualization)
		return output, hidden, attn_weights


