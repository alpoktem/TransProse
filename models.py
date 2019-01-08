import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence#, masked_cross_entropy
from masked_cross_entropy import *
import numpy as np

GENSIM_VEC_SIZE = 100

class GenericEncoder(nn.Module):
	def __init__(self, input_size, hidden_size, weights_matrix=None, n_layers=1, dropout=0.1, rnn_type='GRU'):
		super(GenericEncoder, self).__init__()

		self.input_size = input_size
		self.hidden_size = hidden_size
		self.n_layers = n_layers
		self.dropout = dropout

		if not weights_matrix is None:
			self.pretrained_embeddings = True
			self.pre_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weights_matrix))
			self.embedding = nn.Linear(GENSIM_VEC_SIZE, hidden_size) #Linear layer to increase embedding vector size
		else:
			self.pretrained_embeddings = False
			self.embedding = nn.Embedding(input_size, hidden_size)
		
		if rnn_type == 'LSTM':
			self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True)
		else:
			self.rnn = nn.GRU(hidden_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True)

	def forward(self, input_word_seqs, input_lengths, hidden=None):
		# Note: we run this all at once (over multiple batches of multiple sequences)
		if self.pretrained_embeddings:
			embedded = self.embedding(self.pre_embedding(input_word_seqs))
		else:
			embedded = self.embedding(input_word_seqs)

		input_seq = embedded   
		packed = torch.nn.utils.rnn.pack_padded_sequence(input_seq, input_lengths)
		outputs, hidden = self.rnn(packed, hidden)
		outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs) # unpack (back to padded)
		outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:] # Sum bidirectional outputs
		return outputs, hidden

#Discrete pause input
class EncoderRNN_discrete_pause(nn.Module):
	def __init__(self, input_size, hidden_size, weights_matrix, n_layers=1, dropout=0.1):
		super(EncoderRNN_sum_ver, self).__init__()
		
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.n_layers = n_layers
		self.dropout = dropout

		self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weights_matrix))
		self.embedding_linear = nn.Linear(GENSIM_VEC_SIZE, hidden_size) #Linear layer to increase embedding vector size
		self.pause_embedding = nn.Embedding(100, hidden_size)


		self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True)

		
	def forward(self, input_word_seqs, input_pause_seqs, input_lengths, hidden=None):
		# Note: we run this all at once (over multiple batches of multiple sequences)
		embedded = self.embedding_linear(self.embedding(input_word_seqs))
		embedded_pause = self.pause_embedding(input_pause_seqs)
		input_seq = embedded + embedded_pause   #sum word embedding and prosody vector
		packed = torch.nn.utils.rnn.pack_padded_sequence(input_seq, input_lengths)
		outputs, hidden = self.gru(packed, hidden)
		outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs) # unpack (back to padded)
		outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:] # Sum bidirectional outputs
		return outputs, hidden

#Gradual continous prosody input
class EncoderRNN_sum_ver(nn.Module):
	def __init__(self, input_size, n_prosody_params, hidden_size, weights_matrix, n_layers=1, dropout=0.1):
		super(EncoderRNN_sum_ver, self).__init__()
		
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.n_layers = n_layers
		self.n_prosody_params = n_prosody_params
		self.dropout = dropout

		self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weights_matrix))
		self.embedding_linear = nn.Linear(GENSIM_VEC_SIZE, hidden_size) #Linear layer to increase embedding vector size
		self.prosody1 = nn.Linear(n_prosody_params, int(hidden_size/4)) #Linear layer
		self.prosody2 = nn.Linear(int(hidden_size/4), int(hidden_size/2)) #Linear layer
		self.prosody3 = nn.Linear(int(hidden_size/2), hidden_size) #Linear layer
		self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True)

		#initialize prosody layer weights to zero (EXPERIMENTAL)
		self.prosody1.weight.data = torch.zeros([int(hidden_size/4), n_prosody_params])
		self.prosody2.weight.data = torch.zeros([int(hidden_size/2), int(hidden_size/4)])
		self.prosody3.weight.data = torch.zeros([int(hidden_size), int(hidden_size/2)])
		
	def forward(self, input_word_seqs, input_pros_seqs, input_lengths, hidden=None):
		# Note: we run this all at once (over multiple batches of multiple sequences)
		embedded = self.embedding_linear(self.embedding(input_word_seqs))
		prosody_vec = self.prosody3(F.tanh(self.prosody2(F.tanh(self.prosody1(input_pros_seqs)))))
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
		self.embedding_linear = nn.Linear(GENSIM_VEC_SIZE, hidden_size) #Linear layer to increase embedding vector size
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

class Attn_m(nn.Module):
	def __init__(self, method, hidden_size, USE_CUDA=False):
		super(Attn_m, self).__init__()
		
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

#Spro's original attention
class Attn(nn.Module):
    def __init__(self, method, hidden_size, USE_CUDA=False):
        super(Attn, self).__init__()
        
        self.method = method
        self.hidden_size = hidden_size
        self.USE_CUDA = USE_CUDA
        
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))
            
        self.linear_cover = nn.Linear(1, hidden_size, bias=False)

    def forward(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)

        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(this_batch_size, max_len)) # B x S

        if self.USE_CUDA:
            attn_energies = attn_energies.cuda()

        # For each batch of encoder outputs
        for b in range(this_batch_size):
            # Calculate energy for each encoder output
            for i in range(max_len):
                attn_energies[b, i] = self.score(hidden[:, b], encoder_outputs[i, b].unsqueeze(0))
                
        # Normalize energies to weights in range 0 to 1, resize to B x 1 x S
        normalized = F.softmax(attn_energies).unsqueeze(1)
        return normalized
    
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
	def __init__(self, attn_model, hidden_size, output_size, weights_matrix=None, n_layers=1, dropout=0.1, input_feed=False, rnn_type='GRU', USE_CUDA=False):
		super(LuongAttnDecoderRNN, self).__init__()

		# Keep for reference
		self.attn_model = attn_model
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.n_layers = n_layers
		self.dropout = dropout
		self.input_feed = input_feed

		# Define layers
		if not weights_matrix is None:
			self.pretrained_embeddings = True
			self.pre_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weights_matrix))     
			self.embedding = nn.Linear(GENSIM_VEC_SIZE, hidden_size) #Linear layer to increase embedding vector size     
		else:
			self.pretrained_embeddings = False
			self.embedding = nn.Embedding(output_size, hidden_size)
		
		self.embedding_dropout = nn.Dropout(dropout)
		if self.input_feed:
			rnn_input_size = hidden_size * 2 
		else:
			rnn_input_size = hidden_size

		if rnn_type == 'LSTM':
			self.rnn = nn.LSTM(rnn_input_size, hidden_size, n_layers, dropout=dropout)
		else:
			self.rnn = nn.GRU(rnn_input_size, hidden_size, n_layers, dropout=dropout)

		self.concat = nn.Linear(hidden_size * 2, hidden_size)
		self.out = nn.Linear(hidden_size, output_size)
		
		# Choose attention model
		if attn_model != 'none':
			self.attn = Attn_m(attn_model, hidden_size, USE_CUDA)

	def forward(self, input_at_t, last_context, last_hidden, encoder_outputs):
		# Note: we run this one step at a time

		# Get the embedding of the current input word (last output word)
		batch_size = input_at_t.size(0)
		if self.pretrained_embeddings:
			embedded = self.embedding(self.pre_embedding(input_at_t))	 #B x N
		else:
			embedded = self.embedding(input_at_t)

		embedded = self.embedding_dropout(embedded)
		embedded = embedded.view(1, batch_size, self.hidden_size) # S=1 x B x N

		if self.input_feed:
			# Combine embedded input word and last context, run through RNN
			rnn_input = torch.cat((embedded, last_context.unsqueeze(0)), 2)
			rnn_output, hidden = self.rnn(rnn_input, last_hidden) # rnn_output S=1 x B x N, hidden L x B X N
		else:
			# Get current hidden state from input word and last hidden state
			rnn_output, hidden = self.rnn(embedded, last_hidden)

		# Calculate attention from current RNN state and all encoder outputs;
		# apply to encoder outputs to get weighted average
		attn_weights = self.attn(rnn_output, encoder_outputs) # B x 1 x S
		context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) # B x S=1 x N

		# Attentional vector using the RNN hidden state and context vector
		# concatenated together (Luong eq. 5)
		rnn_output = rnn_output.squeeze(0) # S=1 x B x N -> B x N
		context = context.squeeze(1)       # B x S=1 x N -> B x N
		concat_input = torch.cat((rnn_output, context), 1) # B x 2N
		concat_output = F.tanh(self.concat(concat_input))

		# Finally predict next token (Luong eq. 6, without softmax)
		output = self.out(concat_output)

		# Return final output, hidden state, and attention weights (for visualization)
		return output, context, hidden, attn_weights


class ProsodicDecoderRNN(nn.Module):
	def __init__(self, attn_model, hidden_size, weights_matrix, output_size, n_layers=1, dropout=0.1, input_feed=False, USE_CUDA=False):
		super(ProsodicDecoderRNN, self).__init__()

		# Keep for reference
		self.attn_model = attn_model
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.n_layers = n_layers
		self.dropout = dropout
		self.input_feed = input_feed

		# Define layers
		self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weights_matrix))     
		self.embedding_linear = nn.Linear(GENSIM_VEC_SIZE, hidden_size) #Linear layer to increase embedding vector size     
		self.embedding_dropout = nn.Dropout(dropout)
		if self.input_feed:
			self.gru = nn.GRU(hidden_size * 2 , hidden_size, n_layers, dropout=dropout)
		else:
			self.gru = nn.GRU(hidden_size , hidden_size, n_layers, dropout=dropout)
		self.concat = nn.Linear(hidden_size * 2, hidden_size)
		self.out = nn.Linear(hidden_size, output_size)
		self.pause_flag_out = nn.Linear(hidden_size * 2, 2)   #Linear
		#self.pause_value_out = nn.Linear(hidden_size * 2, 1)   #Linear
		
		# Choose attention model
		if attn_model != 'none':
			self.attn = Attn_m(attn_model, hidden_size, USE_CUDA)

	def forward(self, input_at_t, last_context, last_hidden, encoder_outputs):
		# Note: we run this one step at a time
		
		# Get the embedding of the current input word (last output word)
		batch_size = input_at_t.size(0)
		embedded = self.embedding_linear(self.embedding(input_at_t))	
		embedded = self.embedding_dropout(embedded)
		embedded = embedded.view(1, batch_size, self.hidden_size) # S=1 x B x N

		if self.input_feed:
			# Combine embedded input word and last context, run through RNN
			rnn_input = torch.cat((embedded, last_context.unsqueeze(0)), 2)
			rnn_output, hidden = self.gru(rnn_input, last_hidden) # rnn_output S=1 x B x N, hidden L x B X N
		else:
			# Get current hidden state from input word and last hidden state
			rnn_output, hidden = self.gru(embedded, last_hidden)


		# Calculate attention from current RNN state and all encoder outputs;
		# apply to encoder outputs to get weighted average
		attn_weights = self.attn(rnn_output, encoder_outputs) # B x 1 x S
		context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) # B x S=1 x N

		# Attentional vector using the RNN hidden state and context vector
		# concatenated together (Luong eq. 5)
		rnn_output = rnn_output.squeeze(0) # S=1 x B x N -> B x N
		context = context.squeeze(1)       # B x S=1 x N -> B x N
		concat_input = torch.cat((rnn_output, context), 1)   # B x N*2
		concat_output = F.tanh(self.concat(concat_input))  # B x N

		# Predict next word token (Luong eq. 6, without softmax)
		token_output = self.out(concat_output)  # B x V

		# Prosodic value inputs
		prosody_input = torch.cat((hidden[-1], concat_output), 1) # B x N*2
				
		pause_flag_output = self.pause_flag_out(prosody_input)
		#pause_value_output = F.sigmoid(self.pause_value_out(prosody_input))
		
		# Return final output, hidden state, and attention weights (for visualization)
		#return token_output, pause_flag_output, pause_value_output, context, hidden, attn_weights
		return token_output, pause_flag_output, context, hidden, attn_weights