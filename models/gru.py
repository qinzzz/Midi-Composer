"""
gru.py

@time: 10/20/20
@author: Qinxin Wang

@desc:
"""
import torch
import torch.nn as nn


class midiGRU(nn.Module):
	def __init__(self, note_nb, hidden_size, layers):
		super(midiGRU, self).__init__()

		self.embeddings = nn.Embedding(note_nb, hidden_size)
		self.linear1 = nn.Linear(note_nb, hidden_size)
		self.gru = nn.GRU(hidden_size, hidden_size, num_layers = layers, batch_first = True)
		self.linear2 = nn.Linear(hidden_size, note_nb)
		self.sigmoid = nn.Sigmoid()
		nn.init.xavier_uniform_(self.linear1.weight)
		nn.init.xavier_uniform_(self.linear2.weight)

	def forward(self, x_seq, x_len, h_0 = None):
		"""
		input
			x: [batch, seq_len, note_nb]
		return
			logits: [batch, seq_len, note_nb]
			h_n: [batch, num_layers * num_directions, hidden_size]
		"""
		x_embed = self.linear1(x_seq)  # [batch, seq_len, input_size]

		packed_x = torch.nn.utils.rnn.pack_padded_sequence(x_embed, x_len)
		packed_output, h_n = self.gru(packed_x, h_0)
		output, output_len = torch.nn.utils.rnn.pad_packed_sequence(packed_output)

		logits = self.linear2(output)
		neg_logits = (1 - logits)

		binary_logits = torch.stack((logits, neg_logits), dim = 3).contiguous()

		logits_flatten = binary_logits.view(-1, 2)

		return logits_flatten, h_n


class lyricsGRU(nn.Module):
	def __init__(self, input_size, hidden_size, num_classes, n_layers = 2):
		super(lyricsGRU, self).__init__()

		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_classes = num_classes
		self.n_layers = n_layers

		# Converts labels into one-hot encoding and runs a linear
		# layer on each of the converted one-hot encoded elements

		# input_size -- size of the dictionary + 1 (accounts for padding constant)
		self.encoder = nn.Embedding(input_size, hidden_size)

		self.gru = nn.GRU(hidden_size, hidden_size, n_layers)

		self.logits_fc = nn.Linear(hidden_size, num_classes)

	def forward(self, input_sequences, input_sequences_lengths, hidden = None):
		batch_size = input_sequences.shape[1]

		embedded = self.encoder(input_sequences)

		# Here we run rnns only on non-padded regions of the batch
		packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_sequences_lengths)
		outputs, hidden = self.gru(packed, hidden)
		outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)

		logits = self.logits_fc(outputs)

		logits = logits.transpose(0, 1).contiguous()

		logits_flatten = logits.view(-1, self.num_classes)

		return logits_flatten, hidden
