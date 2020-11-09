"""
rnn.py

@time: 10/20/20
@author: Qinxin Wang

@desc:
"""
import torch
import torch.nn as nn


class midiRNN(nn.Module):
	def __init__(self, input_size, hidden_size, num_classes, layers):
		super(midiRNN, self).__init__()

		self.notes_encoder = nn.Linear(input_size, hidden_size)
		self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers = layers, batch_first = True)
		self.logits_fc = nn.Linear(hidden_size, num_classes)

		nn.init.xavier_uniform_(self.notes_encoder.weight)
		nn.init.xavier_uniform_(self.logits_fc.weight)

	def forward(self, x_seq, x_len, h_0 = None):
		"""
		input
			x: [batch, seq_len, note_nb]
		return
			logits: [batch, seq_len, note_nb]
			h_n: [batch, num_layers * num_directions, hidden_size]
		"""
		x_embed = self.notes_encoder(x_seq)  # [batch, seq_len, input_size]

		packed_x = torch.nn.utils.rnn.pack_padded_sequence(x_embed, x_len)
		packed_output, h_n = self.lstm(packed_x, h_0)
		output, output_len = torch.nn.utils.rnn.pad_packed_sequence(packed_output)

		logits = self.logits_fc(output)
		neg_logits = (1 - logits)

		binary_logits = torch.stack((logits, neg_logits), dim = 3).contiguous()

		logits_flatten = binary_logits.view(-1, 2)

		return logits_flatten, h_n


class lyricsRNN(nn.Module):
	def __init__(self, input_size, hidden_size, num_classes, layers = 2):
		super(lyricsRNN, self).__init__()

		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_classes = num_classes

		# input_size -- size of the dictionary + 1 (accounts for padding constant)
		self.encoder = nn.Embedding(input_size, hidden_size)
		self.gru = nn.LSTM(hidden_size, hidden_size, layers)
		self.logits_fc = nn.Linear(hidden_size, num_classes)

		nn.init.xavier_uniform_(self.encoder.weight)
		nn.init.xavier_uniform_(self.logits_fc.weight)

	def forward(self, input_sequences, input_sequences_lengths, hidden = None):

		input_emb = self.encoder(input_sequences) 	# [len, batch] -> [len, batch, dim]

		# Here we run rnns only on non-padded regions of the batch
		packed = torch.nn.utils.rnn.pack_padded_sequence(input_emb, input_sequences_lengths)
		outputs, hidden = self.gru(packed, hidden)
		outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)

		logits = self.logits_fc(outputs)
		logits = logits.transpose(0, 1).contiguous()
		logits_flatten = logits.view(-1, self.num_classes)

		return logits_flatten, hidden
