"""
gru.py

@time: 10/20/20
@author: Qinxin Wang

@desc:
"""
import torch
import torch.nn as nn


class GRU(nn.Module):
	def __init__(self, note_nb = 88, hidden_size = 1024):
		super(GRU, self).__init__()

		self.embeddings = nn.Embedding(note_nb, hidden_size)
		self.linear1 = nn.Linear(note_nb, hidden_size)
		self.gru = nn.GRU(hidden_size, hidden_size, num_layers = 1, batch_first = True)
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
