"""
gru.py

@time: 10/20/20
@author: Qinxin Wang

@desc:
"""
import torch
import torch.nn as nn


class GRU(nn.Module):
	def __init__(self, note_nb, input_size = 128, hidden = 128):
		super(GRU, self).__init__()

		self.embeddings = nn.Embedding(note_nb, input_size)
		self.linear1 = nn.Linear(note_nb, input_size)
		self.gru = nn.GRU(input_size, hidden, num_layers = 1, batch_first = True)
		self.linear2 = nn.Linear(hidden, note_nb)
		self.sigmoid = nn.Sigmoid()
		nn.init.xavier_uniform_(self.linear1.weight)
		nn.init.xavier_uniform_(self.linear2.weight)

	def forward(self, x, h_0 = None):
		"""
		input
			x: [batch, seq_len, note_nb]
		return
			logits: [batch, seq_len, note_nb]
			h_n: [batch, num_layers * num_directions, hidden_size]
		"""
		# x_embed = self.embeddings(x)  # [batch, seq_len, input_size]
		x_embed = self.linear1(x) # [batch, seq_len, input_size]
		if h_0 is not None:
			output, h_n = self.gru(x_embed, h_0)
		else:
			output, h_n = self.gru(x_embed)

		logits = self.sigmoid(self.linear2(output))

		return logits, h_n
