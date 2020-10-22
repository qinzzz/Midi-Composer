"""
composer.py

@time: 10/20/20
@author: Qinxin Wang

@desc:
"""
import torch
import torch.nn as nn

from models.gru import GRU
from song_components.song import Song


class RNNComposer:
	def __init__(self):
		self.note_nb = 88
		self.look_before_limit = None
		self.model = GRU(self.note_nb)
		self.loss_function = nn.BCELoss()

	def _get_note_from_logits(self, logits):
		"""

		:param logits: [batch, seq_len, note_nb]
		:return: next_note: [batch, note_nb] with 0 or 1
		"""
		next_logits = logits[:, -1, :]  # [batch, note_nb] between (0, 1)
		next_note = (next_logits > 0.5).float()

		return next_note

	def predict_next_single(self, prev_notes):
		"""

		:param prev_notes: [batch, seq_len, note_nb]
		:return: next_note: [batch, note_nb] with 0 or 1
		"""
		logits, h_n = self.model.forward(prev_notes)
		return self._get_note_from_logits(logits)

	def predict_next_sequence(self, prev_notes, length):
		"""

		:param prev_notes: [batch, seq_len, note_nb]
		:param length: int
		:return: next_notes: [batch, length, note_nb] with 0 or 1
		"""
		if not self.look_before_limit:
			next_notes = torch.empty(prev_notes.size(0), length, prev_notes.size(2))

			logits, h_n = self.model.forward(prev_notes)
			next_note = self._get_note_from_logits(logits)
			next_notes[:, 0, :] = next_note

			for i in range(1, length):
				logits, h_n = self.model.forward(next_note, h_n)
				next_note = self._get_note_from_logits(logits)

				next_notes[:, i, :] = next_note

			return next_notes

	def train(self, data_loader):
		raise NotImplementedError('training not implemented.')

	def test(self):
		raise NotImplementedError('training not implemented.')

	def generate(self, song: Song, bars = 8) -> Song:
		raise NotImplementedError('training not implemented.')
