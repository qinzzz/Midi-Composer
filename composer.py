"""
composer.py

@time: 10/20/20
@author: Qinxin Wang

@desc:
"""
import torch
import torch.nn as nn

from model.gru import midiRNN, lyricsRNN


class LyricComposer:
	def __init__(self, input_size = 100, layers = 2, hidden_size = 512):
		self.input_size = input_size
		self.layers = layers
		self.hidden_size = hidden_size
		self.look_before_limit = None
		self.model = lyricsRNN(self.input_size, hidden_size = hidden_size, num_classes = self.input_size, layers = layers)

		self.model = self.model.cuda()


class RNNSongComposer:
	def __init__(self, input_size = 88, layers = 2, hidden_size = 512):
		self.input_size = input_size
		self.layers = layers
		self.hidden_size = hidden_size
		self.look_before_limit = None
		self.model = midiRNN(self.input_size, layers = layers, hidden_size = hidden_size, num_classes = self.input_size)

		self.model = self.model.cuda()

	def _get_note_from_logits(self, logits):
		"""

		:param logits: [batch, seq_len, note_nb]
		:return: next_note: [batch, note_nb] with 0 or 1
		"""
		next_logits = logits[:, -1, :]  # [batch, note_nb] between (0, 1)
		next_note = (next_logits > 0.01).float()

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
				next_note.unsqueeze_(1)
				logits, h_n = self.model.forward(next_note, h_n)
				next_note = self._get_note_from_logits(logits)

				next_notes[:, i, :] = next_note

			return next_notes

	def sample_from_piano_rnn(self, sample_length = 4, temperature = 1, starting_sequence = None):

		if starting_sequence is None:
			current_sequence_input = torch.zeros(1, 1, 88)
			current_sequence_input[0, 0, 40] = 1
			current_sequence_input[0, 0, 50] = 1
			current_sequence_input[0, 0, 56] = 1
		else:
			current_sequence_input = starting_sequence

		final_output_sequence = [current_sequence_input.data.squeeze(1)]
		hidden = None

		for i in range(sample_length):
			output, hidden = self.model(current_sequence_input, [1], hidden)
			probabilities = nn.functional.softmax(output.div(temperature), dim = 1)
			current_sequence_input = torch.multinomial(probabilities.data, 1).squeeze().unsqueeze(0).unsqueeze(
				1).float()

			final_output_sequence.append(current_sequence_input.data.squeeze(1))

		sampled_sequence = torch.cat(final_output_sequence, dim = 0).cpu().numpy()

		return sampled_sequence
