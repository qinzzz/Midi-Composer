"""
notedataset.py

@time: 10/28/20
@author: Qinxin Wang

@desc:
"""
import os

import torch
import torch.utils.data as data

from midiparser import midi_filename_to_piano_roll, pad_piano_roll


class NotesGenerationDataset(data.Dataset):

	def __init__(self, midi_folder_path, longest_sequence_length = 1491):
		self.midi_folder_path = midi_folder_path

		midi_filenames = os.listdir(midi_folder_path)

		self.longest_sequence_length = longest_sequence_length

		midi_full_filenames = map(lambda filename: os.path.join(midi_folder_path, filename),
								  midi_filenames)

		self.midi_full_filenames = list(midi_full_filenames)

		if longest_sequence_length is None:
			self.update_the_max_length()

	def update_the_max_length(self):
		"""Recomputes the longest sequence constant of the dataset.

		Reads all the midi files from the midi folder and finds the max
		length.
		"""

		sequences_lengths = list(map(lambda filename: midi_filename_to_piano_roll(filename).shape[1],
									 self.midi_full_filenames))

		max_length = max(sequences_lengths)

		self.longest_sequence_length = max_length

	def __len__(self):
		return len(self.midi_full_filenames)

	def __getitem__(self, index):
		midi_full_filename = self.midi_full_filenames[index]

		piano_roll = midi_filename_to_piano_roll(midi_full_filename)

		# -1 because we will shift it
		sequence_length = piano_roll.shape[1] - 1

		# Shifted by one time step
		input_sequence = piano_roll[:, :-1]
		ground_truth_sequence = piano_roll[:, 1:]

		# pad sequence so that all of them have the same lenght
		# Otherwise the batching won't work
		input_sequence_padded = pad_piano_roll(input_sequence, max_length = self.longest_sequence_length)

		ground_truth_sequence_padded = pad_piano_roll(ground_truth_sequence,
													  max_length = self.longest_sequence_length,
													  pad_value = -100)

		input_sequence_padded = input_sequence_padded.transpose()
		ground_truth_sequence_padded = ground_truth_sequence_padded.transpose()

		return (torch.FloatTensor(input_sequence_padded),
				torch.LongTensor(ground_truth_sequence_padded),
				torch.LongTensor([sequence_length]))

