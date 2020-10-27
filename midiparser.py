"""
midiparser.py

@time: 10/20/20
@author: Qinxin Wang

@desc:
	parse midi file to our defined song structure.
	based on Daniil Pakhomov's repo.
"""
import os

import numpy as np
import torch
import torch.utils.data as data

from midi.utils import midiread


def midi_filename_to_piano_roll(midi_filename):
	midi_data = midiread(midi_filename, dt = 0.3)

	piano_roll = midi_data.piano_roll.transpose()

	# Binarize the pressed notes
	piano_roll[piano_roll > 0] = 1

	return piano_roll


def pad_piano_roll(piano_roll, max_length = 132333, pad_value = 0):
	# 128 pitches

	original_piano_roll_length = piano_roll.shape[1]

	padded_piano_roll = np.zeros((88, max_length))
	padded_piano_roll[:] = pad_value

	padded_piano_roll[:, :original_piano_roll_length] = piano_roll

	return padded_piano_roll


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


def post_process_sequence_batch(batch_tuple):
	input_sequences, output_sequences, lengths = batch_tuple

	splitted_input_sequence_batch = input_sequences.split(split_size = 1)
	splitted_output_sequence_batch = output_sequences.split(split_size = 1)
	splitted_lengths_batch = lengths.split(split_size = 1)

	training_data_tuples = zip(splitted_input_sequence_batch,
							   splitted_output_sequence_batch,
							   splitted_lengths_batch)

	training_data_tuples_sorted = sorted(training_data_tuples,
										 key = lambda p: int(p[2]),
										 reverse = True)

	splitted_input_sequence_batch, splitted_output_sequence_batch, splitted_lengths_batch = zip(
		*training_data_tuples_sorted)

	input_sequence_batch_sorted = torch.cat(splitted_input_sequence_batch)
	output_sequence_batch_sorted = torch.cat(splitted_output_sequence_batch)
	lengths_batch_sorted = torch.cat(splitted_lengths_batch)

	# Here we trim overall data matrix using the size of the longest sequence
	input_sequence_batch_sorted = input_sequence_batch_sorted[:, :lengths_batch_sorted[0, 0], :]
	output_sequence_batch_sorted = output_sequence_batch_sorted[:, :lengths_batch_sorted[0, 0], :]

	input_sequence_batch_transposed = input_sequence_batch_sorted.transpose(0, 1)

	# pytorch's api for rnns wants lenghts to be list of ints
	lengths_batch_sorted_list = list(lengths_batch_sorted)
	lengths_batch_sorted_list = list(map(lambda x: int(x), lengths_batch_sorted_list))

	return input_sequence_batch_transposed, output_sequence_batch_sorted, lengths_batch_sorted_list


if __name__ == "__main__":
	# Test
	piano_roll = midi_filename_to_piano_roll("Piano-midi/valid/bach_846.mid")
	print(piano_roll.nonzero())
