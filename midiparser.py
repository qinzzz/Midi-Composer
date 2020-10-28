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


def midi_filename_to_piano_roll(midi_filename, dt= 0.3):
	midi_data = midiread(midi_filename, dt = dt)

	piano_roll = midi_data.piano_roll.transpose()

	# Binarize the pressed notes
	piano_roll[piano_roll > 0] = 1

	return piano_roll


def pad_piano_roll(piano_roll, max_length = 132333, pad_value = 0):
	# 88 pitches

	original_piano_roll_length = piano_roll.shape[1]

	padded_piano_roll = np.zeros((88, max_length))
	padded_piano_roll[:] = pad_value

	padded_piano_roll[:, :original_piano_roll_length] = piano_roll

	return padded_piano_roll


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
