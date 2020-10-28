"""
lyricsdataset.py

@time: 10/28/20
@author: Qinxin Wang

@desc:
"""
import os
import string

import torch
import torch.utils.data as data

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

all_characters = string.printable
number_of_characters = len(all_characters)


def character_to_label(character):
	character_label = all_characters.find(character)
	return character_label


def string_to_labels(character_string):
	return list(map(lambda character: character_to_label(character), character_string))


def pad_sequence(seq, max_length, pad_label = 100):
	seq += [pad_label for i in range(max_length - len(seq))]
	return seq


class LyricsGenerationDataset(data.Dataset):

	def __init__(self, dir_root, minimum_song_count = None, artists = None):

		self.dir_root = dir_root
		self.lyrics_files = os.listdir(dir_root)

		self.lyrics_data = []
		self.read_lyrics()

		# Get the length of the biggest lyric text
		# We will need that for padding
		self.max_text_len = max([len(t) for t in self.lyrics_data])

		whole_dataset_len = len(self.lyrics_data)

		self.indexes = range(whole_dataset_len)

	def read_lyrics(self):
		for file in self.lyrics_files:
			self.lyrics_data += self.read_text(file)

	def read_text(self, filename):
		filepath = os.path.join(self.dir_root, filename)
		with open(filepath, "r") as f:
			lines = f.readlines()
		return lines

	def __len__(self):
		return len(self.indexes)

	def __getitem__(self, index):
		sequence_raw_string = self.lyrics_data[index]
		# print(sequence_raw_string)

		sequence_string_labels = string_to_labels(sequence_raw_string)

		sequence_length = len(sequence_string_labels) - 1

		# Shifted by one char
		input_string_labels = sequence_string_labels[:-1]
		output_string_labels = sequence_string_labels[1:]

		# pad sequence so that all of them have the same length
		# Otherwise the batching won't work
		input_string_labels_padded = pad_sequence(input_string_labels, max_length = self.max_text_len)

		output_string_labels_padded = pad_sequence(output_string_labels, max_length = self.max_text_len,
												   pad_label = -100)

		return (torch.LongTensor(input_string_labels_padded),
				torch.LongTensor(output_string_labels_padded),
				torch.LongTensor([sequence_length]))


if __name__ == "__main__":
	dset = LyricsGenerationDataset("lyrics")
	print(dset.__getitem__(0))
	print(dset.__getitem__(1))
	print(dset.__getitem__(2))
	print(dset.__getitem__(3))
