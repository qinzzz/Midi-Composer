"""
pianorolldataset.py

@time: 10/21/20
@author: Qinxin Wang

@desc:
"""
import os
import pickle
import unittest

import numpy as np
import torch
from pypianoroll import Multitrack
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.base import BaseDataset

MIDI_NOTES_RANGE = [21, 109]  # Min and max (included) midi note on a piano


class PianorollDataset(BaseDataset):
	def __init__(self, data_root = "data"):
		super(PianorollDataset, self).__init__(data_root)
		self.example_version = "v0.1"
		self.chunk_size = 100
		self.time_unit = 8 # 1/8 note
		self.example_file = "training_exs_t{}_c{}.pkl".format(self.time_unit, self.chunk_size)

		# initialize

		self._prepare_examples()

	def _load_song_from_file(self, file) -> Multitrack:
		return Multitrack(file)

	def _process_song(self, song: Multitrack) -> np.ndarray:
		"""
		convert song object to array
		:param song:
		:return: normalized numpy array [track, time, pitch]
		"""
		pianoroll = song.get_stacked_pianoroll()  # [time, pitch, track]

		# normalize by 1/32 note
		scale = song.beat_resolution // (self.time_unit/4)
		pianoroll = pianoroll[0:len(song.tempo):scale]
		# trans pose
		pianoroll = np.transpose(pianoroll, (2, 0, 1))  # [time, pitch, track] --> [track, time, pitch]
		# the first track is not considered
		pianoroll = pianoroll[1:, :, MIDI_NOTES_RANGE[0]:MIDI_NOTES_RANGE[1]]

		if pianoroll.shape[0] == 0:
			return None

		print(pianoroll.shape)
		return pianoroll

	def _prepare_examples(self):
		example_path = os.path.join(self.data_root, self.example_file)
		if os.path.exists(example_path):
			print("loading training examples from [{}]", example_path)
			self._load_training_examples()

		else:
			print("load processed songs...")
			self._load_data_as_song()

			print("Generating training examples...")
			self.ex_input = []
			self.ex_output = []
			for pianoroll in tqdm(self.processed_songs):
				# truncate song into fixed length
				for track in pianoroll:
					# [time, pitch]
					chunk_pieces = [track[i:i + self.chunk_size] for i in range(0, len(track), self.chunk_size)]
					# chunk_pieces: [chunks, chunk_size, pitch]
					for chunk in chunk_pieces:
						# chunk: [chunk_size, pitch]
						if len(chunk) != self.chunk_size:
							chunk = np.pad(chunk, [(0, self.chunk_size - len(chunk)), (0, 0)], mode = "constant")
						self.ex_input.append(chunk[:-1])
						self.ex_output.append(chunk[1:])

			self.ex_input_tensor = torch.tensor(self.ex_input).float()
			self.ex_output_tensor = torch.tensor(self.ex_output).float()

			self._save_training_examples()
			print("Examples saved. Input tensor size: {}".format(self.ex_input_tensor.shape))

	def _record(self, midi_file, song):
		tqdm.write('Song loaded {}'.format(midi_file))
		tqdm.write("Song properties: tempo = {}, resolution = {}, time = {}".format(song.tempo[0], song.beat_resolution,
																					len(song.tempo)))

	def _save_training_examples(self):
		example_path = os.path.join(self.data_root, self.example_file)
		with open(example_path, "wb") as f:
			data = {
				'version': self.example_version,
				'input': self.ex_input_tensor,
				"output": self.ex_output_tensor
			}
			pickle.dump(data, f, -1)

	def _load_training_examples(self):
		example_path = os.path.join(self.data_root, self.example_file)
		with open(example_path, "rb") as f:
			examples = pickle.load(f)
		self.ex_input_tensor = examples["input"]
		self.ex_output_tensor = examples["output"]

	def __len__(self):
		return self.ex_input_tensor.size(0)

	def __getitem__(self, item):
		return self.ex_input_tensor[item], self.ex_output_tensor[item]


class PianorollDatasetTest(unittest.TestCase):
	def test(self):
		dataset = PianorollDataset("../data")
		data_loader = DataLoader(dataset, batch_size = 1)

		for input, output in data_loader:
			print(input)
			self.assertEqual(input, 0)


if __name__ == '__main__':
	unittest.main()
