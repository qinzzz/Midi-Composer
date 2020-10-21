"""
pianorolldataset.py

@time: 10/21/20
@author: Qinxin Wang

@desc:
"""
import unittest

import numpy as np
import torch
from pypianoroll import Multitrack
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.base import BaseDataset

MIDI_NOTES_RANGE = [21, 109]  # Min and max (included) midi note on a piano


class PianorollDataset(BaseDataset):
	def __init__(self):
		super(PianorollDataset, self).__init__()
		self.PROCESSED_SONG_MARKER = "piano roll matrix"

		# initialize
		self._load_data_as_song()
		self._save_processed_songs(self.PROCESSED_SONG_MARKER)

	def _load_song_from_file(self, file) -> Multitrack:
		return Multitrack(file)

	def _preprocess(self, song: Multitrack) -> np.ndarray:
		"""

		:param song:
		:return: normalized numpy array [num_track, pitch_range, time]
		"""
		pianoroll = song.get_stacked_pianoroll()  # [time, pitch, track]

		# normalize by 1/32 note
		scale = song.beat_resolution // 8
		pianoroll = torch.tensor(pianoroll[0:len(song.tempo):scale])
		# trans pose
		pianoroll = torch.transpose(pianoroll, 0, 2)  # [track, pitch, time]
		# the first track is not considered
		pianoroll = pianoroll[1:, MIDI_NOTES_RANGE[0]:MIDI_NOTES_RANGE[1], :]

		if pianoroll.size(1) == 0:
			return None

		return pianoroll

	def _record(self, midi_file, song):
		tqdm.write('Song loaded {}'.format(midi_file))
		tqdm.write("Song properties: tempo = {}, resolution = {}, time = {}".format(song.tempo[0], song.beat_resolution, len(song.tempo)))

	def __len__(self):
		return len(self.songs)

	def __getitem__(self, item):
		return self.processed_songs[item]


class PianorollDatasetTest(unittest.TestCase):
	def test(self):
		dataset = PianorollDataset()
		data_loader = DataLoader(dataset, batch_size = 1)

		for i in data_loader:
			print(i.shape)
			self.assertIsNotNone(i)


if __name__ == '__main__':
	unittest.main()
