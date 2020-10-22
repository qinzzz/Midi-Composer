"""
songdataset.py

@time: 10/21/20
@author: Qinxin Wang

@desc: from midi to dataset
"""
from dataset.base import BaseDataset


class SongDataset(BaseDataset):
	def __init__(self):
		super(SongDataset, self).__init__()

	def __getitem__(self, item):
		raise NotImplementedError

	def __len__(self):
		raise NotImplementedError

	def _process_song(self, song):
		raise NotImplementedError
