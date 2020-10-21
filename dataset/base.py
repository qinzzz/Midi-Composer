"""
base.py

@time: 10/21/20
@author: Qinxin Wang

@desc:
"""
import os
import pickle

from torch.utils.data import Dataset
from tqdm import tqdm


class BaseDataset(Dataset):
	def __init__(self):
		super(BaseDataset, self).__init__()

		self.SONG_MARKER = "raw songs"
		self.PROCESSED_SONG_MARKER = "processed songs"

		self.data_root = "../data"
		self.midi_dir = "midi"
		self.song_file = "song"
		self.processed_song_file = "song_dataset"

		self.midi_files = []
		self.songs = []
		self.processed_songs = []

	def _load_midi_files(self):
		if not os.path.isdir(self.data_root):
			raise RuntimeError("data root [{}] is not a valid directory.".format(self.data_root))

		midi_path = os.path.join(self.data_root, self.midi_dir)
		print("Loading midi files from directory [{}]".format(midi_path))

		for root, dirs, files in os.walk(midi_path):
			for file in files:
				if not self._is_midi(file):
					pass
				self.midi_files.append(os.path.join(self.data_root, self.midi_dir, file))

	def _load_data_as_song(self):
		if len(self.midi_files) == 0:
			print("midi files not find! Start loading ...")
			self._load_midi_files()

		print("Start creating songs from midi files...")
		for midi_file in tqdm(self.midi_files):

			try:
				song = self._load_song_from_file(midi_file)  # song: Song
			except Exception as e:
				tqdm.write('File ignored ({}): {}'.format(midi_file, e))
				pass

			processed_song = self._preprocess(song)
			if processed_song is None:
				pass

			self.songs.append(song)
			self.processed_songs.append(processed_song)

			self._record(midi_file, song)

	def _save_data_as_song(self):
		self._save_songs(self.SONG_MARKER)
		self._save_processed_songs(self.PROCESSED_SONG_MARKER)

	def _load_songs(self):
		song_path = os.path.join(self.data_root, self.song_file)
		if not os.path.exists(song_path):
			raise RuntimeError("{} does not exist!".format(song_path))

		with open(song_path, "r") as f:
			songs = pickle.load(f)
		return songs["songs"]

	def _load_processed_songs(self):
		song_path = os.path.join(self.data_root, self.processed_song_file)
		if not os.path.exists(song_path):
			raise RuntimeError("{} does not exist!".format(song_path))

		with open(song_path, "r") as f:
			songs = pickle.load(f)
		return songs["songs"]

	def _save_songs(self, marker):
		song_path = os.path.join(self.data_root, self.song_file)
		if os.path.exists(song_path):
			print("Song file {} already exists!", song_path)
			return

		with open(song_path, "w") as f:
			data = {
				'version': marker,
				'songs': self.songs
			}
			pickle.dump(data, f, -1)

	def _save_processed_songs(self, marker):
		song_path = os.path.join(self.data_root, self.processed_song_file)
		if os.path.exists(song_path):
			print("Song file {} already exists!", song_path)
			return

		with open(song_path, "wb") as f:
			data = {
				'version': marker,
				'songs': self.processed_songs
			}
			pickle.dump(data, f, -1)

	def _is_midi(self, filename):
		if filename.endswith(".mid"):
			return True
		return False

	def _load_song_from_file(self, file):
		"""

		:param file:
		:return: song object
		"""
		raise NotImplementedError

	def _preprocess(self, song):
		"""
		convert song object to array
		:param song:
		:return:
		"""
		raise NotImplementedError

	def _record(self, file, song):
		raise NotImplementedError

	def __getitem__(self, item):
		raise NotImplementedError

	def __len__(self):
		raise NotImplementedError
