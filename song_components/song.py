"""
song.py

@time: 10/20/20
@author: Qinxin Wang

@desc:
	based on Conchylicultor's repo.
	https://github.com/Conchylicultor/MusicGenerator

"""
import operator  # To rescale the song


class Song:
	""" Structure which encapsulate the song data
	"""
	# Define the time unit
	# Invert of time note which define the maximum resolution for a song. Ex: 2 for 1/2 note, 4 for 1/4 of note
	MAXIMUM_SONG_RESOLUTION = 4
	NOTES_PER_BAR = 4  # Waltz not supported

	def __init__(self):
		self.ticks_per_beat = 96
		self.tempo_map = []
		self.tracks = []  # List[Track]
		self.track_nb = 0

	def __len__(self):
		""" Return the absolute tick when the last note end
		Note that the length is recomputed each time the function is called
		"""
		return max([max([n.tick + n.duration for n in t.notes]) for t in self.tracks])

	def _get_scale(self):
		"""
		Compute the unit scale factor for the song
		The scale factor allow to have a tempo independent time unit, to represent the song as an array
		of dimension [key, time_unit]. Once computed, one has just to divide (//) the ticks or multiply
		the time units to go from one representation to the other.

		Return:
			int: the scale factor for the current song
		"""

		# TODO: Assert that the scale factor is not a float (the % =0)
		return 4 * self.ticks_per_beat // (Song.MAXIMUM_SONG_RESOLUTION * Song.NOTES_PER_BAR)

	def normalize(self, inverse = False):
		"""
		Transform the song into a tempo independent song
		Warning: If the resolution of the song is more fine that the given
		scale, some information will be definitively lost
		Args:
			inverse (bool): if true, we reverse the normalization
		"""
		scale = self._get_scale()
		op = operator.floordiv if not inverse else operator.mul

		# Shifting all notes
		for track in self.tracks:
			for note in track.notes:
				note.tick = op(note.tick, scale)  # //= or *=

	def display(self):
		print("--- Song ---")
		print(("ticks pre beat: {}, scale: {}".format(self.ticks_per_beat, self._get_scale())))
		print(("track numbers:", self.track_nb))
		print(("tempo map:", self.tempo_map))

		for i, track in enumerate(self.tracks):
			track.display()

