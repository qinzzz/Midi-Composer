"""
track.py

@time: 10/20/20
@author: Qinxin Wang

@desc:
	based on Conchylicultor's repo.
	https://github.com/Conchylicultor/MusicGenerator

"""


class Track:
	""" Structure which encapsulate a track of the song
	Ideally, each track should correspond to a single instrument and one channel. Multiple tracks could correspond
	to the same channel if different instruments use the same channel.
	"""

	def __init__(self):
		# self.tempo_map = None  # Use a global tempo map
		self.id = 0
		self.instrument = None
		self.notes = []  # List[Note]
		self.is_drum = False

	def set_instrument(self, msg):
		""" Initialize from a mido message

		:msg:  mido.MidiMessage. a valid control_change message
		"""
		if self.instrument is not None:  # Already an instrument set
			return False

		assert msg.type == 'program_change'

		self.instrument = msg.program
		if msg.channel == 9 or msg.program > 112:  # Warning: Mido shift the channels (start at 0)
			self.is_drum = True

		return True

	def display(self):
		print("--- Track {} ---".format(self.id))
		for note in self.notes:
			note.display()