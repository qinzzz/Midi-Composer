"""
note.py

@time: 10/20/20
@author: Qinxin Wang

@desc:
	based on Conchylicultor's repo.
	https://github.com/Conchylicultor/MusicGenerator

"""

MIDI_NOTES_RANGE = [21, 108]  # Min and max (included) midi note on a piano
NB_NOTES = MIDI_NOTES_RANGE[1] - MIDI_NOTES_RANGE[0] + 1  # 88 of keys for a piano
BAR_DIVISION = 16  # Nb of tics in a bar


class Note:
	""" Structure which encapsulate the song data
	"""

	def __init__(self):
		self.tick = 0
		self.note = 0
		self.duration = 32

	def get_relative_note(self):
		""" Convert the absolute midi position into the range given by MIDI_NOTES_RANGE
		Return
		:int: The new position relative to the range (position on keyboard)
		"""
		return self.note - MIDI_NOTES_RANGE[0]

	def set_relative_note(self, rel):
		""" Convert given note into a absolute midi position
		:rel: Int. The new position relative to the range (position on keyboard)
		"""
		assert NB_NOTES >= rel >= 0
		self.note = rel + MIDI_NOTES_RANGE[0]

	def display(self):
		print("note:{}, tick: {}, duration:{}".format(self.note, self.tick, self.duration))
