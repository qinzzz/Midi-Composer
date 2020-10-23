"""
trainer.py

@time: 10/22/20
@author: Qinxin Wang

@desc: training and evaluation
"""
import unittest

import torch
from tqdm import tqdm
import numpy as np
from pypianoroll import Multitrack
from torch import optim
from torch.utils.data import DataLoader

from composer import RNNComposer
from dataset.pianorolldataset import PianorollDataset


class Trainer:
	def __init__(self):
		print("init trainer... ")
		self.INIT_LR = 1e-3
		self.EPOCHS = 1

		self.composer = RNNComposer()

		self.dataset = PianorollDataset()
		self.data_loader = DataLoader(self.dataset, batch_size = 1)
		print("data loaded.")

		self.optimizer = optim.Adam(self.composer.model.parameters(), lr = self.INIT_LR)

	def train(self):
		print("Start training...")

		for epoch in range(self.EPOCHS):
			print("--- Epoch {} ---".format(epoch))
			total_loss = 0.0
			num_batches = 0
			for song_input, song_output in tqdm(self.data_loader):
				# debug
				if len(torch.nonzero(song_input))==0:
					# print("emtpy training sample!")
					continue

				num_batches += 1
				# input/output: matrix[batch, time, pitch]
				self.composer.model.train()
				self.optimizer.zero_grad()

				logits, _ = self.composer.model.forward(song_input)  # [batch, seq_len, note_nb]

				loss = self.composer.loss_function(logits, song_output)
				total_loss += loss
				loss.backward()
				self.optimizer.step()

			total_loss /= num_batches
			print("Training epoch {}, loss: {}".format(epoch, total_loss))

	def get_composer(self):
		return self.composer


class TrainerTest(unittest.TestCase):
	def test_train(self):
		trainer = Trainer()
		trainer.train()
		composer = trainer.get_composer()
		# [batch, seq_len, note_nb]

		input_file = "data/midi/younight.mid"
		given_song = Multitrack(input_file)
		given_song.binarize()
		given_pianoroll = trainer.dataset._process_song(given_song)
		given_notes = given_pianoroll[1, :500]
		given_notes = torch.tensor(given_notes).float().unsqueeze(0)

		next_notes = composer.predict_next_sequence(given_notes, length = 5000) #  [batch, length, note_nb]
		print(next_notes.nonzero())

		next_notes = next_notes.numpy()
		auto_song = trainer.dataset._np_to_song(next_notes)



		out_file = "data/midi/younight_auto.mid"
		auto_song.write(out_file)
