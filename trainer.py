"""
trainer.py

@time: 10/22/20
@author: Qinxin Wang

@desc: training and evaluation
"""
import unittest

from torch import optim
from torch.utils.data import DataLoader

from composer import RNNComposer
from dataset.pianorolldataset import PianorollDataset


class Trainer:
	def __init__(self):
		print("init trainer... ")
		self.INIT_LR = 1e-3
		self.EPOCHS = 10

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
			for song_input, song_output in self.data_loader:
				num_batches += 1
				# input/output: matrix[batch, time, pitch = 88]
				self.composer.model.train()
				self.optimizer.zero_grad()

				logits, _ = self.composer.model.forward(song_input)  # [batch, seq_len, note_nb]

				loss = self.composer.loss_function(logits, song_output)
				total_loss += loss
				loss.backward()
				self.optimizer.step()

			total_loss /= num_batches
			print("Training epoch {}, loss: {}".format(epoch, total_loss))


class TrainerTest(unittest.TestCase):
	def test_train(self):
		trainer = Trainer()
		trainer.train()
