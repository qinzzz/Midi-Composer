"""
trainer.py

@time: 10/22/20
@author: Qinxin Wang

@desc: training and evaluation
"""
import unittest

import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from pypianoroll import Multitrack
from torch import optim
from torch.utils.data import DataLoader

from composer import RNNComposer
from midiparser import post_process_sequence_batch
from dataset.pianorolldataset import PianorollDataset


class Trainer:
	def __init__(self, train_dataloader, val_dataloader):
		print("init trainer... ")
		self.INIT_LR = 1e-3
		self.EPOCHS = 1

		self.composer = RNNComposer()
		self.composer.model = self.composer.model.cuda()

		self.train_dataloader = train_dataloader
		self.val_dataloader = val_dataloader

		self.optimizer = optim.Adam(self.composer.model.parameters(), lr = self.INIT_LR)

		self.criterion = nn.CrossEntropyLoss().cuda()
		self.criterion_val = nn.CrossEntropyLoss(size_average = False).cuda()

		self.clip = 1.0
		self.epochs_number = 12000000
		self.sample_history = []
		self.best_val_loss = float("inf")

		self.val_list = []
		self.loss_list = []

	def train(self):
		print("Start training...")

		for epoch_number in range(self.epochs_number):

			for batch in tqdm(self.train_dataloader):
				input_sequences_batch, output_sequences_batch, sequences_lengths = post_process_sequence_batch(batch)

				output_sequences_batch = output_sequences_batch.contiguous().view(-1).cuda()

				input_sequences_batch = input_sequences_batch.cuda()

				self.optimizer.zero_grad()

				logits, _ = self.composer.model(input_sequences_batch, sequences_lengths)

				loss = self.criterion(logits, output_sequences_batch)
				self.loss_list.append(loss.item())
				loss.backward()

				torch.nn.utils.clip_grad_norm(self.composer.model.parameters(), self.clip)

				self.optimizer.step()

			current_val_loss = self.validate()
			self.val_list.append(current_val_loss)

			if current_val_loss < best_val_loss:
				torch.save(self.composer.model.state_dict(), 'music_rnn_pianode.pth')
				best_val_loss = current_val_loss

	def validate(self):
		full_val_loss = 0.0
		overall_sequence_length = 0.0

		for batch in self.val_dataloader:
			post_processed_batch_tuple = post_process_sequence_batch(batch)

			input_sequences_batch, output_sequences_batch, sequences_lengths = post_processed_batch_tuple

			output_sequences_batch = output_sequences_batch.contiguous().view(-1).cuda()

			input_sequences_batch = input_sequences_batch.cuda()

			logits, _ = self.composer.model(input_sequences_batch, sequences_lengths)

			loss = self.criterion_val(logits, output_sequences_batch)

			full_val_loss += loss.item()
			overall_sequence_length += sum(sequences_lengths)

		return full_val_loss / (overall_sequence_length * 88)

	def get_composer(self):
		return self.composer


class TrainerTest(unittest.TestCase):
	def test_train(self):
		trainer = Trainer()
		trainer.train()

		# composer = trainer.get_composer()
		# # [batch, seq_len, note_nb]
		#
		# input_file = "data/midi/younight.mid"
		# given_song = Multitrack(input_file)
		# given_song.binarize()
		# given_pianoroll = trainer.dataset._process_song(given_song)
		# given_notes = given_pianoroll[1, :500]
		# given_notes = torch.tensor(given_notes).float().unsqueeze(0)
		#
		# next_notes = composer.predict_next_sequence(given_notes, length = 5000) #  [batch, length, note_nb]
		# print(next_notes.nonzero())
		#
		# next_notes = next_notes.numpy()
		# auto_song = trainer.dataset._np_to_song(next_notes)
		#
		#
		#
		# out_file = "data/midi/younight_auto.mid"
		# auto_song.write(out_file)
