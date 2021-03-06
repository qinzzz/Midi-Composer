"""
trainer.py

@time: 10/22/20
@author: Qinxin Wang

@desc: training and evaluation
"""

import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from midiparser import process_midi_sequence_batch, process_lyric_sequence_batch


class Trainer:
	def __init__(self, composer, task, dataset, train_dataloader, val_dataloader, lr = 1e-3, epochs = 100000,
				 parallel = False):
		print("init trainer... ")
		print("args: lr = {}, epochs = {}".format(lr, epochs))
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		print("device:{}, number:{}".format(self.device, torch.cuda.device_count()))

		self.task = task

		self.init_lr = lr
		self.epochs_number = epochs

		self.dataset = dataset
		self.composer = composer

		if torch.cuda.device_count()>1:
			print("Use", torch.cuda.device_count(), "GPUs.")
			self.composer.model = nn.DataParallel(self.composer.model)
		self.composer.model.to(self.device)

		self.train_dataloader = train_dataloader
		self.val_dataloader = val_dataloader

		self.optimizer = optim.Adam(self.composer.model.parameters(), lr = self.init_lr)

		self.loss_func = nn.CrossEntropyLoss(size_average = False).to(self.device)
		self.loss_func_val = nn.CrossEntropyLoss(size_average = False).to(self.device)

		self.clip = 1.0
		self.sample_history = []
		self.best_val_loss = float("inf")

		self.train_loss_list = []
		self.val_list = []

	def train(self):
		print("Start training...")

		for epoch_number in range(self.epochs_number):
			print("Epoch: {}".format(epoch_number))
			train_loss = 0.
			num_batches = 0
			overall_sequence_length = 0.

			for batch in self.train_dataloader:
				num_batches+=1

				if self.task == "lyric":
					input_sequences_batch, output_sequences_batch, sequences_lengths = process_lyric_sequence_batch(
						batch)
				# print(input_sequences_batch.shape, output_sequences_batch.shape, sequences_lengths)
				else:
					input_sequences_batch, output_sequences_batch, sequences_lengths = process_midi_sequence_batch(
						batch)
				# print(input_sequences_batch.shape, output_sequences_batch.shape, sequences_lengths)

				output_sequences_batch = output_sequences_batch.contiguous().view(-1).to(self.device)
				input_sequences_batch = input_sequences_batch.to(self.device)
				self.optimizer.zero_grad()

				logits, _ = self.composer.model(input_sequences_batch, sequences_lengths)
				loss = self.loss_func(logits, output_sequences_batch)
				loss.backward()
				train_loss += loss.item()
				overall_sequence_length += sum(sequences_lengths)

				torch.nn.utils.clip_grad_norm_(self.composer.model.parameters(), self.clip)
				self.optimizer.step()

			print("train loss: {}".format(train_loss / (overall_sequence_length * self.composer.input_size)))

			current_val_loss = self.validate()
			self.val_list.append(current_val_loss)
			print("val loss: {}".format(current_val_loss))

			if current_val_loss < self.best_val_loss:
				torch.save(self.composer.model.state_dict(),
						   '{}_model_best_{}_{}_{}.pth'.format(self.task, self.dataset, self.composer.layers,
															   self.composer.hidden_size))
				self.best_val_loss = current_val_loss

			if epoch_number % 500 == 0 and epoch_number > 0:
				print("Save model at epoch {}".format(epoch_number))
				torch.save(self.composer.model.state_dict(),
						   '{}_model_epoch{}_{}_{}.pth'.format(self.task, epoch_number, self.composer.layers,
															   self.composer.hidden_size))

	def train_lyrics(self):
		for epoch_number in range(self.epochs_number):

			for batch in self.train_dataloader:
				post_processed_batch_tuple = process_lyric_sequence_batch(batch)

				input_sequences_batch, output_sequences_batch, sequences_lengths = post_processed_batch_tuple

				output_sequences_batch_var = output_sequences_batch.contiguous().view(-1).to(self.device)
				input_sequences_batch_var = input_sequences_batch.to(self.device)

				self.optimizer.zero_grad()

				logits, _ = self.composer.model(input_sequences_batch_var, sequences_lengths)

				loss = self.loss_func(logits, output_sequences_batch_var)
				loss.backward()

				# torch.nn.utils.clip_grad_norm(rnn.parameters(), clip)

				self.optimizer.step()

	def validate(self):
		full_val_loss = 0.0
		overall_sequence_length = 0.0

		for batch in self.val_dataloader:
			if self.task == "lyric":
				input_sequences_batch, output_sequences_batch, sequences_lengths = process_lyric_sequence_batch(
					batch)
			else:
				input_sequences_batch, output_sequences_batch, sequences_lengths = process_midi_sequence_batch(batch)

			output_sequences_batch = output_sequences_batch.contiguous().view(-1).to(self.device)
			input_sequences_batch = input_sequences_batch.to(self.device)

			logits, _ = self.composer.model(input_sequences_batch, sequences_lengths)
			loss = self.loss_func_val(logits, output_sequences_batch)

			full_val_loss += loss.item()
			overall_sequence_length += sum(sequences_lengths)

		return full_val_loss / (overall_sequence_length * self.composer.input_size)

	def get_composer(self):
		return self.composer
