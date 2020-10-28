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

from midiparser import post_process_sequence_batch


class Trainer:
	def __init__(self, task, composer, dataset, train_dataloader, val_dataloader, lr = 1e-3, epochs = 100000, parallel=False):
		print("init trainer... ")
		print("args: lr = {}, epochs = {}".format(lr, epochs))

		self.task = task

		self.init_lr = lr
		self.epochs_number = epochs

		self.dataset = dataset
		self.composer = composer

		self.train_dataloader = train_dataloader
		self.val_dataloader = val_dataloader

		self.optimizer = optim.Adam(self.composer.model.parameters(), lr = self.init_lr)

		self.loss_func = nn.CrossEntropyLoss().cuda()
		self.loss_func_val = nn.CrossEntropyLoss(size_average = False).cuda()

		self.clip = 1.0
		self.sample_history = []
		self.best_val_loss = float("inf")

		self.val_list = []

	def train(self):
		print("Start training...")

		for epoch_number in tqdm(range(self.epochs_number)):
			print("Epoch: {}".format(epoch_number))

			for batch in self.train_dataloader:
				input_sequences_batch, output_sequences_batch, sequences_lengths = post_process_sequence_batch(batch)

				output_sequences_batch = output_sequences_batch.contiguous().view(-1).cuda()
				input_sequences_batch = input_sequences_batch.cuda()
				self.optimizer.zero_grad()

				logits, _ = self.composer.model(input_sequences_batch, sequences_lengths)
				loss = self.loss_func(logits, output_sequences_batch)
				loss.backward()

				torch.nn.utils.clip_grad_norm(self.composer.model.parameters(), self.clip)
				self.optimizer.step()

			current_val_loss = self.validate()
			self.val_list.append(current_val_loss)
			print("val loss: {}".format(current_val_loss))

			if current_val_loss < self.best_val_loss:
				torch.save(self.composer.model.state_dict(), '{}_model_best_{}_{}_{}.pth'.format(self.task, self.dataset, self.composer.layers, self.composer.hidden_size))
				self.best_val_loss = current_val_loss

			if epoch_number % 200 == 0 and epoch_number > 0:
				torch.save(self.composer.model.state_dict(), '{}_model_epoch{}_{}_{}.pth'.format(self.task, epoch_number, self.composer.layers, self.composer.hidden_size))

	def validate(self):
		full_val_loss = 0.0
		overall_sequence_length = 0.0

		for batch in self.val_dataloader:
			post_processed_batch_tuple = post_process_sequence_batch(batch)

			input_sequences_batch, output_sequences_batch, sequences_lengths = post_processed_batch_tuple
			output_sequences_batch = output_sequences_batch.contiguous().view(-1).cuda()
			input_sequences_batch = input_sequences_batch.cuda()

			logits, _ = self.composer.model(input_sequences_batch, sequences_lengths)
			loss = self.loss_func_val(logits, output_sequences_batch)

			full_val_loss += loss.item()
			overall_sequence_length += sum(sequences_lengths)

		return full_val_loss / (overall_sequence_length * self.composer.input_size)

	def get_composer(self):
		return self.composer
