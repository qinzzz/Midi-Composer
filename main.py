"""
main.py

@time: 10/20/20
@author: Qinxin Wang

@desc:
"""
import os
import string
import argparse
import torch
import numpy as np
from notedataset import NotesGenerationDataset
from lyricsdataset import LyricsGenerationDataset
from trainer import Trainer
from composer import RNNSongComposer, LyricComposer
from torch.utils.data.sampler import SubsetRandomSampler

dataset_dict = {"piano": "Piano-midi", "nottingham": "Nottingham"}


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--batch", type = int, default = 32)
	parser.add_argument("--lr", type = float, default = 1e-3)
	parser.add_argument("--epochs", type = int, default = 10000)
	parser.add_argument("--dataset", type = str, default = "nottingham")
	parser.add_argument("--parallel", action = "store_true")
	parser.add_argument("--task", type = str, default = "lyric")

	args = parser.parse_args()
	return args


def train_song_dataset(args):
	dataset = dataset_dict[args.dataset]

	trainset = NotesGenerationDataset(os.path.join(dataset, "train"))
	trainset.update_the_max_length()

	trainset_loader = torch.utils.data.DataLoader(trainset, batch_size = args.batch,
												  shuffle = True, num_workers = 4, drop_last = True)

	valset = NotesGenerationDataset(os.path.join(dataset, "valid"))
	valset.update_the_max_length()

	valset_loader = torch.utils.data.DataLoader(valset, batch_size = args.batch, shuffle = False, num_workers = 4,
												drop_last = False)
	print("data loaded.")

	composer = RNNSongComposer()
	trainer = Trainer(composer, args.task, args.dataset, trainset_loader, valset_loader, args.lr, args.epochs,
					  parallel = args.parallel)
	trainer.train()


def train_lyrics_dataset(args):
	dataset = LyricsGenerationDataset("lyrics")
	dataset_size = len(dataset)
	indices = list(range(dataset_size))
	validation_split = 0.1
	split = int(np.floor(validation_split * dataset_size))
	train_indices, val_indices = indices[split:], indices[:split]

	train_sampler = SubsetRandomSampler(train_indices)
	valid_sampler = SubsetRandomSampler(val_indices)

	trainset_loader = torch.utils.data.DataLoader(dataset, batch_size = args.batch, num_workers = 4, drop_last = True,
												  sampler = train_sampler)
	valset_loader = torch.utils.data.DataLoader(dataset, batch_size = args.batch, num_workers = 4, drop_last = True,
												sampler = valid_sampler)
	print("data loaded.")

	composer = LyricComposer()
	trainer = Trainer(composer, args.task, args.dataset, trainset_loader, valset_loader, args.lr, args.epochs,
					  parallel = args.parallel)
	trainer.train()


if __name__ == "__main__":
	args = parse_args()
	print(args)
	if args.task == "lyric":
		train_lyrics_dataset(args)
	else:
		train_song_dataset(args)
