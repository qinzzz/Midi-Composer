"""
main.py

@time: 10/20/20
@author: Qinxin Wang

@desc:
"""
import os
import argparse
import torch

from notedataset import NotesGenerationDataset
from lyricsdataset import LyricsGenerationDataset
from trainer import Trainer

dataset_dict = {"piano": "Piano-midi", "nottingham": "Nottingham"}

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--batch", type = int, default = 32)
	parser.add_argument("--lr", type=float, default = 1e-3)
	parser.add_argument("--epochs", type = int, default = 100000)
	parser.add_argument("--dataset", type = str, default = "nottingham")

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

	trainer = Trainer(args.dataset, trainset_loader, valset_loader, args.lr, args.epochs)
	trainer.train()


def train_lyrics_dataset():
	trainset = LyricsGenerationDataset(csv_file_path = 'songdata.csv')

	trainset_loader = torch.utils.data.DataLoader(trainset, batch_size = 50,
												  shuffle = True, num_workers = 4, drop_last = True)

if __name__ == "__main__":
	args = parse_args()
	print(args)
	train_song_dataset(args)
