"""
main.py

@time: 10/20/20
@author: Qinxin Wang

@desc:
"""
import argparse
import torch

from midiparser import NotesGenerationDataset
from trainer import Trainer

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--batch", type = int, default = 12)
	parser.add_argument("--lr", type=float, default = 1e-3)
	parser.add_argument("--epochs", type = int, default = 1000)

	args = parser.parse_args()
	return args


if __name__ == "__main__":
	args = parse_args()
	print(args)

	trainset = NotesGenerationDataset('Piano-midi/train/')
	trainset.update_the_max_length()

	trainset_loader = torch.utils.data.DataLoader(trainset, batch_size = args.batch,
												  shuffle = True, num_workers = 4, drop_last = True)

	valset = NotesGenerationDataset('Piano-midi/valid/')
	valset.update_the_max_length()

	valset_loader = torch.utils.data.DataLoader(valset, batch_size = args.batch, shuffle = False, num_workers = 4,
												drop_last = False)
	print("data loaded.")

	trainer = Trainer(trainset_loader, valset_loader, args.lr, args.epochs)
	trainer.train()
