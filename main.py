"""
main.py

@time: 10/20/20
@author: Qinxin Wang

@desc:
"""

import composer
import torch
from trainer import Trainer
from midiparser import NotesGenerationDataset

if __name__ == "__main__":
	trainset = NotesGenerationDataset('Piano-midi/train/')
	trainset.update_the_max_length()

	trainset_loader = torch.utils.data.DataLoader(trainset, batch_size = 4,
												  shuffle = True, num_workers = 4, drop_last = True)

	valset = NotesGenerationDataset('Piano-midi/valid/')
	valset.update_the_max_length()

	valset_loader = torch.utils.data.DataLoader(valset, batch_size = 4, shuffle = False, num_workers = 4,
												drop_last = False)
	print("data loaded.")

	trainer = Trainer(trainset_loader, valset_loader)
	trainer.train()
