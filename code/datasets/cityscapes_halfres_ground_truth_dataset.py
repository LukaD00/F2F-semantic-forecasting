import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class CityscapesHalfresGroundTruthDataset(Dataset):
	"""
	A Dataset object that groups together past feature tensors and future features with corresponding ground truth
	labels, used for validating models.

	Past features are used to evaluate forecasting models,
	while future features can be used to evaluate oracles.
	"""

	def __init__(self, train = False, num_past = 4, future_distance = 3, print_files = False):
		"""
		Initializes the dataset by loading the file names and grouping them appropriately 
		(num_past past tensors with one corresponding ground truth set of labels).

		Args:
			train (bool): If true, loads training dataset, else validation dataset (default)
			num_past (int): How many past sets of features (images) will be provided for every set of features.
			future_distance (int): How many frames away will the future image be (default 3)
			print_files (bool): If true, will print file paths during indexing
		"""
		super().__init__()
		
		self.tensor_dir = "../cityscapes_halfres_features_r18/"
		self.truth_dir = "../cityscapes-gt/"
		if (train):
			self.tensor_dir += "train"
			self.truth_dir += "train"
		else:
			self.tensor_dir += "val"
			self.truth_dir += "val"

		self.num_past = num_past
		self.print_files = print_files
		
		self.items = [] # array that stores groups of (past tensors, future tensor, ground truth labels)
		tensor_files = os.listdir(self.tensor_dir)
		truth_files = os.listdir(self.truth_dir)   
		seq_length = 7	# how many frames per sequence we have in tensor dataset
		seq_step = 3	# step between frames
		for i in range(0, len(tensor_files), seq_length):
			sequence = tensor_files[i:i+seq_length]
			
			future_features = sequence[-1]
			
			last_index = int(len(sequence) - future_distance/seq_step)
			first_index = int(last_index - num_past)
			past_features = sequence[first_index:last_index]

			ground_truth = truth_files[i//seq_length]

			self.items.append(past_features + [future_features,ground_truth])


	def __len__(self):
		"""
		Returns length of the dataset - more precisely, the amount of (past features, future features, ground truth) items.
		"""
		return len(self.items)

	
	def __getitem__(self, idx):
		"""
		Returns the idx-th item in the dataset - a tensor representing a set of several past tensors, 
		a future tensor, and labeled future ground truth.

		Args:
			idx (int): Index of the wanted item.

		Returns:
			((128 * num_past, 16, 32) torch.Tensor), ((128, 16, 32) torch.Tensor), ((1024, 512) np.array)
		"""
		item = self.items[idx]
		past_features = torch.from_numpy(np.vstack([np.load(os.path.join(self.tensor_dir, img)) for img in item[0:self.num_past]]))
		future_features = future_features = torch.from_numpy(np.load(os.path.join(self.tensor_dir, item[-2])))
		ground_truth = np.array(Image.open(os.path.join(self.truth_dir, item[-1])))

		if (self.print_files):
			print("Past: ")
			for i in range(self.num_past):
				print("\t" + item[i])
			print("Future: ")
			print("\t" + item[-2])
			print("Ground truth: ")
			print("\t" + item[-1])			

		return past_features, future_features, ground_truth