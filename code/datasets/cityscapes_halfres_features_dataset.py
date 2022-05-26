import os
import torch
import numpy as np
from torch.utils.data import Dataset


class CityscapesHalfresFeaturesDataset(Dataset):
	"""
	A Dataset object for the features of the Cityscapes dataset with half-resolution images.

	Groups together past feature tensors with expected future tensors.
	"""
	
	TRAIN_DIR = "../cityscapes_halfres_features_r18/train"
	VAL_DIR = "../cityscapes_halfres_features_r18/val"

	def __init__(self, train = True, num_past = 4, future_distance = 3, num_sequence = 1, print_files = False):
		"""
		Initializes the dataset by loading the file names and grouping them appropriately 
		(num_past past images with one corresponding future image).

		Args:
			train (bool): If true, training dataset will be loaded. If false, validation dataset will be loaded.
			num_past (int): How many past sets of features (images) will be provided for every set of future features.
			future_distance (int): How many frames away will the future image be (default 3)
			num_sequence (int): How many dataset items will be returned per sequence
			print_files (bool): If true, will print file paths during indexing
		"""
		super().__init__()
		
		self.num_past = num_past
		self.print_files = print_files

		self.file_dir = CityscapesHalfresFeaturesDataset.TRAIN_DIR if train else CityscapesHalfresFeaturesDataset.VAL_DIR
		seq_length = 7	# how many frames per sequence we have in dataset
		seq_step = 3	# step between frames

		self.feature_groups = [] # array that stores groups of (past tensors, future tensor)
		files = os.listdir(self.file_dir)
		for i in range(0, len(files), seq_length):
			for j in range(num_sequence):
				sequence = files[i:i+seq_length-j]
				
				future_features = sequence[-1]
				
				last_index = int(len(sequence) - future_distance/seq_step)
				first_index = int(last_index - num_past)
				past_features = sequence[first_index:last_index]
				self.feature_groups.append(past_features + [future_features])



	def __len__(self):
		"""
		Returns length of the dataset - more precisely, the amount of past+future feature groups.
		"""
		return len(self.feature_groups)

	
	def __getitem__(self, idx):
		"""
		Returns the idx-th item in the dataset - a tensor representing a set of several past tensors, and an expected future tensor.

		Args:
			idx (int): Index of the wanted item.

		Returns:
			((128 * num_past, 16, 32) torch.Tensor), ((128, 16, 32) torch.Tensor)
		"""
		feature_group = self.feature_groups[idx]
		past_features = torch.from_numpy(np.vstack([np.load(os.path.join(self.file_dir, img)) for img in feature_group[0:self.num_past]])) 
		future_features = torch.from_numpy(np.load(os.path.join(self.file_dir, feature_group[-1])))
		

		if (self.print_files):
			print("Past: ")
			for i in range(self.num_past):
				print("\t" + feature_group[i])
			print("Future: ")
			print("\t" + feature_group[-1])		


		return past_features, future_features
