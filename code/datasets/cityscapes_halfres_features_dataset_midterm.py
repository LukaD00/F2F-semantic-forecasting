import os
import torch
import numpy as np
from torch.utils.data import Dataset


class CityscapesHalfresFeaturesDatasetMidterm(Dataset):
	"""
	A Dataset object for the features of the Cityscapes dataset with half-resolution images.

	Groups together past feature tensors with expected future tensors.
	"""
	
	TRAIN_DIR = "../cityscapes_halfres_features_r18/train"
	VAL_DIR = "../cityscapes_halfres_features_r18/val"

	def __init__(self, train = True, num_past = 4):
		"""
		Initializes the dataset by loading the file names and grouping them appropriately 
		(num_past past images with one corresponding future image).

		Args:
			train (bool): If true, training dataset will be loaded. If false, validation dataset will be loaded.
			num_past (int): How many past sets of features (images) will be provided for every set of features.
		"""
		super().__init__()
		
		self.num_past = num_past

		self.file_dir = CityscapesHalfresFeaturesDatasetMidterm.TRAIN_DIR if train else CityscapesHalfresFeaturesDatasetMidterm.VAL_DIR
		
		self.feature_groups = [] # array that stores groups of (past tensors, future tensor)
		files = os.listdir(self.file_dir)    
		for future_features in files:
			feature_group = [future_features]
			future_features_split = future_features.split("_") # file name should be in the format "city_sequence_time_leftImg8bit.npy"
			found_all = True
			for i in range(0, num_past):
				expected_past_features_split = future_features_split[:]
				expected_past_features_split[2] = str(int(expected_past_features_split[2]) - 9 - 3 * i).zfill(6) # replace "time" with "time-3*i"
				expected_past_features = "_".join(expected_past_features_split)
				if expected_past_features not in files:
					found_all = False # only group if all 4 past tensors are present
				feature_group.append(expected_past_features)
			if not found_all:
				continue
			self.feature_groups.append(feature_group[::-1])


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
		
		return past_features, future_features