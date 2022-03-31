import os
import torch
import numpy as np
from torch.utils.data import Dataset


class CityscapesHalfresFeaturesDataset(Dataset):
	"""
	A Dataset object for the features of the Cityscapes dataset with half-resolution images.

	Training dataset is expected to be located in \"./train\", and validation dataset 
	is expected to be located in \"./val\".

	Feature file names are expected to be of the form "city_sequence_time_*.npy".
	"""
	
	TRAIN_DIR = "../cityscapes_halfres_features_r18/train"
	VAL_DIR = "../cityscapes_halfres_features_r18/val"

	def __init__(self, train = True, num_past = 4, transform = None, target_transform = None):
		"""
		Initializes the dataset by loading the file names and grouping them appropriately 
		(num_past past images with one corresponding future image).

		Args:
			train (bool): If true, training dataset will be loaded. If false, validation dataset will be loaded.
			num_past (int): How many past sets of features (images) will be provided for every set of features.
			transform (Transform): Transform that will be applied to the past features tensor.
			target_transform (Transform): Transform that will be applied to the target future features tensor.
		"""
		super().__init__()
		
		self.num_past = num_past
		self.transform = transform
		self.target_transform = target_transform

		self.file_dir = CityscapesHalfresFeaturesDataset.TRAIN_DIR if train else CityscapesHalfresFeaturesDataset.VAL_DIR
		
		self.feature_groups = []
		files = os.listdir(self.file_dir)    
		for future_features in files:
			feature_group = [future_features]
			future_features_split = future_features.split("_")
			found_all = True
			for i in range(1, num_past+1):
				expected_past_features_split = future_features_split[:]
				expected_past_features_split[2] = str(int(expected_past_features_split[2]) - 3 * i).zfill(6)
				expected_past_features = "_".join(expected_past_features_split)
				if expected_past_features not in files:
					found_all = False
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
		Returns the idx-th item in the dataset.

		Args:
			idx (int): Index of the wanted item.

		Returns:
			past_features : (512, 16, 32) torch.Tensor
			future_features : (128, 16, 32) torch.Tensor
		"""
		feature_group = self.feature_groups[idx]
		past_features = torch.from_numpy(np.vstack([np.load(os.path.join(self.file_dir, img)) for img in feature_group[0:self.num_past]]))
		future_features = torch.from_numpy(np.load(os.path.join(self.file_dir, feature_group[-1])))

		if self.transform: 
			past_features = self.transform(past_features)

		if self.target_transform:
			future_features = self.target_transform(future_features)
		
		return past_features, future_features