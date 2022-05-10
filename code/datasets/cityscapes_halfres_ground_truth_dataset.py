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
	
	TENSOR_DIR = "../cityscapes_halfres_features_r18/val"
	TRUTH_DIR = "../cityscapes-gt/val"

	def __init__(self, num_past = 4, transform = None, target_transform = None):
		"""
		Initializes the dataset by loading the file names and grouping them appropriately 
		(num_past past tensors with one corresponding ground truth set of labels).

		Args:
			num_past (int): How many past sets of features (images) will be provided for every set of features.
			transform (Transform): Transform that will be applied to the past features tensor.
			target_transform (Transform): Transform that will be applied to the target future features tensor.
		"""
		super().__init__()
		
		self.num_past = num_past
		self.transform = transform
		self.target_transform = target_transform
		
		self.items = [] # array that stores groups of (past tensors, future tensor, ground truth labels)
		tensor_files = os.listdir(CityscapesHalfresGroundTruthDataset.TENSOR_DIR)
		truth_files = os.listdir(CityscapesHalfresGroundTruthDataset.TRUTH_DIR)   
		for ground_truth in truth_files:
			item = [ground_truth]
			ground_truth_split = ground_truth.split("_") # file name should be in the format "city_sequence_time_gtFine_labelTrainIds.png"
			found_all = True
			for i in range(0, num_past+1):
				# feature file name should be in the format "city_sequence_time_leftImg8bit.npy"
				# i = 0 for future tensor, otherwise past
				expected_past_features_split = ground_truth_split[:]
				expected_past_features_split.pop() # remove "labelTrainIds.png"
				expected_past_features_split[3] = "leftImg8bit.npy" # replace "gtFine" with "leftImg8Bit.npy"
				expected_past_features_split[2] = str(int(expected_past_features_split[2]) - 3 * i).zfill(6) # replace "time" with "time-3*i"
				expected_past_features = "_".join(expected_past_features_split)
				if expected_past_features not in tensor_files:
					found_all = False # only group if all 4 past tensors and future tensor are present
				item.append(expected_past_features)
			if not found_all:
				continue
			self.items.append(item[::-1])


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
		past_features = torch.from_numpy(np.vstack([np.load(os.path.join(CityscapesHalfresGroundTruthDataset.TENSOR_DIR, img)) for img in item[0:self.num_past]]))
		future_features = future_features = torch.from_numpy(np.load(os.path.join(CityscapesHalfresGroundTruthDataset.TENSOR_DIR, item[-2])))
		ground_truth = np.array(Image.open(os.path.join(CityscapesHalfresGroundTruthDataset.TRUTH_DIR, item[-1])).resize((1024, 512), Image.NEAREST))

		# TODO: delete comments, used for debugging for now
		#print(f"Past features loaded from {item[0]}, {item[1]}, {item[2]}, {item[3]}")
		#print(f"Future features loaded from {item[-2]}")
		#print(f"Ground truth loaded from {item[-1]}")
		#print(f"Features: {item[-2]}, GT: {item[-1]}")

		if self.transform: 
			past_features = self.transform(past_features)

		if self.target_transform:
			future_features = self.target_transform(future_features)
		
		return past_features, future_features, ground_truth