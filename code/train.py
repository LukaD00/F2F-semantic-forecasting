import numpy as np

import torch
import torchvision.transforms as transforms

from datasets.cityscapes_halfres_features_dataset import CityscapesHalfresFeaturesDataset
from models.convf2f.conv_f2f import ConvF2F


if __name__=="__main__":

	print("==> Preparing data")
	
	input_features = 128

	mean = np.load("../cityscapes_halfres_features_r18/mean.npy")
	std = np.load("../cityscapes_halfres_features_r18/std.npy")
	normalize = transforms.Normalize(mean, std)

	trainset = CityscapesHalfresFeaturesDataset(train=True, transform=normalize, target_transform=normalize)
	valset = CityscapesHalfresFeaturesDataset(train=False, transform=normalize, target_transform=normalize)

	trainloader = torch.utils.data.DataLoader(trainset, batch_size=12, shuffle=True, num_workers=4)
	valloader = torch.utils.data.DataLoader(trainset, batch_size=20, shuffle=True, num_workers=4)


	