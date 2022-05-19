from abc import abstractmethod
import sys, os
import torch
import torch.nn as nn
from torchmetrics import JaccardIndex
import numpy as np

from models.convf2f.conv_f2f import ConvF2F
from models.dilatedf2f.dilated_f2f import DilatedF2F
from models.deformf2f.deform_f2f import DeformF2F
from models.model import Model, Oracle, CopyLast, F2F

from datasets.cityscapes_halfres_ground_truth_dataset import CityscapesHalfresGroundTruthDataset


def miou(net : nn.Module) -> float:
	"""
	Tests the given F2F net on Cityscapes Dataset with ResNet-18.

	Arguments:
		net (nn.Module) - pretrained model to test

	Returns:
		mIoU achieved by model on dataset
	"""
	sys.stdout = open(os.devnull, 'w') # disable printing
	f2f = F2F(net, None)
	sys.stdout = sys.__stdout__ # enable printing
	return miouModel(f2f)


def miouModel(model : Model) -> float:
	"""
	Tests the given Model on Cityscapes Dataset.

	Arguments:
		model (Model) - forecasting net wrapped in Model object

	Returns:
		mIoU achieved by model on dataset
	"""
	with torch.no_grad():
		miou = JaccardIndex(num_classes=20, ignore_index=19)
		for past_features, future_features, ground_truth in CityscapesHalfresGroundTruthDataset():
			past_features, future_features = past_features.to("cuda"), future_features.to("cuda")
			prediction = model.forecast(past_features, future_features)
			ground_truth[ground_truth==255] = 19
			miou.update(prediction, torch.from_numpy(ground_truth))
			del past_features, future_features, ground_truth, prediction
	return miou.compute().item()


if __name__ == '__main__':
	dataset = CityscapesHalfresGroundTruthDataset(num_past=4)

	models = [
		#F2F(DilatedF2F(layers=5), "DilatedF2F-5")
		F2F(DeformF2F(), "DeformF2F-8")
		#F2F(DeformF2F(layers=5), "DeformF2F-5")
		#F2F(ConvF2F(), "ConvF2F-8"),
		#CopyLast(),
		#Oracle()
	]

	for model in models:
		print(f"Testing {model.getName()}...")
		print(f"\tmIoU: {miouModel(model)}")
		print()



