from abc import abstractmethod
import sys, os
import torch
import torch.nn as nn
import numpy as np
from models.sci import ScaleInvariantModel
from models.resnet.resnet_relu_noskip import resnet18
from models.convf2f.conv_f2f import ConvF2F
from models.dilatedf2f.dilated_f2f import DilatedF2F
from models.deformf2f.deform_f2f import DeformF2F
from datasets.cityscapes_halfres_ground_truth_dataset import CityscapesHalfresGroundTruthDataset
from torchmetrics import JaccardIndex


class Model():
	"""
	An abstract class that represents some model that takes past features and future features
	and gives a semantic segmentation prediction.

	Implementations of this abstract class should override the forecasting method.
	"""
	def __init__(self):
		self.num_classes = 19
		self.output_features_res = (128, 256)
		self.output_preds_res = (512, 1024)
		resnet = resnet18(pretrained=False, efficient=False)
		self.segm_model = ScaleInvariantModel(resnet, self.num_classes)
		self.segm_model.load_state_dict(torch.load("../weights/r18_halfres_semseg.pt"))
		self.segm_model.to("cuda")

		input_features = 128
		self.mean = torch.tensor(np.load("../cityscapes_halfres_features_r18/mean.npy"), requires_grad=False).view(1, input_features, 1, 1).to("cuda")
		self.std = torch.tensor(np.load("../cityscapes_halfres_features_r18/std.npy"), requires_grad=False).view(1, input_features, 1, 1).to("cuda")
	
	@abstractmethod
	def name(self) -> str:
		pass

	@abstractmethod
	def forecast(self, past_features : torch.Tensor, future_features : torch.Tensor) -> torch.tensor:
		pass

class Oracle(Model):
	"""
	An oracle model - it will take future_features and 
	simply do the upsampling path of a ResNet-18 network.
	This model should provide the upper bound for F2F mIoU.
	"""
	def name(self):
		return "Oracle"

	def forecast(self, past_features : torch.Tensor, future_features : torch.Tensor) -> torch.tensor:
		future_features_denormalized = future_features * self.std + self.mean
		logits, additional_dict = self.segm_model.forward_up(future_features_denormalized, self.output_features_res, self.output_preds_res)
		preds = torch.argmax(logits, 1).squeeze().cpu()
		return preds

class CopyLast(Model):
	"""
	A model which just copies the most recent past image's tensors.
	This model should provide the lower bound for F2F mIoU.
	"""
	def name(self):
		return "Copy-Last"

	def forecast(self, past_features : torch.Tensor, future_features : torch.Tensor) -> torch.tensor:
		last = past_features[(512-128):512] # TODO: potentially look into not hardcoding these values?
		last_denormalized = last  * self.std + self.mean
		logits, additional_dict = self.segm_model.forward_up(last_denormalized, self.output_features_res, self.output_preds_res)
		preds = torch.argmax(logits, 1).squeeze().cpu()
		return preds

class F2F(Model):
	"""
	A general abstract ResNet-18 F2F model - it will take past_features,
	make an F2F forecasting, and do the upsampling path of a ResNet-18 network.
	"""

	def __init__(self, f2f : nn.Module, name : str):
		"""
		Arguments:
			f2f (nn.Module) - Initialized F2F net
			name (str) - State dict will be loaded from "../weights/{name}.pt, this name will also be used in print statements. W
							Won't try to load state dict if name is None.
		"""
		self.f2f = f2f.to("cuda")
		self.f2f.eval()
		if (name != None) :
			self.f2f.load_state_dict(torch.load(f"../weights/{name}.pt"))

		self.name = name

	def forecast(self, past_features : torch.Tensor, future_features : torch.Tensor) -> torch.tensor:
		predicted_future_features = self.f2f.forward(past_features.unsqueeze(0))
		predicted_future_features_denormalized = predicted_future_features  * self.std + self.mean
		logits, additional_dict = self.segm_model.forward_up(predicted_future_features_denormalized, self.output_features_res, self.output_preds_res)
		preds = torch.argmax(logits, 1).squeeze().cpu()
		return preds

	def name(self) -> str:
		return self.name

def test(net : nn.Module) -> float:
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
	return testModel(f2f)

def testModel(model : Model) -> float:
	"""
	Tests the given Model on Cityscapes Dataset.

	Arguments:
		model (Model) - forecasting net wrapped in Model object

	Returns:
		mIoU achieved by model on dataset
	"""
	miou = JaccardIndex(num_classes=20, ignore_index=19)
	for past_features, future_features, ground_truth in CityscapesHalfresGroundTruthDataset():
		past_features, future_features = past_features.to("cuda"), future_features.to("cuda")
		prediction = model.forecast(past_features, future_features)
		ground_truth[ground_truth==255] = 19
		miou.update(prediction, torch.from_numpy(ground_truth))
	return miou.compute()

if __name__ == '__main__':
	dataset = CityscapesHalfresGroundTruthDataset(num_past=4)
	print(f"Dataset contains {len(dataset)} items")

	sys.stdout = open(os.devnull, 'w') # disable printing
	models: list[Model] = [Oracle()]
	sys.stdout = sys.__stdout__ # enable printing

	for model in models:
		print(f"Testing {model.name()}...")
		print(f"\tmIoU: {testModel(model)}")
		print()



