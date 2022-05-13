from abc import abstractmethod
import torch
import torch.nn as nn
import numpy as np

from models.sci import ScaleInvariantModel
from models.resnet.resnet_relu_noskip import resnet18

class Model():
	"""
	An abstract class that represents some model that takes past features and future features
	and gives a semantic segmentation prediction.

	Implementations of this abstract class should override the forecasting method.

	This abstract class is used for calculating mIoU for different nets.
	"""
	def __init__(self):
		self.num_classes = 19
		self.output_features_res = (256, 512)
		self.output_preds_res = (1024, 2048)
		resnet = resnet18(pretrained=False, efficient=False)
		self.segm_model = ScaleInvariantModel(resnet, self.num_classes)
		self.segm_model.eval()
		self.segm_model.load_state_dict(torch.load("../weights/r18_halfres_semseg.pt"))
		self.segm_model.to("cuda")

		input_features = 128
		self.mean = torch.tensor(np.load("../cityscapes_halfres_features_r18/mean.npy"), requires_grad=False).view(1, input_features, 1, 1).to("cuda")
		self.std = torch.tensor(np.load("../cityscapes_halfres_features_r18/std.npy"), requires_grad=False).view(1, input_features, 1, 1).to("cuda")
	
	@abstractmethod
	def getName(self) -> str:
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
	def getName(self):
		return "Oracle"

	def forecast(self, past_features : torch.Tensor, future_features : torch.Tensor) -> torch.tensor:
		past_features = past_features.to("cuda")
		future_features = future_features.to("cuda")
		future_features_denormalized = future_features * self.std + self.mean
		logits, additional_dict = self.segm_model.forward_up(future_features_denormalized, self.output_features_res, self.output_preds_res)
		preds = torch.argmax(logits, 1).squeeze().cpu()
		return preds

class CopyLast(Model):
	"""
	A model which just copies the most recent past image's tensors.
	This model should provide the lower bound for F2F mIoU.
	"""
	def getName(self):
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
		super(F2F, self).__init__()

		self.f2f = f2f.to("cuda")
		self.f2f.eval()
		if (name != None) :
			self.f2f.load_state_dict(torch.load(f"../weights/{name}.pt"))

		self.name = name

	def forecast(self, past_features : torch.Tensor, future_features : torch.Tensor) -> torch.tensor:
		past_features = past_features.to("cuda")
		future_features = future_features.to("cuda")
		predicted_future_features = self.f2f.forward(past_features.unsqueeze(0))
		predicted_future_features_denormalized = predicted_future_features  * self.std + self.mean
		logits, additional_dict = self.segm_model.forward_up(predicted_future_features_denormalized, self.output_features_res, self.output_preds_res)
		preds = torch.argmax(logits, 1).squeeze().cpu()
		return preds

	def getName(self) -> str:
		return self.name