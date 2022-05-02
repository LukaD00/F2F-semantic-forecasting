from abc import abstractmethod
import sys, os
import torch
import numpy as np
from models.sci import ScaleInvariantModel
from models.resnet.resnet_relu_noskip import resnet18
from models.convf2f.conv_f2f import ConvF2F
from models.dilatedconvf2f.dilated_conv_f2f import DilatedConvF2F
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
		resnet = resnet18(pretrained=True, efficient=False)
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
		future_features_normalized = future_features  * self.std + self.mean
		logits, additional_dict = self.segm_model.forward_up(future_features_normalized, self.output_features_res, self.output_preds_res)
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
		last_normalized = last  * self.std + self.mean
		logits, additional_dict = self.segm_model.forward_up(last_normalized, self.output_features_res, self.output_preds_res)
		preds = torch.argmax(logits, 1).squeeze().cpu()
		return preds

class F2F(Model):
	"""
	A general abstract ResNet-18 F2F model - it will take past_features,
	make an F2F forecasting, and do the upsampling path of a ResNet-18 network.

	Implementations of this model should just define which F2F model it uses.
	"""
	@abstractmethod
	def F2Fmodel(self) -> torch.nn.Module:
		pass

	def forecast(self, past_features : torch.Tensor, future_features : torch.Tensor) -> torch.tensor:
		f2f_model = self.F2Fmodel()
		predicted_future_features = f2f_model.forward(past_features).unsqueeze(0)
		predicted_future_features_normalized = predicted_future_features  * self.std + self.mean
		logits, additional_dict = self.segm_model.forward_up(predicted_future_features_normalized, self.output_features_res, self.output_preds_res)
		preds = torch.argmax(logits, 1).squeeze().cpu()
		return preds

class ConvF2F_8(F2F):
	"""
	A simple convolutional F2F model.
	"""
	def name(self) -> str:
		return "ConvF2F-8"

	def F2Fmodel(self) -> torch.nn.Module:
		model = ConvF2F().to("cuda")
		model.eval()
		model.load_state_dict(torch.load("../weights/ConvF2F-8.pt"))
		return model

class DilatedConvF2F_8(F2F):
	"""
	A simple convolutional F2F model.
	"""
	def name(self) -> str:
		return "DilatedConvF2F-8"

	def F2Fmodel(self) -> torch.nn.Module:
		model = DilatedConvF2F().to("cuda")
		model.eval()
		model.load_state_dict(torch.load("../weights/DilatedConvF2F-8.pt"))
		return model

if __name__ == '__main__':
	dataset = CityscapesHalfresGroundTruthDataset(num_past=4)
	print(f"Dataset contains {len(dataset)} items")

	sys.stdout = open(os.devnull, 'w') # disable printing
	#models: list[Model] = [Oracle(), CopyLast(), ConvF2F_8(), DilatedConvF2F_8()]
	models: list[Model] = [DilatedConvF2F_8()]
	sys.stdout = sys.__stdout__ # enable printing

	all_classes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
	moving_objects_classes = [0,1,2,4,5,7,8,10,11,13]

	for model in models:
		print(f"Testing {model.name()}...")

		miou = JaccardIndex(num_classes=20, ignore_index=19)
		for past_features, future_features, ground_truth in dataset:
			past_features, future_features = past_features.to("cuda"), future_features.to("cuda")
			prediction = model.forecast(past_features, future_features)
			ground_truth[ground_truth==255] = 19
			miou.update(prediction, torch.from_numpy(ground_truth))
		
		print(f"\tmIoU: {miou.compute()}")
		print()




	