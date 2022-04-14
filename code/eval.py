from abc import abstractmethod
import sys, os
import torch
import numpy as np
from models.sci import ScaleInvariantModel
from models.resnet.resnet_relu_noskip import resnet18
from models.convf2f.conv_f2f import ConvF2F
from datasets.cityscapes_halfres_ground_truth_dataset import CityscapesHalfresGroundTruthDataset
from util import mIoU


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
	def forecast(self, past_features : torch.Tensor, future_features : torch.Tensor) -> np.ndarray:
		pass

class Oracle(Model):
	"""
	An oracle model - it will take future_features and 
	simply do the upsampling path of a ResNet-18 network.
	"""
	def name(self):
		return "Oracle"

	def forecast(self, past_features : torch.Tensor, future_features : torch.Tensor) -> np.ndarray:
		future_features_normalized = future_features  * self.std + self.mean
		logits, additional_dict = self.segm_model.forward_up(future_features_normalized, self.output_features_res, self.output_preds_res)
		preds = torch.argmax(logits, 1).squeeze().cpu().numpy()
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

	def forecast(self, past_features : torch.Tensor, future_features : torch.Tensor) -> np.ndarray:
		f2f_model = self.F2Fmodel()
		predicted_future_features = f2f_model.forward(past_features).unsqueeze(0)
		logits, additional_dict = self.segm_model.forward_up(predicted_future_features, self.output_features_res, self.output_preds_res)
		preds = torch.argmax(logits, 1).squeeze().cpu().numpy()
		return preds

class Conv_F2F(F2F):
	"""
	A simple convolutional F2F model.
	"""
	def name(self) -> str:
		return "Conv-F2F"

	def F2Fmodel(self) -> torch.nn.Module:
		model = ConvF2F().to("cuda")
		model.eval()
		model.load_state_dict(torch.load("../weights/conv-f2f.pt"))
		return model

if __name__ == '__main__':
	dataset = CityscapesHalfresGroundTruthDataset(num_past=4)
	print(f"Dataset contains {len(dataset)} items")

	sys.stdout = open(os.devnull, 'w') # disable printing
	models: list[Model] = [Oracle(), Conv_F2F()]
	sys.stdout = sys.__stdout__ # enable printing

	all_classes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
	moving_objects_classes = [0,1,2,4,5,7,8,10,11,13]

	for model in models:
		print(f"Evaluating {model.name()}...")

		total_miou = 0
		count = 0
		for past_features, future_features, ground_truth in dataset:
			past_features, future_features = past_features.to("cuda"), future_features.to("cuda")
			prediction = model.forecast(past_features, future_features)
			total_miou += mIoU(prediction, ground_truth, all_classes)
			count += 1
		miou = total_miou / count
		print("\tmIoU: %.3f" % miou) 

		total_miou = 0
		count = 0
		for past_features, future_features, ground_truth in dataset:
			past_features, future_features = past_features.to("cuda"), future_features.to("cuda")
			prediction = model.forecast(past_features, future_features)
			total_miou += mIoU(prediction, ground_truth, moving_objects_classes)
			count += 1
		miou = total_miou / count
		print("\tmIoU-MO: %.3f" % miou) 

	