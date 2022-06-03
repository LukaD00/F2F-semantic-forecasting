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



def get_mious(model : Model, dataset : CityscapesHalfresGroundTruthDataset) -> None :
	"""
	Tests the given model on CityscapesHalfresGroundTruthDataset.

	Args:
		model (Model) - A full segmentation forecasting model wrapped in a Model object
		dataset (CityscapesHalfresGroundTruthDataset) - A dataset to test the model on

	"""
	ignore_index = 19
	with torch.no_grad():
		miouConfMat = JaccardIndex(num_classes=20, ignore_index=ignore_index)
		for past_features, future_features, ground_truth in dataset:
			past_features, future_features = past_features.to("cuda"), future_features.to("cuda")
			prediction = model.forecast(past_features, future_features)
			ground_truth[ground_truth==255] = 19
			miouConfMat.update(prediction, torch.from_numpy(ground_truth))
			del past_features, future_features, ground_truth, prediction
	confmat = miouConfMat.confmat
	confmat[ignore_index] = 0.0
	intersection = torch.diag(confmat)
	union = confmat.sum(0) + confmat.sum(1) - intersection
	scores = intersection.float() / union.float()
	scores[union == 0] = 0
	
	miou = torch.mean(scores[0:19])
	miouMO = torch.mean(scores[11:19])

	return miou, miouMO

if __name__ == '__main__':

	""" baseline
	baseline = [
		Oracle(),
		CopyLast()
	]
	"""

	""" midterm
	models_midterm = [
		F2F(ConvF2F(layers=5), "ConvF2F-5-M"),
		F2F(ConvF2F(layers=8), "ConvF2F-8-M"),
		F2F(DilatedF2F(layers=5), "DilatedF2F-5-M"),
		F2F(DilatedF2F(layers=8), "DilatedF2F-8-M"),
		F2F(DeformF2F(layers=5), "DeformF2F-5-M"),
		F2F(DeformF2F(layers=8), "DeformF2F-8-M")
	]
	"""

	""" shortterm with one training example per sequence
	models_shortterm_1 = [
		F2F(ConvF2F(layers=5), "ConvF2F-5"),
		F2F(ConvF2F(layers=8), "ConvF2F-8"),
		F2F(DilatedF2F(layers=5), "DilatedF2F-5"),
		F2F(DilatedF2F(layers=8), "DilatedF2F-8"),
		F2F(DeformF2F(layers=5), "DeformF2F-5"),
		F2F(DeformF2F(layers=8), "DeformF2F-8")
	]
	"""

	""" shortterm with three training examples per sequence
	models_shortterm_3 = [
		F2F(ConvF2F(layers=5), "ConvF2F-5-3"),
		F2F(ConvF2F(layers=8), "ConvF2F-8-3"),
		F2F(DilatedF2F(layers=5), "DilatedF2F-5-3"),
		F2F(DilatedF2F(layers=8), "DilatedF2F-8-3"),
		F2F(DeformF2F(layers=5), "DeformF2F-5-3"),
		F2F(DeformF2F(layers=8), "DeformF2F-8-3")
	]
	"""

	""" shortterm with three training examples per sequence, and 24 batch size
	models_shortterm_3_24 = [
		F2F(ConvF2F(layers=8), "ConvF2F-8-3-24"),
		F2F(ConvF2F(layers=5), "ConvF2F-5-3-24"),
		F2F(DilatedF2F(layers=8), 	"DilatedF2F-8-3-24"),
		F2F(DilatedF2F(layers=5), "DilatedF2F-5-3-24"),
		F2F(DeformF2F(layers=8), "DeformF2F-8-3-24"),
		F2F(DeformF2F(layers=5), "DeformF2F-5-3-24")
	]
	"""

	""" shortterm, one training example per sequence, 24 batch size 
	models = [
		F2F(ConvF2F(layers=5), "ConvF2F-5-24"),
		F2F(ConvF2F(layers=8), "ConvF2F-8-24"),
		F2F(DilatedF2F(layers=5), "DilatedF2F-5-24"),
		F2F(DilatedF2F(layers=8), "DilatedF2F-8-24"),
		F2F(DeformF2F(layers=5), "DeformF2F-5-24"),
		F2F(DeformF2F(layers=8), "DeformF2F-8-24")
	]
	"""

	models = [
		(F2F(DeformF2F(layers=8, num_past=2), "DeformF2F-8-M-24-3past"), 
			CityscapesHalfresGroundTruthDataset(num_past=3, future_distance=9)),

		(F2F(DeformF2F(layers=8, num_past=2), "DeformF2F-8-M-24-3past"), 
			CityscapesHalfresGroundTruthDataset(num_past=3, future_distance=9)),

		(F2F(DeformF2F(layers=8, num_past=2), "DeformF2F-8-M-24-3past"), 
			CityscapesHalfresGroundTruthDataset(num_past=3, future_distance=9))
	]

	for model, dataset in models:

		if dataset == None:
			dataset = CityscapesHalfresGroundTruthDataset(num_past=2, future_distance=9)

		print(f"Testing {model.getName()}...")
		miou, miouMO = get_mious(model, dataset)
		print(f"\tmIoU: {miou}")
		print(f"\tmIoU - MO: {miouMO}")
		print()



