from PIL import Image
import torch
import numpy as np
import os

from models.convf2f.conv_f2f import ConvF2F
from models.dilatedf2f.dilated_f2f import DilatedF2F
from models.deformf2f.deform_f2f import DeformF2F
from models.model import Model, Oracle, CopyLast, F2F
from colormap import colormap

if __name__ == '__main__':

	midterm = True

	images = [
		("val", "frankfurt", "000000", 294, 4),
		("val", "frankfurt", "000000", 10351, 4),
		("val", "frankfurt", "000001", 25921, 4),
		("val", "frankfurt", "000001", 25713, 4)
	]
	
	models: list[Model] = [
		#Oracle(),
		#F2F(ConvF2F(layers=8), "ConvF2F-8-3"),
		#F2F(DilatedF2F(layers=8), "DilatedF2F-8-3-24"),
		#F2F(DeformF2F(layers=8), "DeformF2F-8-3-24")

		Oracle(),
		F2F(ConvF2F(layers=8), "ConvF2F-8-M-24"),
		F2F(DilatedF2F(layers=8), "DilatedF2F-8-M"),
		F2F(DeformF2F(layers=8), "DeformF2F-8-M")
	]

	for folder, city, seq, tag, num_past in images:
		city_seq = city + "_" + seq

		future_tag = str(tag).zfill(6)
		if midterm:
			save_path = f"../output/segmentations/{city}_{seq}_{future_tag}-midterm/"
		else:
			save_path = f"../output/segmentations/{city}_{seq}_{future_tag}/"
		if (not os.path.exists(save_path)): os.mkdir(save_path)

		# LOAD FUTURE FEATURES
		future_feature_path = f"../cityscapes_halfres_features_r18/{folder}/{city_seq}_{future_tag}_leftImg8bit.npy"
		future_features = torch.from_numpy(np.load(future_feature_path))

		# LOAD PAST FEATURES
		past_features = []
		for i in range(1,num_past+1):
			if midterm:
				past_tag = str(tag-6-i*3).zfill(6)
			else:
				past_tag = str(tag-i*3).zfill(6)
			past_feature_path = f"../cityscapes_halfres_features_r18/{folder}/{city_seq}_{past_tag}_leftImg8bit.npy"
			past_feature = np.load(past_feature_path)
			past_features.append(past_feature)
		past_features = torch.from_numpy(np.vstack(past_features[::-1]))

		# (FUTURE_FEATURES, PAST_FEATURES) => MODEL PREDICTIONS
		for model in models:
			preds = model.forecast(past_features, future_features)
			preds_colored = colormap[preds]
			image = Image.fromarray(preds_colored)
			image.save(save_path + f"{model.getName()}.png")
			print(f"{model.getName()} prediction saved to {save_path}")

		# FUTURE GROUND TRUTH LABELING
		future_gt_path = f"../cityscapes-gt/{folder}/{city_seq}_{future_tag}_gtFine_labelTrainIds.png"
		ground_truth = np.array(Image.open(future_gt_path))
		ground_truth_colored = colormap[ground_truth]
		image = Image.fromarray(ground_truth_colored)
		image.save(save_path + "GroundTruth.png")
		print(f"Ground truth saved to {save_path}")
		
		# ORIGINAL FUTURE IMAGE
		original_path = f"../cityscapes-original/{folder}/{city}/{city_seq}_{future_tag}_leftImg8bit.png"
		original = Image.open(original_path)
		original.save(save_path + "Future.png")
		print(f"Original future image saved to {save_path}")

		# ORIGINAL PAST IMAGES
		for i in range(1,num_past+1):
			if midterm:
				past_tag = str(tag-6-i*3).zfill(6)
			else:
				past_tag = str(tag-i*3).zfill(6)
			past_original_path = f"../cityscapes-original/{folder}/{city}/{city_seq}_{past_tag}_leftImg8bit.png"
			past_original = Image.open(past_original_path)
			past_original.save(save_path + f"Past{i}.png")
			print(f"Original past{i} image saved to {save_path}")
	

	

	

	