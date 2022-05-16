from PIL import Image
import torch
import numpy as np

from models.convf2f.conv_f2f import ConvF2F
from models.dilatedf2f.dilated_f2f import DilatedF2F
from models.deformf2f.deform_f2f import DeformF2F
from models.model import Model, Oracle, CopyLast, F2F

def create_cityscapes_label_colormap():
	"""Creates a label colormap used in CITYSCAPES segmentation benchmark.
	Returns:
	A colormap for visualizing segmentation results.
	"""
	colormap = np.zeros((256, 3), dtype=np.uint8)
	colormap[0] = [128, 64, 128]
	colormap[1] = [244, 35, 232]
	colormap[2] = [70, 70, 70]
	colormap[3] = [102, 102, 156]
	colormap[4] = [190, 153, 153]
	colormap[5] = [153, 153, 153]
	colormap[6] = [250, 170, 30]
	colormap[7] = [220, 220, 0]
	colormap[8] = [107, 142, 35]
	colormap[9] = [152, 251, 152]
	colormap[10] = [70, 130, 180]
	colormap[11] = [220, 20, 60]
	colormap[12] = [255, 0, 0]
	colormap[13] = [0, 0, 142]
	colormap[14] = [0, 0, 70]
	colormap[15] = [0, 60, 100]
	colormap[16] = [0, 80, 100]
	colormap[17] = [0, 0, 230]
	colormap[18] = [119, 11, 32]
	return colormap 

if __name__ == '__main__':

	folder = "val"
	city = "frankfurt"
	seq = "000001"
	city_seq = city + "_" + seq
	tag = 34047

	past0_path = f"../cityscapes_halfres_features_r18/{folder}/{city_seq}_{str(tag-12).zfill(6)}_leftImg8bit.npy"
	past1_path = f"../cityscapes_halfres_features_r18/{folder}/{city_seq}_{str(tag-9).zfill(6)}_leftImg8bit.npy"
	past2_path = f"../cityscapes_halfres_features_r18/{folder}/{city_seq}_{str(tag-6).zfill(6)}_leftImg8bit.npy"
	past3_path = f"../cityscapes_halfres_features_r18/{folder}/{city_seq}_{str(tag-3).zfill(6)}_leftImg8bit.npy"
	future_path = f"../cityscapes_halfres_features_r18/{folder}/{city_seq}_{str(tag).zfill(6)}_leftImg8bit.npy"
	future_gt_path = f"../cityscapes-gt/{folder}/{city_seq}_{str(tag).zfill(6)}_gtFine_labelTrainIds.png"
	original_path = f"../cityscapes-original/{folder}/{city}/{city_seq}_{str(tag).zfill(6)}_leftImg8bit.png"

	past0 = np.load(past0_path)
	past1 = np.load(past1_path)
	past2 = np.load(past2_path)
	past3 = np.load(past3_path)
	past_features = torch.from_numpy(np.vstack([past0, past1, past2, past3]))
	future_features = torch.from_numpy(np.load(future_path))

	colormap = create_cityscapes_label_colormap()

	models: list[Model] = [
		#Oracle(),
		#F2F(ConvF2F(layers=8), "ConvF2F-8"),
		#F2F(DilatedF2F(layers=8), "DilatedF2F-8"),
		F2F(DeformF2F(layers=8), "DeformF2F-8")
	]

	for model in models:
		preds = model.forecast(past_features, future_features)
		preds_colored = colormap[preds]
		image = Image.fromarray(preds_colored)
		path = f"../output/{city_seq}_{str(tag).zfill(6)}_{model.getName()}.png"
		image.save(path)
		print(f"{model.getName()} prediction saved to {path}")

	ground_truth = np.array(Image.open(future_gt_path))
	ground_truth_colored = colormap[ground_truth]
	image = Image.fromarray(ground_truth_colored)
	path = f"../output/{city_seq}_{str(tag).zfill(6)}_gt.png"
	image.save(path)
	print(f"Ground truth saved to {path}")

	original = Image.open(original_path)
	path = f"../output/{city_seq}_{str(tag).zfill(6)}_original.png"
	original.save(path)
	print(f"Original image saved to {path}")