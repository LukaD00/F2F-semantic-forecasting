import torch
import numpy as np
from models.sci import ScaleInvariantModel
from models.resnet.resnet_relu_noskip import resnet18
from PIL import Image

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
	input_features = 128
	num_classes = 19
	output_features_res = (128, 256)
	output_preds_res = (512, 1024)
	resnet = resnet18(pretrained=False, efficient=False)
	segm_model = ScaleInvariantModel(resnet, num_classes)
	segm_model.load_state_dict(torch.load("../weights/r18_halfres_semseg.pt"))

	f2f_model = torch.load("../weights/Conv-F2F.pt").to("cpu")

	mean = torch.tensor(np.load("../cityscapes_halfres_features_r18/mean.npy"), requires_grad=False).view(1, input_features, 1, 1)
	std = torch.tensor(np.load("../cityscapes_halfres_features_r18/std.npy"), requires_grad=False).view(1, input_features, 1, 1)

	future = torch.from_numpy(np.load("../cityscapes_halfres_features_r18/val/frankfurt_000000_000288_leftImg8bit.npy")).unsqueeze(0)
	future = future * std + mean
	logits, additional_dict = segm_model.forward_up(future, output_features_res, output_preds_res)
	preds = torch.argmax(logits, 1).squeeze().numpy()
	colormap = create_cityscapes_label_colormap()
	preds_colored = colormap[preds]
	Image.fromarray(preds_colored).save("../demo_out_oracle.png")
	print("Oracle prediction saved to ../demo_out_oracle.png")

	past0 = np.load("../cityscapes_halfres_features_r18/val/frankfurt_000000_000276_leftImg8bit.npy")
	past1 = np.load("../cityscapes_halfres_features_r18/val/frankfurt_000000_000279_leftImg8bit.npy")
	past2 = np.load("../cityscapes_halfres_features_r18/val/frankfurt_000000_000282_leftImg8bit.npy")
	past3 = np.load("../cityscapes_halfres_features_r18/val/frankfurt_000000_000285_leftImg8bit.npy")
	past_tensor = torch.from_numpy(np.vstack([past0, past1, past2, past3]))
	predicted_features = f2f_model.forward(past_tensor).unsqueeze(0)
	logits, additional_dict = segm_model.forward_up(predicted_features, output_features_res, output_preds_res)
	preds = torch.argmax(logits, 1).squeeze().numpy()
	colormap = create_cityscapes_label_colormap()
	preds_colored = colormap[preds]
	Image.fromarray(preds_colored).save("../demo_out_predicted.png")
	print("F2F prediction saved to ../demo_out_predicted.png")

	
	
	