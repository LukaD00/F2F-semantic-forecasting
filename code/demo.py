import torch
import numpy as np
from models.sci import ScaleInvariantModel
from models.resnet.resnet_relu_noskip import resnet18
from models.convf2f.conv_f2f import ConvF2F
from util import mIoU
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

	future_img_features = "../cityscapes_halfres_features_r18/val/frankfurt_000001_027325_leftImg8bit.npy"
	past0_features = "../cityscapes_halfres_features_r18/val/frankfurt_000001_027313_leftImg8bit.npy"
	past1_features = "../cityscapes_halfres_features_r18/val/frankfurt_000001_027316_leftImg8bit.npy"
	past2_features = "../cityscapes_halfres_features_r18/val/frankfurt_000001_027319_leftImg8bit.npy"
	past3_features = "../cityscapes_halfres_features_r18/val/frankfurt_000001_027322_leftImg8bit.npy"
	future_img_gt = "../cityscapes-gt/val/frankfurt_000001_027325_gtFine_labelTrainIds.png"

	input_features = 128
	num_classes = 19
	output_features_res = (128, 256)
	output_preds_res = (512, 1024)
	resnet = resnet18(pretrained=False, efficient=False)
	segm_model = ScaleInvariantModel(resnet, num_classes)
	segm_model.load_state_dict(torch.load("../weights/r18_halfres_semseg.pt"))
	f2f_model = ConvF2F()
	f2f_model.eval()
	f2f_model.load_state_dict(torch.load("../weights/conv-f2f.pt"))
	colormap = create_cityscapes_label_colormap()

	mean = torch.tensor(np.load("../cityscapes_halfres_features_r18/mean.npy"), requires_grad=False).view(1, input_features, 1, 1)
	std = torch.tensor(np.load("../cityscapes_halfres_features_r18/std.npy"), requires_grad=False).view(1, input_features, 1, 1)

	oracle = torch.from_numpy(np.load(future_img_features)).unsqueeze(0)
	oracle = oracle * std + mean
	oracle_logits, oracle_additional_dict = segm_model.forward_up(oracle, output_features_res, output_preds_res)
	oracle_preds = torch.argmax(oracle_logits, 1).squeeze().numpy()
	preds_colored = colormap[oracle_preds]
	Image.fromarray(preds_colored).save("../demo_out_oracle.png")
	print("Oracle prediction saved to ../demo_out_oracle.png")

	past0 = np.load(past0_features)
	past1 = np.load(past1_features)
	past2 = np.load(past2_features)
	past3 = np.load(past3_features)
	past_tensor = torch.from_numpy(np.vstack([past0, past1, past2, past3]))
	predicted_features = f2f_model.forward(past_tensor).unsqueeze(0)
	logits, additional_dict = segm_model.forward_up(predicted_features, output_features_res, output_preds_res)
	preds = torch.argmax(logits, 1).squeeze().numpy()
	preds_colored = colormap[preds]
	Image.fromarray(preds_colored).save("../demo_out_predicted.png")
	print("F2F prediction saved to ../demo_out_predicted.png")

	real_img_semantics = np.array(Image.open(future_img_gt).resize((1024, 512), Image.NEAREST))
	real_img_semantics_colored = colormap[real_img_semantics]
	Image.fromarray(real_img_semantics_colored).save("../demo_out_real.png")
	print("Real semantics saved to ../demo_out_real.png")

	print(mIoU(preds, real_img_semantics, [0,1,2,4,5,7,8,10,11,13]))