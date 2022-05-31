import torch
from torch.nn import functional as F
import numpy as np
from PIL import Image
from models.sci import ScaleInvariantModel
from models.resnet.resnet_relu_noskip import resnet18
from models.model import F2F, Model
from models.deformf2f.deform_f2f import DeformF2F
from models.convf2f.conv_f2f import ConvF2F
from models.dilatedf2f.dilated_f2f import DilatedF2F
from colormap import colormap
import os

def load_features(feature_path : str) -> torch.Tensor:

	num_classes = 19
	resnet = resnet18(pretrained=False, efficient=False)
	segm_model = ScaleInvariantModel(resnet, num_classes)
	segm_model.eval()
	segm_model.load_state_dict(torch.load("../weights/r18_halfres_semseg.pt"))

	img = np.array(Image.open(feature_path), dtype=np.float64) 
	img = np.moveaxis(img, -1, 0)								# (H,W,C) -> (C,H,W)	|	(1024,2048,3) -> (3,1024,2048)
	img_tensor = torch.tensor(img, requires_grad=True)
	img = img_tensor.unsqueeze(0).float()								# (C,H,W) -> (B,C,H,W)	| 	(3,1024,2048) -> (1,3,1024,2048)
	img = F.interpolate(img, scale_factor=0.5, mode='bilinear') #  (1,3,1024,2048) -> (1,3,512,1024)

	# img normalization
	img = img/255
	mean = torch.tensor([0.485, 0.456, 0.406], requires_grad=False).view(1,3,1,1)
	std = torch.tensor([0.229, 0.224, 0.225], requires_grad=False).view(1,3,1,1)
	img = (img - mean) / std

	feats = segm_model.forward_down(img)

	# feature normalization
	mean = torch.tensor(np.load("../cityscapes_halfres_features_r18/mean.npy"), requires_grad=False).view(1, 128, 1, 1)
	std = torch.tensor(np.load("../cityscapes_halfres_features_r18/std.npy"), requires_grad=False).view(1, 128, 1, 1)
	std += 1e-6
	feats = (feats - mean) / std

	return feats, img_tensor

def generate_pixel_neighbourhood(pixel : tuple[int,int], size : int) -> list[tuple[int,int]]:
	pixels = []
	for i in range(-size, size+1):
		for j in range(-size, size+1):
			pixels.append((pixel[0]+i, pixel[1]+j))
	return pixels



if __name__=="__main__":

	midterm = True

	images = [
		("val", "frankfurt", "000000", 8451, 4, (564, 1644)),
		("val", "frankfurt", "000001", 32556, 4, (405, 2013)),
		("val", "frankfurt", "000001", 7285, 4, (599, 31)),
		("val", "frankfurt", "000000", 294, 4, (448, 1725))
	]

	models: list[Model] = [
		#F2F(ConvF2F(layers=8), "ConvF2F-8-3"),
		#F2F(DilatedF2F(layers=8), "DilatedF2F-8-3-24"),
		#F2F(DeformF2F(layers=8), "DeformF2F-8-3-24")

		F2F(ConvF2F(layers=8), "ConvF2F-8-M-24"),
		F2F(DilatedF2F(layers=8), "DilatedF2F-8-M"),
		F2F(DeformF2F(layers=8), "DeformF2F-8-M")
	]

	for folder, city, seq, tag, num_past, chosen_pixel in images:

		for model in models:

			city_seq = city + "_" + seq

			future_tag = str(tag).zfill(6)

			if midterm:
				save_path = f"../output/gradients/{city}_{seq}_{future_tag}-midterm/"
			else:
				save_path = f"../output/gradients/{city}_{seq}_{future_tag}/"
			if (not os.path.exists(save_path)): os.mkdir(save_path)

			save_path += f"{model.getName()}/"
			if (not os.path.exists(save_path)): os.mkdir(save_path)

			chosen_pixel_color = [0, 255, 0]
			gradient_pixel_color = [255, 0, 0]
			chosen_pixel_neighbourhood = generate_pixel_neighbourhood(chosen_pixel, 8)
			k = 3000  # how many pixels in X_t will be marked


			# LOAD FUTURE FEATURES
			future_feature_path = f"../cityscapes-original/{folder}/{city}/{city_seq}_{future_tag}_leftImg8bit.png"
			future_features, future_img_tensor = load_features(future_feature_path)

			# LOAD PAST FEATURES
			past_features_list = []
			past_tensors_list = []
			for i in range(1,num_past+1):
				if midterm:
					past_tag = str(tag-6-i*3).zfill(6)
				else:
					past_tag = str(tag-i*3).zfill(6)
				past_feature_path = f"../cityscapes-original/{folder}/{city}/{city_seq}_{past_tag}_leftImg8bit.png"
				past_feature, past_tensor = load_features(past_feature_path)
				past_features_list.append(past_feature.squeeze(0))
				past_tensors_list.append(past_tensor)
			past_features_list = past_features_list[::-1]
			past_tensors_list = past_tensors_list[::-1]
			past_features = torch.vstack(past_features_list)

	
			# PREDICT NEW FEATURES
			logits = model.forecastLogits(past_features, future_features)
			preds = torch.argmax(logits, 1).squeeze()
			preds_colored = colormap[preds]
			for x,y in chosen_pixel_neighbourhood:
				preds_colored[x,y] = chosen_pixel_color
			image = Image.fromarray(preds_colored)
			image.save(save_path + f"{model.getName()}.png")
			print(f"{model.getName()} prediction saved to {save_path}")

			# ORIGINAL FUTURE IMAGE
			original_path = f"../cityscapes-original/{folder}/{city}/{city_seq}_{future_tag}_leftImg8bit.png"
			original = Image.open(original_path)
			for x,y in chosen_pixel_neighbourhood:
				original.putpixel((y,x), tuple(chosen_pixel_color))
			original.save(save_path + "Future.png")
			print(f"Original future image saved to {save_path}")


			# GRADIENTS
			logits = logits.squeeze()
			logits = logits.permute(1,2,0)
			chosen_logits = logits[chosen_pixel[0], chosen_pixel[1]]
			logmaxsoftmax = torch.log(torch.max(F.softmax(chosen_logits, dim=0))) # TODO: check dim
			logmaxsoftmax.backward() 

			past_last = past_tensors_list[-1]
			sorted, indices = torch.abs(torch.sum(past_last.grad, dim=0)).flatten().sort()
			threshold = sorted[-(k-1)]

			for i in range(len(past_tensors_list)):
				
				past_tensor = past_tensors_list[i].clone().detach()
				past_tensor = past_tensor.permute(1,2,0)
				grads = torch.abs(torch.sum(past_tensors_list[i].grad, dim=0))

				pixels = grads>threshold
				indices = pixels.nonzero()
				for x,y in indices:
					for x1, y1 in generate_pixel_neighbourhood((x,y), 2):
						past_tensor[x1][y1] = torch.tensor(gradient_pixel_color).double()
				for x,y in chosen_pixel_neighbourhood:
					past_tensor[x][y] = torch.tensor(chosen_pixel_color).double()
				
				#past_tensor[grads>threshold] = torch.tensor(gradient_pixel_color).double()
				
				past_tensor_np = past_tensor.int().numpy()
				img = Image.fromarray(past_tensor_np.astype(np.uint8))
				img.save(save_path + f"Past{i}.png")
				print(f"Original past{i} image saved to {save_path}")