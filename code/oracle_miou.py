import torch
import numpy as np
from torchmetrics import JaccardIndex
from datasets.cityscapes_halfres_ground_truth_dataset import CityscapesHalfresGroundTruthDataset
from models.sci import ScaleInvariantModel
from models.resnet.resnet_relu_noskip import resnet18

# Inicijalizacija modela za semanticku segmentaciju.
input_features = 128
num_classes = 19
output_features_res = (128, 256)
output_preds_res = (512, 1024)
resnet = resnet18(pretrained=False, efficient=False)
segm_model = ScaleInvariantModel(resnet, num_classes)
segm_model.load_state_dict(torch.load("../weights/r18_halfres_semseg.pt"))
segm_model.to("cuda")

mean = torch.tensor(np.load("../cityscapes_halfres_features_r18/mean.npy"), requires_grad=False).view(1, input_features, 1, 1).to("cuda")
std = torch.tensor(np.load("../cityscapes_halfres_features_r18/std.npy"), requires_grad=False).view(1, input_features, 1, 1).to("cuda")

# Torcheva implementacija mIoU-a
# Ignore_index mora biti manji od num_classes, pa sam stavio 19 umjesto 255 (i onda u svakoj ground truth slici zamijenim sve 255 s 19)
miou = JaccardIndex(num_classes=20, ignore_index=19).to("cuda")

# Moj Dataset objekt koji grupira i vraca odgovarajuce podatke.
# future_features ce biti (128, 16, 32) tensor ucitan iz npr. frankfurt_000000_000294_leftImg8bit.npy,
# te ce za njega vratiti odgovarajucu ground_truth sliku, (1024, 512) polje ucitanu iz npr frankfurt_000000_000294_gtFine_labelTrainIds.png
dataset = CityscapesHalfresGroundTruthDataset()

for _, future_features, ground_truth in dataset:

	# Predaja znacajki ResNet-u
	future_features = future_features.to("cuda")
	future_features = future_features.unsqueeze(0) * std + mean
	logits, additional_dict = segm_model.forward_up(future_features, output_features_res, output_preds_res)
	preds = torch.argmax(logits, 1).squeeze() # Predikcija

	ground_truth = torch.from_numpy(ground_truth).to("cuda")
	ground_truth[ground_truth==255] = 19 

	# Azuriranje matrice zabune
	miou.update(preds, ground_truth)

print(f"mIoU: {miou.compute()}") # 0.596687376499176