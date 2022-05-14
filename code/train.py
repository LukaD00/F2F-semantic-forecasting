import time

import torch
import torch.nn as nn
import torch.optim as optim

from datasets.cityscapes_halfres_features_dataset import CityscapesHalfresFeaturesDataset

from models.convf2f.conv_f2f import ConvF2F
from models.dilatedf2f.dilated_f2f import DilatedF2F
from models.deformf2f.deform_f2f import DeformF2F
from miou import miou


if __name__=="__main__":

	device = "cuda"
	
	# list of nets to train in the format of (net, name, load)
	# if load is true, weights will be loaded from filesystem  	
	nets = [
		(DeformF2F(layers=8),	"DeformF2F-8", True),
		(DilatedF2F(layers=5), 	"DilatedF2F-5", False),
		(DilatedF2F(layers=8),	"DilatedF2F-8", False),
		(DeformF2F(layers=5), 	"DeformF2F-5", False),
		(ConvF2F(layers=8),		"ConvF2F-8", True),
		(ConvF2F(layers=5), 	"ConvF2F-5", True)
	]

	for net, name, load in nets:

		trainset = CityscapesHalfresFeaturesDataset(train=True)
		valset = CityscapesHalfresFeaturesDataset(train=False)

		trainloader = torch.utils.data.DataLoader(trainset, batch_size=24, shuffle=True, num_workers=2)
		valloader = torch.utils.data.DataLoader(valset, batch_size=20, shuffle=True, num_workers=2)

		net = net.to(device)
		if load:
			net.load_state_dict(torch.load(f"../weights/{name}.pt"))

		criterion = nn.MSELoss()
		optimizer = optim.Adam(net.parameters(), lr=5e-4)
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

		print(f"\n\nStarted training {name}")
		start_time = time.time()

		num_epochs = 160
		best_miou = 0
		for epoch in range(num_epochs):
			
			net.train()
			train_loss = 0
			for batch_idx, (inputs, targets) in enumerate(trainloader):
				inputs, targets = inputs.to(device), targets.to(device)
				optimizer.zero_grad()
				outputs = net(inputs)
				loss = criterion(outputs, targets)
				loss.backward()
				optimizer.step()
				
				train_loss += loss.item()

			if (epoch % 20 == 0 or epoch == num_epochs-1):
				
				print("\n\tEpoch: %d, Time: %.2f min" % (epoch, (time.time() - start_time)/60))
				print("\t\tTrain -> Loss: %.3f" % (train_loss/len(trainloader)))

				net.eval()
				test_loss = 0
				with torch.no_grad():
					for batch_idx, (inputs, targets) in enumerate(valloader):
						inputs, targets = inputs.to(device), targets.to(device)
						outputs = net(inputs)
						loss = criterion(outputs, targets)
						test_loss += loss.item()
				print("\t\tEval -> Loss: %.3f" % (test_loss/(len(valloader))))
				
				current_miou = miou(net)
				if (current_miou > best_miou):
					best_miou = current_miou
					torch.save(net.state_dict(), f"../weights/{name}.pt")
					print("\t\t     -> mIoU: %.3f (saved .pth)" % current_miou)
				else:
					print("\t\t     -> mIoU: %.3f" % current_miou)
		
			scheduler.step()

		print(f"\t\tModel saved to ../weights/{name}.pt")
		print("\t\tFinal mIoU: %.3f" % best_miou)