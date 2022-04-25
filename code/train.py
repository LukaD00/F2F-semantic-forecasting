import numpy as np
import time

import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

from datasets.cityscapes_halfres_features_dataset import CityscapesHalfresFeaturesDataset
from models.convf2f.conv_f2f import ConvF2F


if __name__=="__main__":

	print("==> Preparing data...")

	mean = np.load("../cityscapes_halfres_features_r18/mean.npy")
	std = np.load("../cityscapes_halfres_features_r18/std.npy")

	trainset = CityscapesHalfresFeaturesDataset(train=True)
	valset = CityscapesHalfresFeaturesDataset(train=False)

	trainloader = torch.utils.data.DataLoader(trainset, batch_size=24, shuffle=True, num_workers=2)
	valloader = torch.utils.data.DataLoader(valset, batch_size=20, shuffle=True, num_workers=2)


	print('==> Building model...')
	device = "cuda"
	net = ConvF2F().to(device)

	criterion = nn.MSELoss()
	optimizer = optim.Adam(net.parameters(), lr=5e-4)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

	start_time = time.time()

	for epoch in range(30):
		print("\nEpoch: %d" % epoch)
		print("Time elapsed: %.2f min" % ((time.time() - start_time)/60))  
		
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
		print("Train -> Loss: %.3f" % (train_loss/(len(trainloader))))

		net.eval()
		test_loss = 0
		with torch.no_grad():
			for batch_idx, (inputs, targets) in enumerate(valloader):
				inputs, targets = inputs.to(device), targets.to(device)
				outputs = net(inputs)
				loss = criterion(outputs, targets)
				test_loss += loss.item()
		print("Eval -> Loss: %.3f" % (test_loss/(len(valloader))))
	
		scheduler.step()

	torch.save(net.state_dict(), "../weights/conv-f2f.pt")