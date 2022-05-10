import time

import torch
import torch.nn as nn
import torch.optim as optim

from datasets.cityscapes_halfres_features_dataset import CityscapesHalfresFeaturesDataset

from models.convf2f.conv_f2f import ConvF2F
from models.dilatedconvf2f.dilated_conv_f2f import DilatedConvF2F
from models.deformconvf2f.deform_conv_f2f import DeformConvF2F
from test import test


if __name__=="__main__":

	trainset = CityscapesHalfresFeaturesDataset(train=True)
	valset = CityscapesHalfresFeaturesDataset(train=False)

	trainloader = torch.utils.data.DataLoader(trainset, batch_size=24, shuffle=True, num_workers=2)
	valloader = torch.utils.data.DataLoader(valset, batch_size=20, shuffle=True, num_workers=2)

	device = "cuda"
	#net = ConvF2F().to(device)
	#net = DilatedConvF2F().to(device)
	#net = DeformConvF2F(layers=5).to(device)
	
	# list of nets to train in the format of (net, name) 	
	nets: list[tuple[nn.Module, str]] = [
		(ConvF2F(layers=5), "ConvF2F-5"),
		(ConvF2F(), "ConvF2F-8"),
		(DeformConvF2F(layers=5), "DeformF2F-5"),
		(DeformConvF2F(), "DeformF2F-8")
	]

	for net, name in nets:

		net = net.to(device)
		criterion = nn.MSELoss()
		optimizer = optim.Adam(net.parameters(), lr=5e-4)
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

		print(f"Started training {name}")
		start_time = time.time()

		for epoch in range(160):
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

			if (epoch % 20 == 0):
				torch.save(net.state_dict(), f"../weights/{name}.pt")

				print("\nEpoch: %d, Time: %.2f min" % (epoch, (time.time() - start_time)/60))
				print("\tTrain -> Loss: %.3f" % (train_loss/len(trainloader)), ((time.time() - start_time)/60))

				net.eval()
				test_loss = 0
				with torch.no_grad():
					for batch_idx, (inputs, targets) in enumerate(valloader):
						inputs, targets = inputs.to(device), targets.to(device)
						outputs = net(inputs)
						loss = criterion(outputs, targets)
						test_loss += loss.item()
				print("\tEval -> Loss: %.3f" % (test_loss/(len(valloader))))
				print("\t     -> mIoU: %.3f" % test(net))
		
			scheduler.step()

		torch.save(net.state_dict(), f"../weights/{name}.pt")
		print(f"Model saved to ../weights/{name}.pt")