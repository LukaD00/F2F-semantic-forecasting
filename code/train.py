import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

from datasets.cityscapes_halfres_features_dataset import CityscapesHalfresFeaturesDataset

from models.convf2f.conv_f2f import ConvF2F
from models.dilatedf2f.dilated_f2f import DilatedF2F
from models.deformf2f.deform_f2f import DeformF2F

if __name__=="__main__":

	device = "cuda"
	print(f"Running on " + torch.cuda.get_device_name(0))
	
	# list of nets to train in the format of (net, name, load, last_epoch)
	# if load is true, weights will be loaded from filesystem  	
	nets = [
		#(ConvF2F(layers=5), "ConvF2F-5-24", False, 0),
		#(ConvF2F(layers=8), "ConvF2F-8-24", False, 0),
		#(DilatedF2F(layers=5), "DilatedF2F-5-24", False, 0),
		#(DilatedF2F(layers=8), 	"DilatedF2F-8-24", False, 0),
		#(DeformF2F(layers=5), "DeformF2F-5-24", False, 0),
		(DeformF2F(layers=8, num_past=3), "DeformF2F-8-3-24-3past", False, 0)		
	]

	for net, name, load, epochs in nets:

		trainset = CityscapesHalfresFeaturesDataset(train=True, num_past=3, future_distance=3, num_sequence=3, print_files=False)
		valset = CityscapesHalfresFeaturesDataset(train=False, num_past=3, future_distance=3, num_sequence=3, print_files=False)

		trainloader = torch.utils.data.DataLoader(trainset, batch_size=24, shuffle=True, num_workers=2)
		valloader = torch.utils.data.DataLoader(valset, batch_size=12, shuffle=True, num_workers=2)

		net = net.to(device)
		if load:
			net.load_state_dict(torch.load(f"../weights/{name}.pt"))

		criterion = nn.MSELoss()
		optimizer = optim.Adam(net.parameters(), lr=5e-4)
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

		print(f"\n\nStarted training {name}")
		start_time = time.time()

		best_val_loss = None
		for epoch in range(epochs+1, 160):
			print("\n\tEpoch: %d, Time elapsed: %.2f min, Datetime: %s" % (epoch, (time.time() - start_time)/60, datetime.today()))

			# TRAIN
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
			print("\t\tTrain -> Loss: %.3f" % (train_loss/len(trainloader)))

			# VALIDATE
			net.eval()
			val_loss = 0
			with torch.no_grad():
				for batch_idx, (inputs, targets) in enumerate(valloader):
					inputs, targets = inputs.to(device), targets.to(device)
					outputs = net(inputs)
					loss = criterion(outputs, targets)
					val_loss += loss.item()
			print("\t\tEval -> Loss: %.3f" % (val_loss/(len(valloader))))
			
			if (best_val_loss == None or val_loss < best_val_loss):
				best_val_loss = val_loss
				torch.save(net.state_dict(), f"../weights/{name}.pt")
				print(f"\t\tModel saved to ../weights/{name}.pt")
		
			scheduler.step()

		