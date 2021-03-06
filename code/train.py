import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

from datasets.cityscapes_halfres_features_dataset import CityscapesHalfresFeaturesDataset

from models.convf2f.conv_f2f import ConvF2F
from models.dilatedf2f.dilated_f2f import DilatedF2F
from models.deformf2f.deform_f2f import DeformF2F
from models.deformf2f.full_deform_f2f import FullDeformF2F
from models.attdeformf2f.att_deform_f2f import AttDeformF2F
from models.attdeformf2f.att_deform_parallel_f2f import AttDeformParallelF2F

if __name__=="__main__":

	device = "cuda"
	print(f"Running on " + torch.cuda.get_device_name(0))
	
	# list of nets to train in the format of (net, name, load, last_epoch, trainset, valset)
	# name is used as name of weights file to be saved (or loaded)
	# if load is true, weights will be loaded from filesystem, and training will resume from last_epoch
	# if trainset and valset are None, a default set defined in the loop below will be used
	nets = [
		
		(DeformF2F(layers=8), "DeformF2F-8-M", False, 0,
			CityscapesHalfresFeaturesDataset(train=True, num_past=4, future_distance=9, num_sequence=1, print_files=False), 
			CityscapesHalfresFeaturesDataset(train=False, num_past=4, future_distance=9, num_sequence=1, print_files=False)),		
		
		(DeformF2F(layers=8), "DeformF2F-8", False, 0,
			CityscapesHalfresFeaturesDataset(train=True, num_past=4, future_distance=3, num_sequence=1, print_files=False), 
			CityscapesHalfresFeaturesDataset(train=False, num_past=4, future_distance=3, num_sequence=1, print_files=False)),		
	
		(DeformF2F(layers=8), "DeformF2F-8-3", False, 0,
			CityscapesHalfresFeaturesDataset(train=True, num_past=4, future_distance=3, num_sequence=3, print_files=False), 
			CityscapesHalfresFeaturesDataset(train=False, num_past=4, future_distance=3, num_sequence=3, print_files=False)),		
	
	]

	for net, name, load, epochs, trainset, valset in nets:

		batch_size = 24

		if trainset == None:
			trainset = CityscapesHalfresFeaturesDataset(train=True, num_past=4, future_distance=3, num_sequence=1, print_files=False)
		if valset == None:
			valset = CityscapesHalfresFeaturesDataset(train=False, num_past=4, future_distance=3, num_sequence=1, print_files=False)

		trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
		valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size//2, shuffle=True, num_workers=2)

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

		