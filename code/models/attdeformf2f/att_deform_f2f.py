from numpy import vectorize
import torch
from torch import nn
from torchvision import ops

class AttDeformF2F(nn.Module):	

	def __init__(self, output_channels=128, num_past=4, layers=8):
		super(AttDeformF2F, self).__init__()
		
		self.conv1 = nn.Conv2d(in_channels=num_past*output_channels, out_channels=2*output_channels, kernel_size=1, padding=0)
		self.relu1 = nn.ReLU()

		self.conv2 = nn.Conv2d(in_channels=2*output_channels, out_channels=output_channels, kernel_size=3, padding=1)
		self.relu2 = nn.ReLU()

		self.mhas = nn.ModuleList()
		for i in range(layers-4):
			self.mhas.append(nn.MultiheadAttention(embed_dim=output_channels, num_heads=4, batch_first=True))

		self.conv3 = nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=3, padding=1)
		self.relu3 = nn.ReLU()

		self.conv4 = nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=3, padding=1)  

		self.reset_parameters()

	def reset_parameters(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
				nn.init.constant_(m.bias, 0)

	def vectorise(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Transforms a (N,C,H,W) tensor to a (N,H*W,C) tensor.
		"""
		n, c, _, _ = x.shape
		x = torch.permute(x, (0,2,3,1))
		x = x.view(n, -1, c)
		return x
	
	def devectorise(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
		"""
		Transforms a (N,H*W,C) tensor to a (N,C,H,W) tensor.
		"""
		n, _, c = x.shape
		x = x.view(n, h, w, c)
		x = torch.permute(x, (0,3,1,2))
		return x

	def forward(self, x):
		x = self.relu1.forward(self.conv1.forward(x))
		x = self.relu2.forward(self.conv2.forward(x))

		x = self.vectorise(x)
		for i in range(len(self.mhas)):
			x = self.mhas[i].forward(x,x,x,  need_weights=False)[0]
		x = self.devectorise(x,16,32)
			

		x = self.relu3.forward(self.conv3.forward(x))
		x = self.conv4.forward(x)

		return x