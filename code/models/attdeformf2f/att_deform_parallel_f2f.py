import torch
from torch import nn
from torchvision import ops

class AttDeformParallelF2F(nn.Module):	

	def __init__(self, output_channels=128, num_past=4, layers=8, height=16, width=32):
		super(AttDeformParallelF2F, self).__init__()
		
		self.conv1 = nn.Conv2d(in_channels=num_past*output_channels, out_channels=2*output_channels, kernel_size=1, padding=0)
		self.relu1 = nn.ReLU()

		self.conv2 = nn.Conv2d(in_channels=2*output_channels, out_channels=output_channels, kernel_size=3, padding=1)
		self.relu2 = nn.ReLU()

		self.deform_convs = nn.ModuleList()
		self.offsets = nn.ModuleList()
		self.relus = nn.ModuleList()
		for i in range(layers-4):
			deform_kernel_size = 3
			deform_padding = 1
			self.deform_convs.append(ops.DeformConv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=deform_kernel_size, padding=deform_padding))
			self.offsets.append(nn.Conv2d(in_channels=output_channels, out_channels=2*deform_kernel_size*deform_kernel_size, kernel_size=deform_kernel_size, padding=deform_padding))
			self.relus.append(nn.ReLU())

		self.conv3 = nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=3, padding=1)
		self.relu3 = nn.ReLU()

		self.conv4 = nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=3, padding=1)  


		self.pre_mha_conv = nn.Conv2d(in_channels=num_past*output_channels, out_channels=output_channels, kernel_size=1)

		self.pos_embedding = nn.Parameter(torch.randn(1, width*height, output_channels))
		
		self.mhas = nn.ModuleList()
		for i in range(layers):
			self.mhas.append(nn.MultiheadAttention(embed_dim=output_channels, num_heads=4, batch_first=True))

		self.reset_parameters()

	def reset_parameters(self):
		for m in [self.conv1, self.conv2, self.conv3, self.conv4, *self.deform_convs]:
			nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
			nn.init.constant_(m.bias, 0)
		for m in self.offsets:
			nn.init.constant_(m.weight, 0)
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

	def forward_conv(self, x):
		x = self.relu1.forward(self.conv1.forward(x))
		x = self.relu2.forward(self.conv2.forward(x))

		for i in range(len(self.deform_convs)):
			offsets = self.offsets[i].forward(x)
			x = self.deform_convs[i].forward(x, offsets)
			x = self.relus[i].forward(x)

		x = self.relu3.forward(self.conv3.forward(x))
		x = self.conv4.forward(x)

		return x

	def forward_att(self, x):
		n, c, h, w = x.shape
		x = self.pre_mha_conv(x)
		x = self.vectorise(x)
		x += self.pos_embedding[:, :(h*w)]
		for i in range(len(self.mhas)):
			x = self.mhas[i].forward(x,x,x,  need_weights=False)[0]
		x = self.devectorise(x, h, w)
		return x

	def forward(self, x):
		return self.forward_conv(x) + self.forward_att(x)