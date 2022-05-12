from torch import nn
from torchvision import ops

class DeformF2F(nn.Module):	

	def __init__(self, output_channels=128, num_past=4, layers=8):
		super(DeformF2F, self).__init__()
		
		self.deform_convs = nn.ModuleList()
		self.offsets = nn.ModuleList()
		self.relus = nn.ModuleList()

		self.deform_convs.append(ops.DeformConv2d(in_channels=num_past*output_channels, out_channels=2*output_channels, kernel_size=1, padding=0))
		self.offsets.append(nn.Conv2d(in_channels=num_past*output_channels, out_channels=2*1*1, kernel_size=1, padding=0))
		self.relus.append(nn.ReLU())

		self.deform_convs.append(ops.DeformConv2d(in_channels=2*output_channels, out_channels=output_channels, kernel_size=3, padding=1))
		self.offsets.append(nn.Conv2d(in_channels=2*output_channels, out_channels=2*3*3, kernel_size=3, padding=1))
		self.relus.append(nn.ReLU())

		for i in range(layers-3):
			deform_kernel_size = 3
			deform_padding = 1
			self.deform_convs.append(ops.DeformConv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=deform_kernel_size, padding=deform_padding))
			self.offsets.append(nn.Conv2d(in_channels=output_channels, out_channels=2*deform_kernel_size*deform_kernel_size, kernel_size=deform_kernel_size, padding=deform_padding))
			self.relus.append(nn.ReLU())

		self.deform_convs.append(ops.DeformConv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=3, padding=1))
		self.offsets.append(nn.Conv2d(in_channels=output_channels, out_channels=2*3*3, kernel_size=3, padding=1))
		self.relus.append(nn.ReLU())

		self.reset_parameters()

	def reset_parameters(self):
		for m in self.deform_convs:
			nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
			nn.init.constant_(m.bias, 0)
		for m in self.offsets:
			nn.init.constant_(m.weight, 0)
			nn.init.constant_(m.bias, 0)

	def forward(self, x):
		for i in range(len(self.deform_convs)):
			offsets = self.offsets[i].forward(x)
			x = self.deform_convs[i].forward(x, offsets)
			x = self.relus[i].forward(x)
		return x