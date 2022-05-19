from torch import nn
from torchvision import ops

class DeformF2F(nn.Module):	

	def __init__(self, output_channels=128, num_past=4, layers=8):
		super(DeformF2F, self).__init__()
		
		self.conv1 = nn.Conv2d(in_channels = num_past*output_channels, out_channels=2*output_channels, kernel_size=1, padding=0)
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

		self.reset_parameters()

	def reset_parameters(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
				nn.init.constant_(m.bias, 0)
		for m in self.offsets:
			nn.init.constant_(m.weight, 0)
			nn.init.constant_(m.bias, 0)

	def forward(self, x):
		x = self.relu1.forward(self.conv1.forward(x))
		x = self.relu2.forward(self.conv2.forward(x))

		for i in range(len(self.deform_convs)):
			offsets = self.offsets[i].forward(x)
			x = self.deform_convs[i].forward(x, offsets)
			x = self.relus[i].forward(x)

		x = self.relu3.forward(self.conv3.forward(x))
		x = self.conv4.forward(x)

		return x