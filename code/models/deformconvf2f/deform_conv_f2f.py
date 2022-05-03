from torch import nn
from torchvision import ops

class DeformConvF2F(nn.Module):	

	def __init__(self, output_channels=128, num_past=4, layers=8):
		super(DeformConvF2F, self).__init__()
		# input (512, 16, 32)

		self.layers = nn.ModuleList()

		self.layers.append(ops.DeformConv2d(in_channels = num_past * output_channels, out_channels=256, kernel_size=1, padding=0))
		self.layers.append(nn.ReLU())
		# (256, 16, 32)
		
		self.layers.append(ops.DeformConv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1))
		self.layers.append(nn.ReLU())
		# (128, 16, 32)

		for i in range(layers-3):		
			self.layers.append(ops.DeformConv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1))
			self.layers.append(nn.ReLU())
			# (128, 16, 32)

		self.layers.append(ops.DeformConv2d(in_channels=128, out_channels = output_channels, kernel_size=3, padding=1))
		self.layers.append(nn.ReLU())

		self.reset_parameters()

	def reset_parameters(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
				nn.init.constant_(m.bias, 0)

	def forward(self, x):
		for layer in self.layers:
			x = layer.forward(x)
		return x