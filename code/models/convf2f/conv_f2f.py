from torch import nn

class ConvF2F(nn.Module):
	
	def __init__(self, channels=128):

		# input (512, 16, 32)

		self.layers = []

		self.layers.append(nn.Conv2d(in_channels = 4 * channels, out_channels = 2 * channels, kernel_size = 1, padding = 0))
		self.layers.append(nn.ReLU)
		# (256, 16, 32)

		self.layers.append(nn.Conv2d(in_channels = 2 * channels, out_channels = channels/2, kernel_size = 3,padding = 1))
		self.layers.append(nn.ReLU)
		# (64, 16, 32)

		for i in range(2):		
			self.layers.append(nn.Conv2d(in_channels = channels/2, out_channels = channels/2, kernel_size = 3,padding = 1))
			self.layers.append(nn.ReLU)
			# (64, 16, 32)

		self.layers.append(nn.Conv2d(in_channels = channels/2, out_channels = channels, kernel_size = 3,padding = 1))
		self.layers.append(nn.ReLU)

		self.reset_parameters()

	def reset_parameters(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
				nn.init.constant_(m.bias, 0)

	def forward(self, x):
		h = x
		for layer in self.layers:
			h = layer.forward(h)
		return h