from torch import nn
from torchvision import ops

class FullDeformF2F(nn.Module):	

	def __init__(self, output_channels=128, num_past=4, layers=8):
		super(FullDeformF2F, self).__init__()
		
		self.dconv1 = ops.DeformConv2d(in_channels=num_past*output_channels, out_channels=2*output_channels, kernel_size=1, padding=0)
		self.offset1 = nn.Conv2d(in_channels=num_past*output_channels, out_channels=2*1*1, kernel_size=1, padding=0)
		self.relu1 = nn.ReLU()

		self.dconv2 = ops.DeformConv2d(in_channels=2*output_channels, out_channels=output_channels, kernel_size=3, padding=1)
		self.offset2 = nn.Conv2d(in_channels=2*output_channels, out_channels=2*3*3, kernel_size=3, padding=1)
		self.relu2 = nn.ReLU()

		self.dconvs = nn.ModuleList()
		self.offsets = nn.ModuleList()
		self.relus = nn.ModuleList()
		for i in range(layers-4):
			deform_kernel_size = 3
			deform_padding = 1
			self.dconvs.append(ops.DeformConv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=deform_kernel_size, padding=deform_padding))
			self.offsets.append(nn.Conv2d(in_channels=output_channels, out_channels=2*deform_kernel_size*deform_kernel_size, kernel_size=deform_kernel_size, padding=deform_padding))
			self.relus.append(nn.ReLU())

		self.dconv3 = ops.DeformConv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=3, padding=1)
		self.offset3 = nn.Conv2d(in_channels=output_channels, out_channels=2*3*3, kernel_size=3, padding=1)
		self.relu3 = nn.ReLU()

		self.dconv4 = ops.DeformConv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=3, padding=1) 
		self.offset4 = nn.Conv2d(in_channels=output_channels, out_channels=2*3*3, kernel_size=3, padding=1) 

		self.reset_parameters()

	def reset_parameters(self):
		for m in [self.dconv1, self.dconv2, self.dconv3, self.dconv4]:
			nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
			nn.init.constant_(m.bias, 0)
		for m in [self.offset1, self.offset2, self.offset3, self.offset4]:
			nn.init.constant_(m.weight, 0)
			nn.init.constant_(m.bias, 0)

	def forward(self, x):
		x = self.relu1.forward(self.dconv1.forward(x, self.offset1.forward(x)))
		x = self.relu2.forward(self.dconv2.forward(x, self.offset2.forward(x)))

		for i in range(len(self.dconvs)):
			offsets = self.offsets[i].forward(x)
			x = self.dconvs[i].forward(x, offsets)
			x = self.relus[i].forward(x)

		x = self.relu3.forward(self.dconv3.forward(x, self.offset3.forward(x)))
		x = self.dconv4.forward(x, self.offset4.forward(x))

		return x