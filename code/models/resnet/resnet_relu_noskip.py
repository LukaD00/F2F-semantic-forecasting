import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from itertools import chain
import torch.utils.checkpoint as cp

from ..util import SpatialPyramidPooling, _BNReluConv, upsample

__all__ = ['ResNet', 'resnet18']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def _bn_function_factory(conv, norm, relu=None):
    def bn_function(x):
        x = conv(x)
        if norm is not None:
            x = norm(x)
        if relu is not None:
            x = relu(x)
        return x

    return bn_function


def do_efficient_fwd(block, x, efficient):
    if efficient and x.requires_grad:
        return cp.checkpoint(block, x)
    else:
        return block(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, efficient=True, use_bn=True, relu_inplace=True):
        super(BasicBlock, self).__init__()
        self.use_bn = use_bn
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes) if self.use_bn else None
        self.relu = nn.ReLU(inplace=relu_inplace)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes) if self.use_bn else None
        self.downsample = downsample
        self.stride = stride
        self.efficient = efficient

    def forward(self, x):
        residual = x

        bn_1 = _bn_function_factory(self.conv1, self.bn1, self.relu)
        bn_2 = _bn_function_factory(self.conv2, self.bn2)

        out = do_efficient_fwd(bn_1, x, self.efficient)
        out = do_efficient_fwd(bn_2, out, self.efficient)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        relu = self.relu(out)
        return relu, out


class _Upsample(nn.Module):
    def __init__(self, num_maps_in, num_maps_out, use_bn=True, k=3):
        super(_Upsample, self).__init__()
        print(f'Upsample layer: in = {num_maps_in}, skip = None, out = {num_maps_out}')
        self.blend_conv = _BNReluConv(num_maps_in, num_maps_out, k=k, batch_norm=use_bn)

    def forward(self, x, skip_size):
        x = upsample(x, skip_size)
        x = self.blend_conv.forward(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, *, num_features=128, k_up=3, efficient=True, use_bn=True, **kwargs):
        super(ResNet, self).__init__()
        self.inplanes = 64
        # num_features = 256
        self.efficient = efficient
        self.use_bn = use_bn
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64) if self.use_bn else lambda x: x
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        if kwargs.get('upsample', True):
            upsamples = []
            upsamples += [_Upsample(num_features, num_features, use_bn=self.use_bn, k=k_up)]
            upsamples += [_Upsample(num_features, num_features, use_bn=self.use_bn, k=k_up)]
            upsamples += [_Upsample(num_features, num_features, use_bn=self.use_bn, k=k_up)]
            self.upsample = nn.ModuleList(list(reversed(upsamples)))
            self.random_init = [self.upsample]
        else:
            self.upsample = []
            self.random_init = []
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.fine_tune = [self.conv1, self.maxpool, self.layer1, self.layer2, self.layer3, self.layer4]
        if self.use_bn:
            self.fine_tune += [self.bn1]

        num_levels = 3
        self.spp_size = num_features
        bt_size = self.spp_size
        level_size = self.spp_size // num_levels

        spp_square_grid = False
        spp_grids = [8, 4, 2, 1]

        self.spp = SpatialPyramidPooling(self.inplanes, num_levels, bt_size=bt_size, level_size=level_size,
                                             out_size=self.spp_size, grids=spp_grids, square_grid=spp_square_grid,
                                             bn_momentum=0.01 / 2, use_bn=self.use_bn)
        self.do_spp_in_up = True

        self.random_init += [self.spp]

        self.num_features = num_features

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            layers = [nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False)]
            if self.use_bn:
                layers += [nn.BatchNorm2d(planes * block.expansion)]
            downsample = nn.Sequential(*layers)
        layers = [block(self.inplanes, planes, stride, downsample, efficient=self.efficient, use_bn=self.use_bn)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == blocks - 1:
                layers += [block(self.inplanes, planes, efficient=self.efficient, use_bn=self.use_bn, relu_inplace=False)]
            else:
                layers += [block(self.inplanes, planes, efficient=self.efficient, use_bn=self.use_bn)]
        return nn.Sequential(*layers)

    def random_init_params(self):
        return chain(*[f.parameters() for f in self.random_init])

    def fine_tune_params(self):
        return chain(*[f.parameters() for f in self.fine_tune])

    def forward_resblock(self, x, layers):
        skip = None
        for l in layers:
            x = l(x)
            if isinstance(x, tuple):
                x, skip = x
        return x, skip

    def forward(self, image):
        x = self.conv1(image)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        features = []
        x, skip = self.forward_resblock(x, self.layer1)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer2)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer3)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer4)
        bbone_out = x
        features += [self.spp.forward(skip)]
        features = features[::-1]

        x = features[0]

        upsamples = []
        for skip, up in zip(features[1:], self.upsample):
            x = up(x, skip.size()[2:4])
            upsamples += [x]
        return x, {'features': features, 'upsamples': upsamples, 'backbone_out': bbone_out}

    def forward_down(self, image):
        x = self.conv1(image)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        features = []
        x, skip = self.forward_resblock(x, self.layer1)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer2)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer3)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer4)
        x = self.spp.forward(self.relu(skip))
        return x

    def forward_up(self, x):
        upsamples = []
        for up in self.upsample:
            h, w = x.shape[2:4]
            x = up(x, (2*h, 2*w))
            upsamples += [x]
        return x, {'features': upsamples[-1], 'upsamples': upsamples}


def resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model
