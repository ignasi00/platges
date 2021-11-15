
import torch
from torch import Tensor
import torch.nn as nn


######

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, 
                                                            groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

######

class BasicBlock(nn.Module):
    """ Copied from torchvision """
    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        """ out_planes = downsample(in_planes) or in_planes
            in_planes -> out_planes
        """

        super(BasicBlock, self).__init__()
        
        norm_layer = norm_layer or nn.BatchNorm2d
        if groups != 1 or base_width != 64 : raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1 : raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = conv3x3(in_planes, out_planes, stride)
        self.bn1 = norm_layer(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_planes, out_planes)
        self.bn2 = norm_layer(out_planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

######

class Bottleneck(nn.Module):
    """ Copied from torchvision """
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        """ out_planes = downsample(in_planes) or in_planes
            out_planes = planes * self.expansion

            in_planes() -> hide_planes(planes, base_width / 64., groups) -> out_planes(planes, self.expansion)
        """ 

        super(Bottleneck, self).__init__()

        norm_layer = norm_layer or nn.BatchNorm2d
        hide_planes = int(planes * (base_width / 64.)) * groups
        out_planes = planes * self.expansion

        self.conv1 = conv1x1(in_planes, hide_planes)
        self.bn1 = norm_layer(hide_planes)
        self.conv2 = conv3x3(hide_planes, hide_planes, stride, groups, dilation)
        self.bn2 = norm_layer(hide_planes)
        self.conv3 = conv1x1(hide_planes, out_planes)
        self.bn3 = norm_layer(out_planes)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out

######

class ResNet(nn.Module):
    """ Mostly copied from torchvision """

    def _make_layer(self, block_class, planes, num_blocks, stride=1, dilate=False):
        
        # stride is the movment of the conv_layer (less sampled convolution) and dilation is the distance from one kernel pixel to the adjacent one (chess board convolution).
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        # the first block funnel the input num_planes into the needed one at the next blocks
        downsample = None
        if stride != 1 or self.current_planes != planes * block_class.expansion:
            downsample = nn.Sequential(
                conv1x1(self.current_planes, planes * block_class.expansion, stride),
                self._norm_layer(planes * block_class.expansion)
            )
        
        layers = [block_class(self.current_planes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, self._norm_layer)]

        # the other blocks planes input and output are compatibles
        self.current_planes = planes * block_class.expansion

        layers.extend([ block_class(self.current_planes, planes, groups=self.groups, base_width=self.base_width, 
                                    dilation=self.dilation, norm_layer=self._norm_layer) for _ in range(1, num_blocks) ])
        
        return nn.Sequential(*layers)

    def __init__(self, block_class, blocks_per_layer, num_classes, *, 
                    in_planes=3, initial_hidden_planes=64, planes_per_layer=None, dilation=1, 
                    groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
        super(ResNet, self).__init__()

        self._norm_layer = norm_layer or nn.BatchNorm2d

        self.current_planes = initial_hidden_planes
        self.dilation = dilation

        replace_stride_with_dilation = replace_stride_with_dilation or [False, False, False]
        if len(replace_stride_with_dilation) != 3 : raise ValueError(f"replace_stride_with_dilation should "
                                        f"be None or a 3-element tuple, got {replace_stride_with_dilation}")

        planes_per_layer = planes_per_layer or [64, 128, 256, 512]
        if len(planes_per_layer) != 4 : raise ValueError(f"planes_per_layer should be None or a 4-element"
                                                                        f" tuple, got {planes_per_layer}")

        self.groups = groups
        self.base_width = width_per_group

        # funnel from input image (in_planes=3) into the ResNet desired input (planes=64); TODO: ¿Why hardwired values or even fixed funneling structure?
        self.conv1 = nn.Conv2d(in_planes, self.current_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self._norm_layer(self.current_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # layers of ResNet with planes_per_layer (funneled if needed) as hidden channel width; TODO: ¿Why hardwired stride=2?
        self.layer1 = self._make_layer(block_class, planes_per_layer[0], blocks_per_layer[0])
        self.layer2 = self._make_layer(block_class, planes_per_layer[1], blocks_per_layer[1], 
                                                    stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block_class, planes_per_layer[2], blocks_per_layer[2], 
                                                    stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block_class, planes_per_layer[3], blocks_per_layer[3], 
                                                    stride=2, dilate=replace_stride_with_dilation[2])

        # Classification head of ResNet (output: scores; _, pred = torch.max(outputs, 1) for class index)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(planes_per_layer[3] * block_class.expansion, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

######

def resnet18(num_classes, **kwargs) : return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, **kwargs)
def resnet34(num_classes, **kwargs) : return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, **kwargs)
def resnet50(num_classes, **kwargs) : return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, **kwargs)
def resnet101(num_classes, **kwargs) : return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, **kwargs)
def resnet152(num_classes, **kwargs) : return ResNet(Bottleneck, [3, 8, 36, 3], num_classes, **kwargs)

######
