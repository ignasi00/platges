
import torch
import torch.nn as nn

from frameworks.pytorch.models.expressions.resnet import ResNet, resnet50, resnet101, resnet152
from frameworks.pytorch.models.expressions.resnet import PyConvResNet, pyconvresnet50, pyconvresnet101, pyconvresnet152
from frameworks.pytorch.models.expressions.pyconvsegnet import PyConvSegNet, PyConvHead
from frameworks.pytorch.models.utils.feature_extractor import FeatureExtractor


def build_pyConvSegNet(num_classes, layers=152, dropout=0.1, zoom_factor=8, BatchNorm=nn.BatchNorm2d, backbone_output_stride=8, backbone_net='pyconvresnet', out_merge_all=256, aux=True):

    if backbone_net == 'pyconvresnet':
        # TODO: for resnet and others, maybe something
        if layers == 50:
            backbone = pyconvresnet50(num_classes)
        elif layers == 101:
            backbone = pyconvresnet101(num_classes)
        elif layers == 152:
            backbone = pyconvresnet152(num_classes)

        if backbone_output_stride == 8:
            for n, m in backbone.layer3.named_modules():
                if 'pyconv_levels.0' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif "pyconv_levels.1" in n:
                    m.dilation, m.padding, m.stride = (2, 2), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)

            for n, m in backbone.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)

        if backbone_output_stride == 16:
            for n, m in backbone.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)

        bb_layers = ['layer4']
        if aux : bb_layers = bb_layers + ['layer3']
        backbone = FeatureExtractor(backbone, bb_layers)
    
    return PyConvSegNet(num_classes, backbone, context_head_class=PyConvHead, backbone_output_maps=2048, 
                        out_merge_all=out_merge_all, dropout=dropout, zoom_factor=zoom_factor,
                        BatchNorm=BatchNorm, aux=aux)
