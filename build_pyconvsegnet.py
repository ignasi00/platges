
import copy
import re
import torch
import torch.nn as nn

from pytorch_models.expressions.resnet import ResNet, resnet50, resnet101, resnet152
from pytorch_models.expressions.resnet import PyConvResNet, pyconvresnet50, pyconvresnet101, pyconvresnet152
from pytorch_models.expressions.pyconvsegnet import PyConvSegNet, PyConvHead
from pytorch_models.utils.feature_extractor import FeatureExtractor


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
                if 'conv2_1' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif "conv2_2" in n:
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

def adapt_state_dict(state_dict):
    state_dict_v2 = copy.deepcopy(state_dict)

    for key in state_dict.keys():
        levels = key.split('.')

        new_key = re.sub(r'conv2_([0-9]+)', lambda match : f'pyconv_levels.{int(match.group(1)) - 1}', key)
        state_dict_v2[new_key] = state_dict_v2.pop(key)

        def replacement(match):
            names = ['conv1', 'bn1', 'bn1', 'bn1']
            return f'backbone.model.{names[int(match.group(1))]}.'
        new_key2 = re.sub(r'^layer0.([0-9]+).', replacement, new_key)
        state_dict_v2[new_key2] = state_dict_v2.pop(new_key)

        new_key = re.sub(r'^layer([1-9]+).', r'backbone.model.layer\1.', new_key2)
        state_dict_v2[new_key] = state_dict_v2.pop(new_key2)
    
    # There is no backbone.model.fc.weight and backbone.model.fc.bias => strict=False is mandatory
    
    return state_dict_v2


if __name__ == "__main__":
    print("This module does not and will not do anything by itself")


"""
import torch

from build_pyconvsegnet import build_pyConvSegNet, adapt_state_dict


a = build_pyConvSegNet(150, layers=152, aux=True)
sd = torch.load('./extern/pyconvsegnet/ade20k_trainval120epochspyconvresnet152_pyconvsegnet.pth')
sd2 = adapt_state_dict(sd)

a.load_state_dict(sd2, strict=False)
    
# Out[17]: _IncompatibleKeys(missing_keys=['backbone.model.fc.weight', 'backbone.model.fc.bias'], unexpected_keys=[])
"""
