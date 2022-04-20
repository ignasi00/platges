
import torch
import torch.nn as nn
import torch.nn.functional as F

from .pyconvsegnet_utils.adapt_state_dict import adapt_state_dict
from .pyconvsegnet_utils.build import build_pyConvSegNet


# Correspondences between ade20k and my project classes
WATER_ID = 1
SAND_ID = 2
OTHERS_ID = 0
FUNNEL_MAP = {
    WATER_ID : [9, 21, 26, 37, 109, 113, 128],
    SAND_ID : [0, 46, 81, 94]
}


def pyConvSegNet(layers, num_classes_pretrain, num_classes, zoom_factor, backbone_output_stride, backbone_net, funnel_map=None, pretrained_back_path=None, pretrained_path=None, bool_adapt_state_dict=False):

    backbone = build_pyConvSegNet(  num_classes_pretrain, layers=layers, dropout=0.1, zoom_factor=zoom_factor, 
                                    BatchNorm=nn.BatchNorm2d, backbone_output_stride=backbone_output_stride,
                                    backbone_net=backbone_net, out_merge_all=256, aux=True)

    if pretrained_back_path is not None:
        checkpoint = torch.load(pretrained_back_path, map_location=torch.device('cpu'))
        if bool_adapt_state_dict : checkpoint = adapt_state_dict(checkpoint)
        backbone.load_state_dict(checkpoint, strict=False)

    head = nn.Conv2d(num_classes_pretrain, num_classes, kernel_size=1)
    model = nn.Sequential(backbone, head)

    if pretrained_path is not None:
        checkpoint = torch.load(pretrained_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint, strict=False)

    elif funnel_map is not None and isinstance(funnel_map, dict):
        head.weight.data = torch.zeros(head.weight.shape)
        for out, in_ in funnel_map.items():
            head.weight.data[out, :, 0, 0] += F.one_hot(torch.LongTensor(in_), num_classes=num_classes_pretrain).sum(dim=0).squeeze()
        head.weight.data[0, :, 0, 0] = (1 - head.weight.data[1:, :, 0, 0].sum(dim=0).squeeze()) / head.weight.data[1:, :, 0, 0].sum().squeeze()
        head.weight.data[head.weight == 0] = torch.randn(head.weight[head.weight == 0].shape) * 1e-3

    return model

def build_authors_pretrained_PyConvSegNet(layers, num_classes_pretrain, num_classes, zoom_factor, backbone_output_stride, backbone_net, funnel_map=None, pretrained_back_path=None, pretrained_path=None, adapt_state_dict=True):
    # As I reimplemented the model based on their code, some names are not there, this is a shortcut to have the state_dict adaptation True by default
    return pyConvSegNet(layers, num_classes_pretrain, num_classes, zoom_factor, backbone_output_stride, backbone_net, funnel_map=funnel_map, pretrained_back_path=pretrained_back_path, pretrained_path=pretrained_path, adapt_state_dict=adapt_state_dict)

def build_PyConvSegNet_from_params(params):
    funnel_map = FUNNEL_MAP if params.funnel_map else params.funnel_map
    return pyConvSegNet(params.layers, params.num_classes_pretrain, params.num_classes, 
                        params.zoom_factor, params.backbone_output_stride, params.backbone_net,
                        funnel_map=funnel_map, pretrained_back_path=params.pretrained_back_path,
                        pretrained_path=params.pretrained_path, bool_adapt_state_dict=params.adapt_state_dict)
