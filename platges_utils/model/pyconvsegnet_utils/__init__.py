
"""
import torch

from build_pyconvsegnet import build_pyConvSegNet, adapt_state_dict


a = build_pyConvSegNet(150, layers=152, aux=True)
sd = torch.load('./extern/pyconvsegnet/ade20k_trainval120epochspyconvresnet152_pyconvsegnet.pth')
sd2 = adapt_state_dict(sd)

a.load_state_dict(sd2, strict=False)
    
# Out[17]: _IncompatibleKeys(missing_keys=['backbone.model.fc.weight', 'backbone.model.fc.bias'], unexpected_keys=[])
"""

