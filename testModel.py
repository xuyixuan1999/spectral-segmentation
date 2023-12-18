import torch

from nets.twoStream import TwoStreamModel
from nets.unet import Unet
from nets.efficientvit import PatchExpanding
import numpy as np
# if __name__ == "__main__":
#     # rgb = torch.randn(1, 3, 224, 416)
#     # spec = torch.randn(1, 25, 224, 416)
    
#     # # model = TwoStreamModel(num_classes=11, pretrained=False, backbone="resnet18")

#     # model.update_backbone('logs/loss_2023_11_12_14_54_52/last_epoch_weights.pth', 
#     #                       'logs/loss_2023_11_12_15_20_52/last_epoch_weights.pth')
#     # out = model(spec, rgb)
#     # print(out.shape)
    
spec = torch.randn(1, 3,416,416)

model = Unet(num_classes=11, pretrained=False, backbone='resnet18', in_channels=3)
model_dict      = model.state_dict()
pretrained_dict = torch.load('logs/loss_2023_11_12_14_54_52_band3_res18/best_epoch_weights.pth', map_location = 'cpu')
load_key, no_load_key, temp_dict = [], [], {}
for k, v in pretrained_dict.items():
    if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
        temp_dict[k] = v
        load_key.append(k)
    else:
        no_load_key.append(k)
model_dict.update(temp_dict)
model.load_state_dict(model_dict)

output = model(spec)

print(output.shape)
    