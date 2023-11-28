import torch

from nets.twoStream import TwoStreamModel
from nets.unet import Unet

if __name__ == "__main__":
    # rgb = torch.randn(1, 3, 224, 416)
    # spec = torch.randn(1, 25, 224, 416)
    
    # # model = TwoStreamModel(num_classes=11, pretrained=False, backbone="resnet18")

    # model.update_backbone('logs/loss_2023_11_12_14_54_52/last_epoch_weights.pth', 
    #                       'logs/loss_2023_11_12_15_20_52/last_epoch_weights.pth')
    # out = model(spec, rgb)
    # print(out.shape)
    
    spec = torch.randn(1, 25, 416, 416)
    
    model = Unet(num_classes=11, backbone='resnet18')
    
    output = Unet(spec)

    print(output.shape)
    