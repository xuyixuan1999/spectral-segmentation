from nets.CFT import CFT
from nets.ESPFNet import ESPFNet
from nets.unet import Unet
from nets.efficientvit import EfficientHybrid, EfficientViTs_m0, EfficientViTs_m1
from nets.twoStream import TwoStreamModel, AFFT
from nets.esanet import build_model
from nets.hrnet import HRnet
from nets.swinunet import SwinUnet, SwinUnet_m0, SwinUnet_m1
from nets.deeplabv3 import DeepLab
from nets.sinet import SINet
from nets.dssnet import DSS
from nets.uwunet import Net
from nets.upernet import UPerNet
from nets.segformer import Segformer
from nets.setr import SETRModel
from nets.pspnet import PSPNet
from nets.newafft import NewAFFT

def generate_model(model_name, num_classes, in_channels=3, pretrained=False, backbone='resnet18', cfg=None):
    if model_name == 'unet':
        model = Unet(num_classes, pretrained, backbone, in_channels=in_channels)
    elif model_name == 'hrnet':
        model = HRnet(num_classes, pretrained=pretrained, backbone=backbone, in_channels=in_channels)
    elif model_name == 'swinunet':
        model = SwinUnet(num_classes, in_channels=in_channels, **eval(cfg))
    elif model_name == 'deeplabv3':
        model = DeepLab(num_classes, backbone, pretrained, in_channels=in_channels)
    elif model_name == 'sinet':
        model = SINet(num_classes, in_channels=in_channels)
    elif model_name == 'upernet': # 效果差
        model = UPerNet(num_classes=num_classes, in_channels=in_channels)
    elif model_name == 'segformer':
        model = Segformer(num_classes=num_classes, in_channels=in_channels)
    elif model_name == 'setr': # 效果差
        model = SETRModel(num_classes=num_classes, in_channels=in_channels)
    elif model_name == 'pspnet':
        model = PSPNet(num_classes=num_classes, in_channels=in_channels, backbone=backbone)
    elif model_name == 'dssnet': # 效果差
        model = DSS(num_classes, in_channels)
    elif model_name == 'uwunet': # 效果差
        model = Net(num_classes, in_channels)
    elif model_name == 'cft':
        model = CFT(num_classes, pretrained, backbone)
    elif model_name == 'espfnet':
        model = ESPFNet(num_classes, pretrained, backbone)
    elif model_name == 'twostreammodel':
        model = TwoStreamModel(num_classes, pretrained, backbone)
    elif model_name == 'afft':
        model = AFFT(num_classes, pretrained, backbone)
    elif model_name == 'newafft':
        model = NewAFFT(num_classes, pretrained, backbone)
    elif model_name == 'efficienthybrid':
        model = EfficientHybrid(num_classes, pretrained, backbone, **eval(cfg))
    elif model_name == 'esanet':
        model = build_model(n_classes=num_classes, pretrained_on_imagenet=pretrained, encoder=backbone)
    else:
        raise NotImplementedError
    return model