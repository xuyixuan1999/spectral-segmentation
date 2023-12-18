import torch
import torch.nn as nn
import sys
sys.path.append('/root/spectral-segmentation/')
from nets.resnet import resnet50, resnet34, resnet18
from nets.vgg import VGG16
from nets.efficientvit import EfficientViTs, EfficientUp, PatchExpanding, EfficientViTs_m1


class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1  = nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 1)
        self.conv2  = nn.Conv2d(out_size, out_size, kernel_size = 3, padding = 1)
        self.up     = nn.UpsamplingBilinear2d(scale_factor = 2)
        # self.up = nn.ConvTranspose2d(in_size-out_size, in_size-out_size, kernel_size = 3, stride = 2, padding = 1, output_padding = 1)
        self.relu   = nn.ReLU(inplace = True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs

class Unet(nn.Module):
    def __init__(self, num_classes = 21, pretrained = False, backbone = 'vgg', in_channels=3):
        super(Unet, self).__init__()
        if backbone == 'vgg':
            self.vgg    = VGG16(pretrained = pretrained, in_channels=in_channels)
            in_filters  = [192, 384, 768, 1024]
            self.backbone = 'vgg'
        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained = pretrained, in_channels=in_channels)
            in_filters  = [192, 512, 1024, 3072]
            self.backbone = 'resnet50'
        elif backbone == "resnet34":
            self.resnet = resnet34(pretrained = pretrained, in_channels=in_channels)
            in_filters = [192, 320, 640, 768]
            self.backbone = 'resnet34'
        elif backbone == "resnet18":
            self.resnet = resnet18(pretrained = pretrained, in_channels=in_channels)
            in_filters = [192, 320, 640, 768]
            self.backbone = 'resnet18'
        elif backbone == 'efficientvit':
            self.efficientvit = EfficientViTs(in_chans=in_channels, **EfficientViTs_m1)
            in_filters = [192, 320, 640, 768]
            filters = [50,50,100,200,400]
            self.backbone = 'efficientvit'
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        out_filters = [64, 128, 256, 512]

        # upsampling
        if self.backbone != 'efficientvit':
            # 64,64,512
            self.up_concat4 = unetUp(in_filters[3], out_filters[3])
            # 128,128,256
            self.up_concat3 = unetUp(in_filters[2], out_filters[2])
            # 256,256,128
            self.up_concat2 = unetUp(in_filters[1], out_filters[1])
            # 512,512,64
            self.up_concat1 = unetUp(in_filters[0], out_filters[0])
        else:
            # 64,64,512
            self.up_concat4 = EfficientUp(filters[3], filters[4])
            # 128,128,256
            self.up_concat3 = EfficientUp(filters[2], filters[3])
            # 256,256,128
            self.up_concat2 = EfficientUp(filters[1], filters[2])
            # 512,512,64
            self.up_concat1 = EfficientUp(filters[0], filters[1])

        if "resnet" in self.backbone:
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor = 2),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
            )
        elif "efficientvit" in self.backbone:
            self.up_conv = nn.Sequential(
                PatchExpanding(filters[0], filters[0]),
                nn.Conv2d(filters[0], filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
                nn.Conv2d(filters[0], filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None
            
        if self.backbone != 'efficientvit':
            self.final = nn.Conv2d(out_filters[0], num_classes, 1)
        else:
            self.final = nn.Conv2d(filters[0], num_classes, 1)

    def forward(self, inputs):
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        if 'resnet' in self.backbone:
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)
        if self.backbone == "efficientvit":
            [feat1, feat2, feat3, feat4, feat5] = self.efficientvit.forward(inputs)

        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

        if self.up_conv != None:
            up1 = self.up_conv(up1)

        final = self.final(up1)
        
        return final

    def freeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False
        if 'resnet' in self.backbone:
            for param in self.resnet.parameters():
                param.requires_grad = False
        if 'efficientvit' in self.backbone:
            for param in self.efficientvit.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
        if 'resnet' in self.backbone:
            for param in self.resnet.parameters():
                param.requires_grad = True
        if 'efficientvit' in self.backbone:
            for param in self.efficientvit.parameters():
                param.requires_grad = True

if __name__ == "__main__":
    
    spec = torch.randn(1, 3, 416, 416)
    
    model = Unet(in_channels=3, num_classes=11, backbone='efficientvit')
    
    output = Unet(spec)

    print(output.shape)