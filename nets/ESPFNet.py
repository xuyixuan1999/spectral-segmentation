import torch
import torch.nn as nn

import sys
sys.path.append('/root/spectral-segmentation')
from nets.resnet import resnet50, resnet18
from nets.vgg import VGG16
from nets.net_utils import scSE, CABF, CASPP



class decoder(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, groups=1,
                 base_width=None, dilation=1, norm_layer=None,
                 activation=nn.ReLU(inplace=True)):
        super(decoder,self).__init__()
        self.up     = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.conv3x1_1 = nn.Conv2d(inplanes, planes, (3, 1),
                                   stride=(stride, 1), padding=(1, 0),
                                   bias=True)
        self.conv1x3_1 = nn.Conv2d(planes, planes, (1, 3),
                                   stride=(1, stride), padding=(0, 1),
                                   bias=True)
        self.bn1 = nn.BatchNorm2d(planes, eps=1e-03)
        self.act = activation
        self.scse = scSE(planes)
        self.conv3x1_2 = nn.Conv2d(planes, planes, (3, 1),
                                   padding=(1 * dilation, 0), bias=True,
                                   dilation=(dilation, 1))
        self.conv1x3_2 = nn.Conv2d(planes, planes, (1, 3),
                                   padding=(0, 1 * dilation), bias=True,
                                   dilation=(1, dilation))
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-03)

        self.downsample = nn.Conv2d(inplanes,planes,kernel_size=1)

    def forward(self, inputs1, inputs2):
        output1 = torch.cat([inputs1, self.up(inputs2)], 1)
        
        output = self.conv3x1_1(output1)
        output = self.act(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = self.act(output)
        output = self.scse(output)
        output = self.conv3x1_2(output)
        output = self.act(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        identity = self.downsample(output1)

        return self.act(output + identity)

class ESPFNet(nn.Module):
    def __init__(self, num_classes=2, pretrained=False, backbone='resnet18'):
        super(ESPFNet, self).__init__()
        self.backbone = backbone
        if backbone == 'vgg':
            self.rgb_backbone = VGG16(pretrained=pretrained, in_channels=3)
            self.spectral_backbone = VGG16(pretrained=pretrained, in_channels=25)
            filters = [64, 128, 256, 512, 512]
            in_filters = [192, 384, 768, 1024]
        elif backbone == "resnet50":
            self.rgb_backbone = resnet50(pretrained=pretrained, in_channels=3)
            self.spectral_backbone = resnet50(pretrained=pretrained, in_channels=25)
            filters = [64, 256, 512, 1024, 2048]
            in_filters = [192, 512, 1024, 3072]
        elif backbone == "resnet18":
            self.rgb_backbone = resnet18(pretrained=pretrained, in_channels=3)
            self.spectral_backbone = resnet18(pretrained=pretrained, in_channels=25)
            filters = [64, 64, 128, 256, 512]
            in_filters = [192, 320, 640, 768]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        out_filters = [64, 128, 256, 512]

        self.up_concat1 = decoder(inplanes=in_filters[3], planes=out_filters[3])
        self.up_concat2 = decoder(in_filters[2], out_filters[2])
        self.up_concat3 = decoder(in_filters[1], out_filters[1])
        self.up_concat4 = decoder(in_filters[0], out_filters[0])
        
        # CABF 
        self.cabf1 = CABF(filters[0])
        self.cabf2 = CABF(filters[1])
        self.cabf3 = CABF(filters[2])
        self.cabf4 = CABF(filters[3])
        self.cabf5 = CABF(filters[4])
        
        # SPP
        self.spp = CASPP(filters[4])

        if "resnet" in backbone:
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor = 2),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
                scSE(out_filters[0]),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)
        self.backbone = backbone

    def forward(self, spec, rgb):
        [rgb_feat1, rgb_feat2, rgb_feat3, rgb_feat4, rgb_feat5] = self.rgb_backbone(rgb)
        [spec_feat1, spec_feat2, spec_feat3, spec_feat4, spec_feat5] = self.spectral_backbone(spec)
            
        feat1 = self.cabf1(spec_feat1, rgb_feat1)
        feat2 = self.cabf2(spec_feat2, rgb_feat2)
        feat3 = self.cabf3(spec_feat3, rgb_feat3)
        feat4 = self.cabf4(spec_feat4, rgb_feat4)
        feat5 = self.cabf5(spec_feat5, rgb_feat5)
        feat5 = self.spp(feat5)
        
        up4 = self.up_concat1(feat4, feat5)
        up3 = self.up_concat2(feat3, up4)
        up2 = self.up_concat3(feat2, up3)
        up1 = self.up_concat4(feat1, up2)
        up1= self.up_conv(up1)
        final = self.final(up1)

        return final

    def freeze_backbone(self):
        print("Freezing all previous layers...")
        for param in self.rgb_backbone.parameters():
            param.requires_grad = False
        for param in self.spectral_backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        print("Unfreezing all previous layers...")
        for param in self.rgb_backbone.parameters():
            param.requires_grad = True
        for param in self.spectral_backbone.parameters():
            param.requires_grad = True
            
    def update_backbone(self, rgb_backbone_path, spec_backbone_path):
        rgb_static_dict = torch.load(rgb_backbone_path)
        spec_static_dict = torch.load(spec_backbone_path)
        rgb_model_dict = self.rgb_backbone.state_dict()
        spec_model_dict = self.spectral_backbone.state_dict()
        for (key, value) in rgb_static_dict.items():
            key = key.split('resnet.')[-1].split('vgg.')[-1]
            if key in rgb_model_dict:
                rgb_model_dict[key] = value
        for (key, value) in spec_static_dict.items():   
            key = key.split('resnet.')[-1].split('vgg.')[-1]
            if key in spec_model_dict:
                spec_model_dict[key] = value
        self.rgb_backbone.load_state_dict(rgb_model_dict)
        self.spectral_backbone.load_state_dict(spec_model_dict)


if __name__ == '__main__':
    img = torch.randn([1, 3, 416, 416])
    img1 = torch.randn([1, 25, 416, 416])
    model = ESPFNet()
    out = model(img1, img)
    print(out.shape)