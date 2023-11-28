import torch
import torch.nn as nn

from nets.resnet import *
from nets.vgg import *
from nets.unet import unetUp

class TwoStreamModel(nn.Module):
    def __init__(self, num_classes=2, pretrained=False, backbone='vgg', in_channels=3):
        super(TwoStreamModel, self).__init__()
        self.backbone = backbone
        if backbone == 'vgg':
            self.rgb_backbone = VGG16(pretrained=pretrained, in_channels=3)
            self.spectral_backbone = VGG16(pretrained=pretrained, in_channels=25)
            filters = [64, 128, 256, 512, 512]
            in_filters  = [192, 384, 768, 1024]
        elif backbone == 'resnet18':
            self.rgb_backbone = resnet18(pretrained=pretrained, in_channels=3)
            self.spectral_backbone = resnet18(pretrained=pretrained, in_channels=25)
            filters = [64, 64, 128, 256, 512]
            in_filters = [192, 320, 640, 768]
        elif backbone == 'resnet34':
            self.rgb_backbone = resnet34(pretrained=pretrained, in_channels=3)
            self.spectral_backbone = resnet34(pretrained=pretrained, in_channels=25)
            filters = [64, 64, 128, 256, 512]
            in_filters = [192, 320, 640, 768]
        elif backbone == 'resnet50':
            self.rgb_backbone = resnet50(pretrained=pretrained, in_channels=3)
            self.spectral_backbone = resnet50(pretrained=pretrained, in_channels=25)
            filters = [64, 256, 512, 1024, 2048]
            in_filters  = [192, 512, 1024, 3072]
        else:
            raise NotImplementedError
        
        out_filters = [64, 128, 256, 512]
        
        self.concat5 = conv1x1(2*filters[4], filters[4])
        self.concat4 = conv1x1(2*filters[3], filters[3])
        self.concat3 = conv1x1(2*filters[2], filters[2])
        self.concat2 = conv1x1(2*filters[1], filters[1])
        self.concat1 = conv1x1(2*filters[0], filters[0])
        
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])
        
        if "resnet" in backbone:
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(filters[0], out_filters[0], 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_filters[0], out_filters[0], 3, padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.up_conv = None
        
        self.final = nn.Conv2d(out_filters[0], num_classes, 1)
        
    def forward(self, spec, rgb):
        [rgb_feat1, rgb_feat2, rgb_feat3, rgb_feat4, rgb_feat5] = self.rgb_backbone(rgb)
        [spec_feat1, spec_feat2, spec_feat3, spec_feat4, spec_feat5] = self.spectral_backbone(spec)


        up4 = self.up_concat4(self.concat4(torch.cat([rgb_feat4, spec_feat4], dim=1)),
                              self.concat5(torch.cat([rgb_feat5, spec_feat5], dim=1)))
        up3 = self.up_concat3(self.concat3(torch.cat([rgb_feat3, spec_feat3], dim=1)), up4)
        up2 = self.up_concat2(self.concat2(torch.cat([rgb_feat2, spec_feat2], dim=1)), up3)
        up1 = self.up_concat1(self.concat1(torch.cat([rgb_feat1, spec_feat1], dim=1)), up2)
        
        if self.up_conv is not None:
            up1 = self.up_conv(up1)
        
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
    
if __name__ == "__main__":
    rgb = torch.randn(1, 3, 224, 416)
    spec = torch.randn(1, 25, 224, 416)
    
    model = TwoStreamModel(num_classes=11, pretrained=False, backbone="resnet18")

    model.update_backbone('logs/loss_2023_11_12_14_54_52/last_epoch_weights.pth', 
                          'logs/loss_2023_11_12_15_20_52/last_epoch_weights.pth')
    out = model(rgb, spec)
    print(out.shape)