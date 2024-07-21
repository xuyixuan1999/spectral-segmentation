import sys
sys.path.append('/root/spectral-segmentation')
import torch
import torch.nn as nn
from timm.models.layers import SqueezeExcite
from nets.resnet import *
from nets.vgg import *
from nets.unet import unetUp
from nets.net_utils import Conv2d_BN, SPPF, AttentionModule
from nets.newafft import EfficientUp, CDSM, MSCA

# class DEM(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.se = SqueezeExcite(dim, rd_ratio=1/16)
#         self.sigmoid = nn.Sigmoid()
    
#     def forward(self, inputs1, inputs2):
#         diff = torch.abs(inputs1 - inputs2)
        
#         max_pool = self.max_pool(diff)
#         avg_pool = self.avg_pool(diff)
        
#         max_pool = self.se(max_pool)
#         avg_pool = self.se(avg_pool)
        
#         dem = self.sigmoid(avg_pool + max_pool)        
#         outputs1 = inputs1 * dem + inputs1
#         outputs2 = inputs2 * dem + inputs2
        
#         return outputs1 + outputs2

# class CSM(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv1 = nn.Conv2d(dim, dim // 16, 1, bias=True)
#         self.conv2_1 = nn.Conv2d(dim // 16, dim, 1, bias=True)
#         self.conv2_2 = nn.Conv2d(dim // 16, dim, 1, bias=True)
#         self.act = nn.ReLU(inplace=True)
        
#         self.hard_sigmoid = nn.Hardsigmoid(inplace=True)
#         self.softmax = nn.Softmax(dim=1)
    
#     def forward(self, inputs1, inputs2):
#         add = inputs1 + inputs2
        
#         avg = self.avg_pool(add)
#         z = self.act(self.conv1(avg))
        
#         z1 = self.hard_sigmoid(self.conv2_1(z))
#         z2 = self.hard_sigmoid(self.conv2_2(z))
        
#         z1 = self.softmax(z1)
#         z2 = self.softmax(z2)
        
#         outputs1 = inputs1 * z1
#         outputs2 = inputs2 * z2
        
#         return outputs1 + outputs2
    
# class CDSM(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.dem = DEM(dim)
#         self.csm = CSM(dim)
        
#     def forward(self, inputs1, inputs2):
#         outputs1 = self.dem(inputs1, inputs2)
#         outputs2 = self.csm(inputs1, inputs2)
#         output = outputs2 + outputs1
#         return  output
    
# class PatchExpanding(nn.Module):
#     def __init__(self, dim, out_dim):
#         super().__init__()
#         hid_dim = int(out_dim * 4)
#         self.conv1 = Conv2d_BN(dim, hid_dim, 1, 1, 0)
#         self.act = nn.ReLU()
#         self.up = nn.UpsamplingBilinear2d(scale_factor=2)
#         self.se = SqueezeExcite(hid_dim, .25)
#         self.conv3 = Conv2d_BN(hid_dim, out_dim, 1, 1, 0)

#     def forward(self, x):
#         """
#         x: B, C, H, W 
#         """
#         x = self.conv3(self.se(self.act(self.up(self.act(self.conv1(x))))))
#         return x

# class EfficientUp(nn.Module):
#     def __init__(self, out_size, up_size):
#         super().__init__()
#         #  [64, 64, 128, 256, 512]
#         self.up = PatchExpanding(up_size, out_size)
#         self.conv1 = Conv2d_BN(2*out_size, 2*out_size, 3, 1, 1)
#         self.conv2 = Conv2d_BN(2*out_size, out_size, 1, 1, 0)
#         self.relu = nn.ReLU(inplace=True)
        
#     def forward(self, inputs1, inputs2):
#         outputs = torch.cat([inputs1, self.up(inputs2)], 1)
#         outputs = self.conv1(outputs)
#         outputs = self.relu(outputs)
#         outputs = self.conv2(outputs)
#         outputs = self.relu(outputs)
#         return outputs

# class FFN_MSCA(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
#         self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, padding=1, groups=hidden_features)
#         self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
#         self.act = nn.GELU()
#         self.drop = nn.Dropout(drop)
    
#     def forward(self, x):
#         x = self.fc1(x)
        
#         x = self.dwconv(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x

# class SpatialAttention(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.proj_1 = nn.Conv2d(dim, dim, 1)
#         self.spatial_gating_unit = AttentionModule(dim)
#         self.act = nn.GELU()
#         self.proj_2 = nn.Conv2d(dim, dim, 1)
    
#     def forward(self, x):
#         shorcut = x.clone()
#         x = self.proj_1(x)
#         x = self.act(x)
#         x = self.spatial_gating_unit(x)
#         x = self.proj_2(x)
#         x = x + shorcut
#         return x
 
# class MSCA(nn.Module):    
#     def __init__(self, dim):
#         super().__init__()
#         self.norm_1 = nn.BatchNorm2d(dim)
#         self.norm_2 = nn.BatchNorm2d(dim)
#         self.attn = SpatialAttention(dim)
#         self.ffn = FFN_MSCA(dim)
        
#         # layer_scale_init_value = 1e-2
#         # self.layer_scale_1 = nn.Parameter(torch.ones(1, dim, 1, 1) * layer_scale_init_value)
#         # self.layer_scale_2 = nn.Parameter(torch.ones(1, dim, 1, 1) * layer_scale_init_value)

#     def forward(self, x):

#         # x = x + self.attn(self.norm_1(x)) * self.layer_scale_1
#         # x = x + self.ffn(self.norm_2(x)) * self.layer_scale_2
#         x = x + self.attn(self.norm_1(x))
#         x = x + self.ffn(self.norm_2(x))
        
#         return x
    
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

class AFFT(nn.Module):
    def __init__(self, num_classes=2, pretrained=False, backbone='vgg', in_channels=3):
        super().__init__()
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
        
        # self.concat5 = CDSM(filters[4])
        # self.concat4 = CDSM(filters[3])
        # self.concat3 = CDSM(filters[2])
        # self.concat2 = CDSM(filters[1])
        # self.concat1 = CDSM(filters[0])
        
        self.sppf = SPPF(filters[4], filters[4])
        
        # MSCA
        # self.att5 = MSCA(filters[4])
        # self.att4 = MSCA(filters[3])
        # self.att3 = MSCA(filters[2])
        # self.att2 = MSCA(filters[1])
        # self.att1 = MSCA(filters[0])
        
        # EDM
        self.up_concat4 = EfficientUp(filters[3], filters[4]) # 512
        self.up_concat3 = EfficientUp(filters[2], filters[3]) # 256
        self.up_concat2 = EfficientUp(filters[1], filters[2]) # 128
        self.up_concat1 = EfficientUp(filters[0], filters[1]) # 64
        
        # self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        # self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        # self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        # self.up_concat1 = unetUp(in_filters[0], out_filters[0])
        
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
        
        # fusion
        fused_feat1 = spec_feat1 + rgb_feat1
        
        fused_feat2 = spec_feat2 + rgb_feat2
        fused_feat3 = spec_feat3 + rgb_feat3
        fused_feat4 = spec_feat4 + rgb_feat4
        fused_feat5 = spec_feat5 + rgb_feat5
        
        # fused_feat2 = self.concat2(spec_feat2, rgb_feat2)
        # fused_feat3 = self.concat3(spec_feat3, rgb_feat3)
        # fused_feat4 = self.concat4(spec_feat4, rgb_feat4)
        # fused_feat5 = self.concat5(spec_feat5, rgb_feat5)
        
        # attention
        # fused_feat5 = self.att5(fused_feat5)
        # fused_feat4 = self.att4(fused_feat4)
        # fused_feat3 = self.att3(fused_feat3)
        # fused_feat2 = self.att2(fused_feat2)
        
        # sppf
        fused_feat5 = self.sppf(fused_feat5)
        
        # upsampling
        up4 = self.up_concat4(fused_feat4, fused_feat5)
        up3 = self.up_concat3(fused_feat3, up4)
        up2 = self.up_concat2(fused_feat2, up3)
        up1 = self.up_concat1(fused_feat1, up2)
        
        if self.up_conv is not None:
            up1 = self.up_conv(up1)
        
        final = self.final(up1)
        
        return final
    
    # def forward(self, spec, rgb):
        
    #     rgb_feat1 = self.rgb_backbone.relu(self.rgb_backbone.bn1(self.rgb_backbone.conv1(rgb)))
    #     spec_feat1 = self.spectral_backbone.relu(self.spectral_backbone.bn1(self.spectral_backbone.conv1(spec)))
    #     fused_feat1 = rgb_feat1 + spec_feat1
        
    #     with warnings.catch_warnings():
    #         warnings.simplefilter('ignore') 
    #         rgb_feat2 = self.rgb_backbone.layer1(self.rgb_backbone.maxpool(rgb_feat1 + fused_feat1))
    #         spec_feat2 = self.spectral_backbone.layer1(self.spectral_backbone.maxpool(spec_feat1 + fused_feat1))
    #         fused_feat2 = self.concat2(spec_feat2, rgb_feat2)
            
    #     rgb_feat3 = self.rgb_backbone.layer2(rgb_feat2 + fused_feat2)
    #     spec_feat3 = self.spectral_backbone.layer2(spec_feat2 + fused_feat2)
    #     fused_feat3 = self.concat3(spec_feat3, rgb_feat3)
        
    #     rgb_feat4 = self.rgb_backbone.layer3(rgb_feat3 + fused_feat3)
    #     spec_feat4 = self.spectral_backbone.layer3(spec_feat3 + fused_feat3)
    #     fused_feat4 = self.concat4(spec_feat4, rgb_feat4)
        
    #     rgb_feat5 = self.rgb_backbone.layer4(rgb_feat4 + fused_feat4)
    #     spec_feat5 = self.spectral_backbone.layer4(spec_feat4 + fused_feat4)
    #     fused_feat5 = self.concat5(spec_feat5, rgb_feat5)
        
    #     # attention
    #     fused_feat5 = self.att5(fused_feat5)
    #     fused_feat4 = self.att4(fused_feat4)
    #     fused_feat3 = self.att3(fused_feat3)
    #     fused_feat2 = self.att2(fused_feat2)
        
    #     # sppf
    #     fused_feat5 = self.sppf(fused_feat5)
        
    #     # upsampling
    #     up4 = self.up_concat4(fused_feat4, fused_feat5)
    #     up3 = self.up_concat3(fused_feat3, up4)
    #     up2 = self.up_concat2(fused_feat2, up3)
    #     up1 = self.up_concat1(fused_feat1, up2)
        
    #     if self.up_conv is not None:
    #         up1 = self.up_conv(up1)
        
    #     final = self.final(up1)
        
    #     return final
    
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
    rgb = torch.randn(4, 3, 416, 416)
    spec = torch.randn(4, 25, 416, 416)
    
