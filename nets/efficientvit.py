# --------------------------------------------------------
# EfficientViT Model Architecture
# Copyright (c) 2022 Microsoft
# Build the EfficientViT Model
# Written by: Xinyu Liu
# --------------------------------------------------------
import sys
sys.path.append('/root/spectral-segmentation')
import torch
import torch.nn as nn
import itertools
import torch.nn.functional as F
from timm.models.vision_transformer import trunc_normal_
from timm.models.layers import SqueezeExcite
from nets.resnet import *
from nets.vgg import *
from nets.net_utils import CASPP


EfficientViTs_m0 = {
    'key_dims' : [16, 16, 32, 32, 32],
    'window_resolution' : [13, 13, 13, 13, 13]
}
# EfficientViTs_m0_ = {
#     'embed_dim' : [64, 64, 128, 256, 512],
#     'key_dim' : [16, 16, 32, 64, 128],
#     'num_heads' : [4, 4, 4, 4, 4]
# }
EfficientViTs_m1 = {
    'key_dims' : [16, 16, 32, 32, 32],
    'window_resolution' : [13, 13, 7, 7, 7]
}

EfficientViTs_m2 = {
    # 'key_dims' : [16, 16, 16, 16, 16],
    # 'window_resolution' : [7, 7, 7, 7, 7]
}


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
    #     self.add_module('c', torch.nn.Conv2d(
    #         a, b, ks, stride, pad, dilation, groups, bias=False))
    #     self.add_module('bn', torch.nn.BatchNorm2d(b))
    #     torch.nn.init.constant_(self.bn.weight, bn_weight_init)
    #     torch.nn.init.constant_(self.bn.bias, 0)

    # @torch.no_grad()
    # def fuse(self):
    #     c, bn = self._modules.values()
    #     w = bn.weight / (bn.running_var + bn.eps)**0.5
    #     w = c.weight * w[:, None, None, None]
    #     b = bn.bias - bn.running_mean * bn.weight / \
    #         (bn.running_var + bn.eps)**0.5
    #     m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
    #         0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups)
    #     m.weight.data.copy_(w)
    #     m.bias.data.copy_(b)
    #     return m
        self.conv = nn.Conv2d(a, b, kernel_size=ks, stride=stride, padding=pad, 
                              dilation=dilation, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(b)
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)
        
    def forward(self, input):
        return self.bn(self.conv(input))

class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
    #     self.add_module('bn', torch.nn.BatchNorm1d(a))
    #     self.add_module('l', torch.nn.Linear(a, b, bias=bias))
    #     trunc_normal_(self.l.weight, std=std)
    #     if bias:
    #         torch.nn.init.constant_(self.l.bias, 0)

    # @torch.no_grad()
    # def fuse(self):
    #     bn, l = self._modules.values()
    #     w = bn.weight / (bn.running_var + bn.eps)**0.5
    #     b = bn.bias - self.bn.running_mean * \
    #         self.bn.weight / (bn.running_var + bn.eps)**0.5
    #     w = l.weight * w[None, :]
    #     if l.bias is None:
    #         b = b @ self.l.weight.T
    #     else:
    #         b = (l.weight @ b[:, None]).view(-1) + self.l.bias
    #     m = torch.nn.Linear(w.size(1), w.size(0))
    #     m.weight.data.copy_(w)
    #     m.bias.data.copy_(b)
    #     return m
        self.bn = nn.BatchNorm1d(a)
        self.linear = nn.Linear(a, b, bias=bias)
        trunc_normal_(self.linear.weight, std=std)
        if bias:
            nn.init.constant_(self.linear.bias, 0)
    
    def forward(self, input):
        return self.linear(self.bn(input))

class PatchMerging(torch.nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        hid_dim = int(dim * 4)
        self.conv1 = Conv2d_BN(dim, hid_dim, 1, 1, 0)
        self.act = torch.nn.ReLU()
        self.conv2 = Conv2d_BN(hid_dim, hid_dim, 3, 2, 1, groups=hid_dim)
        self.se = SqueezeExcite(hid_dim, .25)
        self.conv3 = Conv2d_BN(hid_dim, out_dim, 1, 1, 0)

    def forward(self, x):
        x = self.conv3(self.se(self.act(self.conv2(self.act(self.conv1(x))))))
        return x

class PatchExpanding(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        hid_dim = int(out_dim * 4)
        self.conv1 = Conv2d_BN(dim, hid_dim, 1, 1, 0)
        self.act = nn.ReLU()
        self.conv2 = nn.Sequential(nn.ConvTranspose2d(hid_dim, hid_dim, kernel_size=2, \
                                    stride=2, groups=hid_dim, bias=False),
                                   nn.BatchNorm2d(hid_dim))
        self.se = SqueezeExcite(hid_dim, .25)
        self.conv3 = Conv2d_BN(hid_dim, out_dim, 1, 1, 0)

    def forward(self, x):
        """
        x: B, C, H, W 
        """
        x = self.conv3(self.se(self.act(self.conv2(self.act(self.conv1(x))))))
        return x

class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)


class FFN(torch.nn.Module):
    def __init__(self, ed, h):
        super().__init__()
        self.pw1 = Conv2d_BN(ed, h)
        self.act = torch.nn.ReLU()
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0)

    def forward(self, x):
        x = self.pw2(self.act(self.pw1(x)))
        return x


class CascadedGroupAttention(torch.nn.Module):
    r""" Cascaded Group Attention.

    Args:
        dim (int): Number of input channels.
        key_dim (int): The dimension for query and key.
        num_heads (int): Number of attention heads.
        attn_ratio (int): Multiplier for the query dim for value dimension.
        resolution (int): Input resolution, correspond to the window size.
        kernels (List[int]): The kernel size of the dw conv on query.
    """
    def __init__(self, dim, key_dim, num_heads=8,
                 kernels=[5, 5, 5, 5],):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim

        qkvs = []
        dws = []
        for i in range(num_heads):
            # qkvs.append(Conv2d_BN(dim // (num_heads), self.key_dim * 2 + self.d))
            qkvs.append(Conv2d_BN(dim // (num_heads), self.key_dim * 3))
            dws.append(Conv2d_BN(self.key_dim, self.key_dim, kernels[i], 1, kernels[i]//2, groups=self.key_dim))
        self.qkvs = torch.nn.ModuleList(qkvs)
        self.dws = torch.nn.ModuleList(dws)
        self.proj = torch.nn.Sequential(torch.nn.ReLU(), Conv2d_BN(
            self.key_dim * num_heads, dim, bn_weight_init=0))

    def forward(self, x):  # x (B,C,H,W)
        B, C, H, W = x.shape

        feats_in = x.chunk(len(self.qkvs), dim=1)
        feats_out = []
        feat = feats_in[0]
        for i, qkv in enumerate(self.qkvs):
            if i > 0: # add the previous output to the input
                feat = feat + feats_in[i]
            feat = qkv(feat)
            q, k, v = feat.view(B, -1, H, W).split([self.key_dim, self.key_dim, self.key_dim], dim=1) # B, C/h, H, W
            q = self.dws[i](q)
            q, k, v = q.flatten(2), k.flatten(2), v.flatten(2) # B, C/h, N
            attn = (
                (q.transpose(-2, -1) @ k) * self.scale
            )
            attn = attn.softmax(dim=-1) # BNN
            feat = (v @ attn.transpose(-2, -1)).view(B, -1, H, W) # BCHW
            feats_out.append(feat)
        x = self.proj(torch.cat(feats_out, 1))
        return x


class LocalWindowAttention(torch.nn.Module):
    r""" Local Window Attention.

    Args:
        dim (int): Number of input channels.
        key_dim (int): The dimension for query and key.
        num_heads (int): Number of attention heads.
        attn_ratio (int): Multiplier for the query dim for value dimension.
        resolution (int): Input resolution.
        window_resolution (int): Local window resolution.
        kernels (List[int]): The kernel size of the dw conv on query.
    """
    def __init__(self, dim, key_dim, num_heads=8,
                 window_resolution=7,
                 kernels=[5, 5, 5, 5],):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        assert window_resolution > 0, 'window_size must be greater than 0'
        self.window_resolution = window_resolution
        self.avg_pool = nn.AdaptiveAvgPool2d((window_resolution, window_resolution))
        
        self.attn = CascadedGroupAttention(dim, key_dim, num_heads,
                                kernels=kernels,)

    def forward(self, x):
        # # method 1
        # b, c, h, w = x.shape
        # x = self.avg_pool(x)
        # x = self.attn(x)
        # x = F.interpolate(x, size=([h, w]), mode='bilinear')
        
        # method 2
        B, C, H, W = x.shape
        if H <= self.window_resolution and W <= self.window_resolution:
            x = self.attn(x)
        else:
            x = x.permute(0, 2, 3, 1)
            pad_b = (self.window_resolution - H %
                     self.window_resolution) % self.window_resolution
            pad_r = (self.window_resolution - H %
                     self.window_resolution) % self.window_resolution
            padding = pad_b > 0 or pad_r > 0

            if padding:
                x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
                
            pH, pW = H + pad_b, W + pad_r
            nH = pH // self.window_resolution
            nW = pW // self.window_resolution
            x = x.view(B, nH, self.window_resolution, nW, self.window_resolution, C).transpose(2, 3).reshape(
                B * nH * nW, self.window_resolution, self.window_resolution, C
            ).permute(0, 3, 1, 2)
            x = self.attn(x)
            # window reverse, (BnHnW)Chw -> (BnHnW)hwC -> BnHnWhwC -> B(nHh)(nWw)C -> BHWC
            x = x.permute(0, 2, 3, 1).view(B, nH, nW, self.window_resolution, self.window_resolution,
                       C).transpose(2, 3).reshape(B, pH, pW, C)
            if padding:
                x = x[:, :H, :W].contiguous()
            x = x.permute(0, 3, 1, 2)
        return x


class EfficientViTBlock(torch.nn.Module):    
    """ A basic EfficientViT building block.

    Args:
        type (str): Type for token mixer. Default: 's' for self-attention.
        ed (int): Number of input channels.
        kd (int): Dimension for query and key in the token mixer.
        nh (int): Number of attention heads.
        ar (int): Multiplier for the query dim for value dimension.
        resolution (int): Input resolution.
        window_resolution (int): Local window resolution.
        kernels (List[int]): The kernel size of the dw conv on query.
    """
    def __init__(self, type,
                 ed, kd, nh=8,
                 window_resolution=7,
                 kernels=[5, 5, 5, 5],):
        super().__init__()
            
        self.dw0 = Residual(Conv2d_BN(ed, ed, 3, 1, 1, groups=ed, bn_weight_init=0.))
        self.ffn0 = Residual(FFN(ed, int(ed * 2)))

        if type == 's':
            self.mixer = Residual(LocalWindowAttention(ed, kd, nh, \
                    window_resolution=window_resolution, kernels=kernels))
                
        self.dw1 = Residual(Conv2d_BN(ed, ed, 3, 1, 1, groups=ed, bn_weight_init=0.))
        self.ffn1 = Residual(FFN(ed, int(ed * 2)))

    def forward(self, x):
        x = self.ffn0(self.dw0(x))
        x = self.mixer(x)
        x = self.ffn1(self.dw1(x))
        return x


class EfficientViT(torch.nn.Module):
    def __init__(self, img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 stages=['s', 's', 's'],
                 embed_dim=[64, 128, 192],
                 key_dim=[16, 16, 16],
                 depth=[1, 2, 3],
                 num_heads=[4, 4, 4],
                 window_size=[7, 7, 7],
                 kernels=[5, 5, 5, 5],
                 down_ops=[['subsample', 2], ['subsample', 2], ['']],
                 distillation=False,):
        super().__init__()

        resolution = img_size
        # Patch embedding
        self.patch_embed = torch.nn.Sequential(Conv2d_BN(in_chans, embed_dim[0] // 8, 3, 2, 1), torch.nn.ReLU(),
                           Conv2d_BN(embed_dim[0] // 8, embed_dim[0] // 4, 3, 2, 1), torch.nn.ReLU(),
                           Conv2d_BN(embed_dim[0] // 4, embed_dim[0] // 2, 3, 2, 1), torch.nn.ReLU(),
                           Conv2d_BN(embed_dim[0] // 2, embed_dim[0], 3, 2, 1))

        resolution = img_size // patch_size
        attn_ratio = [embed_dim[i] / (key_dim[i] * num_heads[i]) for i in range(len(embed_dim))]
        self.blocks1 = []
        self.blocks2 = []
        self.blocks3 = []

        # Build EfficientViT blocks
        for i, (stg, ed, kd, dpth, nh, ar, wd, do) in enumerate(
                zip(stages, embed_dim, key_dim, depth, num_heads, attn_ratio, window_size, down_ops)):
            for d in range(dpth):
                eval('self.blocks' + str(i+1)).append(EfficientViTBlock(stg, ed, kd, nh, ar, resolution, wd, kernels))
            if do[0] == 'subsample':
                # Build EfficientViT downsample block
                #('Subsample' stride)
                blk = eval('self.blocks' + str(i+2))
                resolution_ = (resolution - 1) // do[1] + 1
                blk.append(torch.nn.Sequential(Residual(Conv2d_BN(embed_dim[i], embed_dim[i], 3, 1, 1, groups=embed_dim[i], resolution=resolution)),
                                    Residual(FFN(embed_dim[i], int(embed_dim[i] * 2), resolution)),))
                blk.append(PatchMerging(*embed_dim[i:i + 2], resolution))
                resolution = resolution_
                blk.append(torch.nn.Sequential(Residual(Conv2d_BN(embed_dim[i + 1], embed_dim[i + 1], 3, 1, 1, groups=embed_dim[i + 1], resolution=resolution)),
                                    Residual(FFN(embed_dim[i + 1], int(embed_dim[i + 1] * 2), resolution)),))
        self.blocks1 = torch.nn.Sequential(*self.blocks1)
        self.blocks2 = torch.nn.Sequential(*self.blocks2)
        self.blocks3 = torch.nn.Sequential(*self.blocks3)
        
        # Classification head
        self.head = BN_Linear(embed_dim[-1], num_classes) if num_classes > 0 else torch.nn.Identity()
        self.distillation = distillation
        if distillation:
            self.head_dist = BN_Linear(embed_dim[-1], num_classes) if num_classes > 0 else torch.nn.Identity()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {x for x in self.state_dict().keys() if 'attention_biases' in x}

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.blocks1(x)
        x = self.blocks2(x)
        x = self.blocks3(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        if self.distillation:
            x = self.head(x), self.head_dist(x)
            if not self.training:
                x = (x[0] + x[1]) / 2
        else:
            x = self.head(x)
        return x
    
class EfficientViTs(nn.Module):
    def __init__(self, img_size=224,
                 in_chans=3,
                 num_classes=1000,
                 stages=['s', 's', 's', 's', 's'],
                 embed_dim=[64, 64, 128, 256, 512],
                 key_dim=[16, 16, 16, 16, 16],
                 depth=[1, 1, 2, 2, 2],
                 num_heads=[4, 4, 8, 16, 32],
                 window_size=[7, 7, 7, 7, 7],
                 kernels=[5, 5, 5, 5]):
        super().__init__()
        kernels = [5 for i in range(num_heads[-1])]
        self.indims = embed_dim[0]
        self.patch_embed = torch.nn.Sequential(Conv2d_BN(in_chans, embed_dim[0], 3, 2, 1), torch.nn.ReLU(),
                                               Conv2d_BN(embed_dim[0], embed_dim[0], 3, 1, 1), nn.ReLU(),
                                               EfficientViTBlock('s', ed=embed_dim[0], kd=key_dim[0], nh=num_heads[0], 
                                                                   window_resolution=window_size[0], kernels=kernels))
        
        self.layer1 = self._make_layer(EfficientViTBlock, blocks=depth[1], stride=1, 
                                       dims=embed_dim[1], key_dims=key_dim[1], n_heads=num_heads[1], 
                                       window_resolution=window_size[1], kernels=kernels)
        
        self.layer2 = self._make_layer(EfficientViTBlock, blocks=depth[2], stride=2,
                                       dims=embed_dim[2], key_dims=key_dim[2], n_heads=num_heads[2],
                                       window_resolution=window_size[2], kernels=kernels)
        
        self.layer3 = self._make_layer(EfficientViTBlock, blocks=depth[3], stride=2, 
                                       dims=embed_dim[3], key_dims=key_dim[3], n_heads=num_heads[3], 
                                       window_resolution=window_size[3], kernels=kernels)
        
        self.layer4 = self._make_layer(EfficientViTBlock, blocks=depth[4], stride=2, 
                                       dims=embed_dim[4], key_dims=key_dim[4], n_heads=num_heads[4], 
                                       window_resolution=window_size[4], kernels=kernels)
        
        
    def _make_layer(self, block, blocks, stride, dims, key_dims, n_heads, window_resolution, kernels):
        
        # EfficientViTBlock('s', ed=128, kd=16, nh=4, ar=128 // (16 * 4),
        #                   window_resolution=8, kernels=[5,5,5,5])
        
        layers = []
        layers.append(PatchMerging(self.indims, dims))
        self.indims = dims
        for i in range(blocks):
            layers.append(block('s', ed=self.indims, kd=key_dims, nh=n_heads,
                            window_resolution=window_resolution, kernels=kernels))
        
        return nn.Sequential(*layers)
        
    def forward(self, x):
        feat1 = self.patch_embed(x)
        
        feat2   = self.layer1(feat1)
        
        feat3   = self.layer2(feat2)
        
        feat4   = self.layer3(feat3)
        
        feat5   = self.layer4(feat4)
        
        return [feat1, feat2, feat3, feat4, feat5]

class EfficientUp(nn.Module):
    def __init__(self, out_size, up_size):
        super().__init__()
        #  [64, 64, 128, 256, 512]
        self.up = PatchExpanding(up_size, out_size)
        self.conv1 = Conv2d_BN(2*out_size, 2*out_size)
        self.conv2 = Conv2d_BN(2*out_size, out_size)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs

class EfficientFusion(nn.Module):
    def __init__(self, embed_dim, key_dims, n_heads, window_resolution, kernels, depth):
        super().__init__()
        # input1: [b, 64, 416, 416]
        # input2: [b, 64, 416, 416]
        self.pos_embed1 = nn.Parameter(torch.ones(1, embed_dim, 1, 1))
        self.pos_embed2 = nn.Parameter(torch.ones(1, embed_dim, 1, 1))
        
        self.blocks = []
        for i in range(depth):
            self.blocks.append(EfficientViTBlock('s', ed=2*embed_dim, kd=2*key_dims, nh=n_heads,
                            window_resolution=window_resolution, kernels=[kernels for _ in range(n_heads)]))
        
        self.blocks = nn.Sequential(*self.blocks)
        
        self.drop = nn.Dropout(0.2)
        
        self.dw_conv1 = Conv2d_BN(embed_dim * 2, embed_dim * 2, 3, 1, 1, groups=embed_dim * 2)
        
        self.dw_conv2 = Conv2d_BN(embed_dim * 2, embed_dim, 1, 1, 0)
        
    def forward(self, inputs1, inputs2):
        embed1 = self.pos_embed1 + inputs1
        embed2 = self.pos_embed2 + inputs2
        
        feat = self.drop(torch.cat([embed1, embed2], dim=1))
        
        out_feat = self.blocks(feat)
        
        out_feat = self.dw_conv1(out_feat)
        
        out_feat = self.dw_conv2(out_feat)
        
        return out_feat 

        
class EfficientHybrid(nn.Module):
    def __init__(self, num_classes=2, pretrained=False, backbone='vgg', 
                 key_dims=[16,16,16,16,16], 
                 window_resolution=[13,13,13,13,13]):
        super().__init__()
        self.backbone = backbone
        if backbone == 'vgg':
            self.rgb_backbone = VGG16(pretrained=pretrained, in_channels=3)
            self.spectral_backbone = VGG16(pretrained=pretrained, in_channels=25)
            filters = [64, 128, 256, 512, 512]
        elif backbone == 'resnet18':
            self.rgb_backbone = resnet18(pretrained=pretrained, in_channels=3)
            self.spectral_backbone = resnet18(pretrained=pretrained, in_channels=25)
            filters = [64, 64, 128, 256, 512]
        elif backbone == 'resnet34':
            self.rgb_backbone = resnet34(pretrained=pretrained, in_channels=3)
            self.spectral_backbone = resnet34(pretrained=pretrained, in_channels=25)
            filters = [64, 64, 128, 256, 512]
        elif backbone == 'resnet50':
            self.rgb_backbone = resnet50(pretrained=pretrained, in_channels=3)
            self.spectral_backbone = resnet50(pretrained=pretrained, in_channels=25)
            filters = [64, 256, 512, 1024, 2048]
        else:
            raise NotImplementedError
        
        
        n_heads = [filters[i] // key_dims[i] for i in range(len(filters))]

        self.up_concat1 = EfficientUp(filters[0], filters[1])
        self.up_concat2 = EfficientUp(filters[1], filters[2])
        self.up_concat3 = EfficientUp(filters[2], filters[3])
        self.up_concat4 = EfficientUp(filters[3], filters[4])

        self.fuse1 = EfficientFusion(filters[0], key_dims[0], n_heads=n_heads[0], window_resolution=window_resolution[0],
                                     kernels=5, depth=1)
        self.fuse2 = EfficientFusion(filters[1], key_dims[1], n_heads=n_heads[1], window_resolution=window_resolution[1],
                                     kernels=5, depth=1)
        self.fuse3 = EfficientFusion(filters[2], key_dims[2], n_heads=n_heads[2], window_resolution=window_resolution[2],  
                                     kernels=5, depth=1)
        self.fuse4 = EfficientFusion(filters[3], key_dims[3], n_heads=n_heads[3], window_resolution=window_resolution[3],
                                     kernels=5, depth=1)
        self.fuse5 = EfficientFusion(filters[4], key_dims[4], n_heads=n_heads[4], window_resolution=window_resolution[4],
                                     kernels=5, depth=1)
        
        # self.spp = CASPP(filters[4])
        
        if "resnet" in backbone:
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(filters[0], filters[0], 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(filters[0], filters[0], 3, padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.up_conv = None
        
        self.final = nn.Conv2d(filters[0], num_classes, 1)
        
    def forward(self, spec, rgb):
        [rgb_feat1, rgb_feat2, rgb_feat3, rgb_feat4, rgb_feat5] = self.rgb_backbone(rgb)
        [spec_feat1, spec_feat2, spec_feat3, spec_feat4, spec_feat5] = self.spectral_backbone(spec)

        # Fuse
        fused_feat1 = self.fuse1(spec_feat1, rgb_feat1)
        fused_feat2 = self.fuse2(spec_feat2, rgb_feat2)
        fused_feat3 = self.fuse3(spec_feat3, rgb_feat3)
        fused_feat4 = self.fuse4(spec_feat4, rgb_feat4)
        fused_feat5 = self.fuse5(spec_feat5, rgb_feat5)
        
        up4 = self.up_concat4(fused_feat4, fused_feat5)
        up3 = self.up_concat3(fused_feat3, up4)
        up2 = self.up_concat2(fused_feat2, up3)
        up1 = self.up_concat1(fused_feat1, up2)
          
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
    input = torch.randn(1, 4, 416, 416)
    # block = EfficientViTBlock('s', ed=128, kd=16, nh=4, ar=128 // (16 * 4), window_resolution=8, kernels=[5,5,5,5])
    # output = block(input)
    EfficientViTs_m0 = {
        'embed_dim' : [64, 64, 128, 256, 512],
        'key_dim' : [16, 16, 32, 32, 64],
        'num_heads' : [4, 4, 4, 8, 8]
    }
    EfficientViTs_m0_ = {
        'embed_dim' : [64, 64, 128, 256, 512],
        'key_dim' : [16, 16, 32, 64, 128],
        'num_heads' : [4, 4, 4, 4, 4],
        'depth':[1, 1, 1, 1, 1],
    }
    # model = EfficientViTs(in_chans=4, num_classes=11, **EfficientViTs_m0).cuda()
    # output = model(input.cuda())
    # print(output.shape)
    # model = EfficientFusion(embed_dim=64, key_dims=16, n_heads=4, window_resolution=13, depth=1, kernels=5)
    model = EfficientHybrid(11, False, 'resnet18')
    
    # inputs1 = torch.randn(1, 25, 416, 416)
    # inputs2 = torch.randn(1, 3, 416, 416)
    # output = model(inputs1, inputs2)
    # print(output.shape)