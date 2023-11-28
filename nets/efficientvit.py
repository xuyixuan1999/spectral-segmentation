# --------------------------------------------------------
# EfficientViT Model Architecture
# Copyright (c) 2022 Microsoft
# Build the EfficientViT Model
# Written by: Xinyu Liu
# --------------------------------------------------------
import torch
import torch.nn as nn
import itertools
import torch.nn.functional as F
from timm.models.vision_transformer import trunc_normal_
from timm.models.layers import SqueezeExcite



EfficientViTs_m0 = {
    'embed_dim' : [64, 64, 128, 256, 512],
    'key_dim' : [16, 16, 32, 32, 64],
    'num_heads' : [4, 4, 4, 8, 8]
}
EfficientViTs_m0_ = {
    'embed_dim' : [64, 64, 128, 256, 512],
    'key_dim' : [16, 16, 32, 64, 128],
    'num_heads' : [4, 4, 4, 4, 4]
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
        'num_heads' : [4, 4, 4, 4, 4]
    }
    model = EfficientViTs(in_chans=4, num_classes=11, **EfficientViTs_m0).cuda()
    output = model(input.cuda())
    print(output.shape)