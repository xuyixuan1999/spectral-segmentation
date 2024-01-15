import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import SqueezeExcite
import warnings

class sSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.Conv1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        q = self.Conv1x1(U)  # U:[bs,c,h,w] to q:[bs,1,h,w]
        q = self.norm(q)
        return U * q  # 广播机制

class cSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.Conv_Squeeze = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False)
        self.Conv_Excitation = nn.Conv2d(in_channels//2, in_channels, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        z = self.avgpool(U)# shape: [bs, c, h, w] to [bs, c, 1, 1]
        z = self.Conv_Squeeze(z) # shape: [bs, c/2]
        z = self.Conv_Excitation(z) # shape: [bs, c]
        z = self.norm(z)
        return U * z.expand_as(U)

class scSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cSE = cSE(in_channels)
        self.sSE = sSE(in_channels)

    def forward(self, U):
        U_sse = self.sSE(U)
        U_cse = self.cSE(U)
        return U_cse+U_sse

class CABF(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CABF, self).__init__()

        # self.h = h
        # self.w = w

        # self.avg_pool_x = nn.AdaptiveAvgPool2d((h, 1))
        # self.avg_pool_y = nn.AdaptiveAvgPool2d((1, w))

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel//reduction, kernel_size=1, stride=1, bias=False)

        self.h_swish = nn.ReLU()

        self.bn = nn.BatchNorm2d(channel//reduction)

        self.F_h = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)
        self.F_w = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()


    def forward(self, x, x1):
    
        h, w = x.shape[-2:]
        # x_h = self.avg_pool_x(x).permute(0, 1, 3, 2)
        # x_w = self.avg_pool_y(x)
        x_h = F.adaptive_avg_pool2d(x, (h, 1)).permute(0, 1, 3, 2)
        x_w = F.adaptive_avg_pool2d(x, (1, w))

        x_cat_conv_relu = self.h_swish(self.conv_1x1(torch.cat((x_h, x_w), 3)))

        # x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([self.h, self.w], 3)
        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out1 = x * s_h.expand_as(x) * s_w.expand_as(x)
        x_h = F.adaptive_avg_pool2d(x1, (h, 1)).permute(0, 1, 3, 2)
        x_w = F.adaptive_avg_pool2d(x1, (1, w))


        x_cat_conv_relu = self.h_swish(self.conv_1x1(torch.cat((x_h, x_w), 3)))

        # x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([self.h, self.w], 3)
        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out2= x * s_h.expand_as(x) * s_w.expand_as(x)

        out = out1 + out2

        return out
    
class CASPP(nn.Module):
    def __init__(self, c):
        super(CASPP, self).__init__()
        c_ = int(c/4)
        self.mean = nn.AdaptiveAvgPool2d((1, 1))  # (1,1)means ouput_dim
        self.conv = nn.Conv2d(c, c_, 1, 1)
        self.atrous_block1 = nn.Conv2d(c_, c_, 1, 1)

        self.atrous_block6 = nn.Conv2d(c_, c_, 3, 1, padding=5, dilation=5)
        self.atrous_block12 = nn.Conv2d(c_, c_, 3, 1, padding=9, dilation=9)
        self.atrous_block18 = nn.Conv2d(c_, c_, 3, 1, padding=13, dilation=13)
        self.conv_1x1_output = nn.Conv2d(c_ * 5, c, 1, 1)

    def forward(self, x):
        size = x.shape[2:]

        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.interpolate(image_features, size=size, mode='bilinear')
        x1,x2,x3,x4 = torch.split(x,split_size_or_sections=128,dim=1)
        atrous_block1 = self.atrous_block1(x1)

        atrous_block6 = self.atrous_block6(x2+atrous_block1)
        atrous_block12 = self.atrous_block12(x3+atrous_block6)
        atrous_block18 = self.atrous_block18(x4+atrous_block12)

        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        return net

# class SPP(nn.Module):
#     def __init__(self, c):
#         super(SPP, self).__init__()
#         c_ = int(c/4)
#         self.mean = nn.AdaptiveAvgPool2d((1, 1))  # (1,1)means ouput_dim
#         self.conv = nn.Conv2d(c, c_, 1, 1)
#         self.atrous_block1 = nn.Conv2d(c_, c_, 1, 1)

#         self.atrous_block6 = nn.Conv2d(c_, c_, 3, 1, padding=2, dilation=2)
#         self.atrous_block12 = nn.Conv2d(c_, c_, 3, 1, padding=3, dilation=3)
#         self.atrous_block18 = nn.Conv2d(c_, c_, 3, 1, padding=5, dilation=5)
#         self.conv_1x1_output = nn.Conv2d(c_ * 5, c, 1, 1)

#     def forward(self, x):
#         size = x.shape[2:]

#         image_features = self.mean(x)
#         image_features = self.conv(image_features)
#         image_features = F.interpolate(image_features, size=size, mode='bilinear')
#         x1,x2,x3,x4 = torch.split(x,split_size_or_sections=128,dim=1)
#         atrous_block1 = self.atrous_block1(x1)

#         atrous_block6 = self.atrous_block6(x2+atrous_block1)
#         atrous_block12 = self.atrous_block12(x3+atrous_block6)
#         atrous_block18 = self.atrous_block18(x4+atrous_block12)

#         net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
#                                               atrous_block12, atrous_block18], dim=1))
#         return net

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))
        
class SPPFCSPC(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=5):
        super(SPPFCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        x2 = self.m(x1)
        x3 = self.m(x2)
        y1 = self.cv6(self.cv5(torch.cat((x1,x2,x3, self.m(x3)),1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))

class SPP(nn.Module):
    # Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))

class FFN_Conv_BN(nn.Module):
    def __init__(self, ed, h):
        super().__init__()
        self.pw1 = Conv2d_BN(ed, h)
        self.act = nn.GELU()
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0)

    def forward(self, x):
        x = self.pw2(self.act(self.pw1(x)))
        return x

class FFN_BN_Conv(nn.Module):
    def __init__(self, ed, h):
        super().__init__()
        self.pw1 = BN_Conv2d(ed, h)
        self.act = nn.GELU()
        self.pw2 = BN_Conv2d(h, ed, bn_weight_init=0)

    def forward(self, x):
        x = self.pw2(self.act(self.pw1(x)))
        return x
        
class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.conv = nn.Conv2d(a, b, kernel_size=ks, stride=stride, padding=pad, 
                              dilation=dilation, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(b)
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)
        
    def forward(self, input):
        return self.bn(self.conv(input))

class BN_Conv2d(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.conv = nn.Conv2d(a, b, kernel_size=ks, stride=stride, padding=pad, 
                              dilation=dilation, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(b)
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)
        
    def forward(self, input):
        return self.conv(self.bn(input))
    
class DEM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.se = SqueezeExcite(dim, rd_ratio=1/16)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, inputs1, inputs2):
        diff = torch.abs(inputs1 - inputs2)
        
        max_pool = self.max_pool(diff)
        avg_pool = self.avg_pool(diff)
        
        max_pool = self.se(max_pool)
        avg_pool = self.se(avg_pool)
        
        dem = self.sigmoid(avg_pool + max_pool)        
        outputs1 = inputs1 * dem + inputs1
        outputs2 = inputs2 * dem + inputs2
        
        return outputs1 + outputs2

class CSM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(dim, dim // 16, 1, bias=True)
        self.conv2_1 = nn.Conv2d(dim // 16, dim, 1, bias=True)
        self.conv2_2 = nn.Conv2d(dim // 16, dim, 1, bias=True)
        self.act = nn.ReLU(inplace=True)
        
        self.hard_sigmoid = nn.Hardsigmoid(inplace=True)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, inputs1, inputs2):
        add = inputs1 + inputs2
        
        avg = self.avg_pool(add)
        z = self.act(self.conv1(avg))
        
        z1 = self.hard_sigmoid(self.conv2_1(z))
        z2 = self.hard_sigmoid(self.conv2_2(z))
        
        z1 = self.softmax(z1)
        z2 = self.softmax(z2)
        
        outputs1 = inputs1 * z1
        outputs2 = inputs2 * z2
        
        return outputs1 + outputs2
    
class CDSM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dem = DEM(dim)
        self.csm = CSM(dim)
        
    def forward(self, inputs1, inputs2):
        outputs1 = self.dem(inputs1, inputs2)
        outputs2 = self.csm(inputs1, inputs2)
        output = outputs2 + outputs1
        return  output


class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()        
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (5, 1), padding=(2, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn = attn + attn_0 + attn_1 + attn_2

        attn = self.conv3(attn)

        return attn * u

if __name__ == "__main__":
    inputs1 = torch.randn(2, 32, 256, 256)
    inputs2 = torch.randn(2, 32, 256, 256)
    model = CDSM(32)
    outputs = model(inputs1, inputs2)
    print(outputs.shape)
        
        