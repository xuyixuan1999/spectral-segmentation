import torch
import torch.nn as nn
import torch.nn.functional as F

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

class SPP(nn.Module):
    def __init__(self, c):
        super(SPP, self).__init__()
        c_ = int(c/4)
        self.mean = nn.AdaptiveAvgPool2d((1, 1))  # (1,1)means ouput_dim
        self.conv = nn.Conv2d(c, c_, 1, 1)
        self.atrous_block1 = nn.Conv2d(c_, c_, 1, 1)

        self.atrous_block6 = nn.Conv2d(c_, c_, 3, 1, padding=2, dilation=2)
        self.atrous_block12 = nn.Conv2d(c_, c_, 3, 1, padding=3, dilation=3)
        self.atrous_block18 = nn.Conv2d(c_, c_, 3, 1, padding=5, dilation=5)
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