import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

BatchNorm2d = nn.BatchNorm2d

bn_mom = 0.0003

class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False,activate_first=True,inplace=True):
        super(SeparableConv2d,self).__init__()
        self.relu0 = nn.ReLU(inplace=inplace)
        self.depthwise = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.bn1 = nn.BatchNorm2d(in_channels, momentum=bn_mom)
        self.relu1 = nn.ReLU(inplace=True)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=bn_mom)
        self.relu2 = nn.ReLU(inplace=True)
        self.activate_first = activate_first
    def forward(self,x):
        if self.activate_first:
            x = self.relu0(x)
        x = self.depthwise(x)
        x = self.bn1(x)
        if not self.activate_first:
            x = self.relu1(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        if not self.activate_first:
            x = self.relu2(x)
        return x

class Block(nn.Module):
    def __init__(self,in_filters,out_filters,strides=1,atrous=None,grow_first=True,activate_first=True,inplace=True):
        super(Block, self).__init__()
        if atrous == None:
            atrous = [1]*3
        elif isinstance(atrous, int):
            atrous_list = [atrous]*3
            atrous = atrous_list
        idx = 0
        self.head_relu = True
        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters, momentum=bn_mom)
            self.head_relu = False
        else:
            self.skip=None
        
        self.hook_layer = None
        if grow_first:
            filters = out_filters
        else:
            filters = in_filters
        self.sepconv1 = SeparableConv2d(in_filters,filters,3,stride=1,padding=1*atrous[0],dilation=atrous[0],bias=False,activate_first=activate_first,inplace=self.head_relu)
        self.sepconv2 = SeparableConv2d(filters,out_filters,3,stride=1,padding=1*atrous[1],dilation=atrous[1],bias=False,activate_first=activate_first)
        self.sepconv3 = SeparableConv2d(out_filters,out_filters,3,stride=strides,padding=1*atrous[2],dilation=atrous[2],bias=False,activate_first=activate_first,inplace=inplace)

    def forward(self,inp):
        
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x = self.sepconv1(inp)
        x = self.sepconv2(x)
        self.hook_layer = x
        x = self.sepconv3(x)

        x+=skip
        return x


class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """
    def __init__(self, downsample_factor, in_channels=3):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()

        stride_list = None
        if downsample_factor == 8:
            stride_list = [2,1,1]
        elif downsample_factor == 16:
            stride_list = [2,2,1]
        else:
            raise ValueError('xception.py: output stride=%d is not supported.'%os) 
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(32,64,3,1,1,bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=bn_mom)
        #do relu here

        self.block1=Block(64,128,2)
        self.block2=Block(128,256,stride_list[0],inplace=False)
        self.block3=Block(256,728,stride_list[1])

        rate = 16//downsample_factor
        self.block4=Block(728,728,1,atrous=rate)
        self.block5=Block(728,728,1,atrous=rate)
        self.block6=Block(728,728,1,atrous=rate)
        self.block7=Block(728,728,1,atrous=rate)

        self.block8=Block(728,728,1,atrous=rate)
        self.block9=Block(728,728,1,atrous=rate)
        self.block10=Block(728,728,1,atrous=rate)
        self.block11=Block(728,728,1,atrous=rate)

        self.block12=Block(728,728,1,atrous=rate)
        self.block13=Block(728,728,1,atrous=rate)
        self.block14=Block(728,728,1,atrous=rate)
        self.block15=Block(728,728,1,atrous=rate)

        self.block16=Block(728,728,1,atrous=[1*rate,1*rate,1*rate])
        self.block17=Block(728,728,1,atrous=[1*rate,1*rate,1*rate])
        self.block18=Block(728,728,1,atrous=[1*rate,1*rate,1*rate])
        self.block19=Block(728,728,1,atrous=[1*rate,1*rate,1*rate])
        
        self.block20=Block(728,1024,stride_list[2],atrous=rate,grow_first=False)
        self.conv3 = SeparableConv2d(1024,1536,3,1,1*rate,dilation=rate,activate_first=False)

        self.conv4 = SeparableConv2d(1536,1536,3,1,1*rate,dilation=rate,activate_first=False)

        self.conv5 = SeparableConv2d(1536,2048,3,1,1*rate,dilation=rate,activate_first=False)
        self.layers = []

        #------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #-----------------------------

    def forward(self, input):
        self.layers = []
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.block1(x)
        x = self.block2(x)
        low_featrue_layer = self.block2.hook_layer
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        x = self.block19(x)
        x = self.block20(x)       

        x = self.conv3(x)

        x = self.conv4(x)
        
        x = self.conv5(x)
        return low_featrue_layer,x

def load_url(url, model_dir='./model_data', map_location=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if os.path.exists(cached_file):
        return torch.load(cached_file, map_location=map_location)
    else:
        return model_zoo.load_url(url,model_dir=model_dir)

def xception(pretrained=True, downsample_factor=16, in_channels=3):
    model = Xception(downsample_factor=downsample_factor, in_channels=in_channels)
    if pretrained:
        # model.load_state_dict(load_url('https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/xception_pytorch_imagenet.pth'), strict=False)
        state_dict = load_url('https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/xception_pytorch_imagenet.pth')
        state_dict = {k: v for k, v in state_dict.items() if (k in model.state_dict()) and (v.size() == model.state_dict()[k].size())}
        model_dict = model.state_dict()
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
        print("xception loaded")
    return model

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                #--------------------------------------------#
                #   进行3x3的逐层卷积，进行跨特征点的特征提取
                #--------------------------------------------#
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                #-----------------------------------#
                #   利用1x1卷积进行通道数的调整
                #-----------------------------------#
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                #-----------------------------------#
                #   利用1x1卷积进行通道数的上升
                #-----------------------------------#
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                #--------------------------------------------#
                #   进行3x3的逐层卷积，进行跨特征点的特征提取
                #--------------------------------------------#
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                #-----------------------------------#
                #   利用1x1卷积进行通道数的下降
                #-----------------------------------#
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1., in_channels=3):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1], # 256, 256, 32 -> 256, 256, 16
            [6, 24, 2, 2], # 256, 256, 16 -> 128, 128, 24   2
            [6, 32, 3, 2], # 128, 128, 24 -> 64, 64, 32     4
            [6, 64, 4, 2], # 64, 64, 32 -> 32, 32, 64       7
            [6, 96, 3, 1], # 32, 32, 64 -> 32, 32, 96
            [6, 160, 3, 2], # 32, 32, 96 -> 16, 16, 160     14
            [6, 320, 1, 1], # 16, 16, 160 -> 16, 16, 320
        ]

        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        # 512, 512, 3 -> 256, 256, 32
        self.features = [conv_bn(in_channels, input_channel, 2)]

        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel

        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        self.features = nn.Sequential(*self.features)

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def load_url(url, model_dir='./model_data', map_location=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if os.path.exists(cached_file):
        return torch.load(cached_file, map_location=map_location)
    else:
        return model_zoo.load_url(url,model_dir=model_dir)

def mobilenetv2(pretrained=False, in_channels=3, **kwargs):
    model = MobileNetV2(n_class=1000, in_channels=in_channels, **kwargs)
    if pretrained:
        state_dict = load_url('https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/mobilenet_v2.pth.tar')  
        state_dict = {k: v for k, v in state_dict.items() if (k in model.state_dict()) and (v.size() == model.state_dict()[k].size())}
        model_dict = model.state_dict()
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
        print("mobilenetv2 loaded")
    return model

class MobileNetV2_(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True, in_channels=3):
        super(MobileNetV2_, self).__init__()
        from functools import partial
        
        model           = mobilenetv2(pretrained, in_channels=in_channels)
        self.features   = model.features[:-1]

        self.total_idx  = len(self.features)
        self.down_idx   = [2, 4, 7, 14]

        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
        
    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        low_level_features = self.features[:4](x)
        x = self.features[4:](low_level_features)
        return low_level_features, x 


#-----------------------------------------#
#   ASPP特征提取模块
#   利用不同膨胀率的膨胀卷积进行特征提取
#-----------------------------------------#
class ASPP(nn.Module):
	def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
		super(ASPP, self).__init__()
		self.branch1 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate,bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),
		)
		self.branch2 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 3, 1, padding=6*rate, dilation=6*rate, bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),	
		)
		self.branch3 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 3, 1, padding=12*rate, dilation=12*rate, bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),	
		)
		self.branch4 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 3, 1, padding=18*rate, dilation=18*rate, bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),	
		)
		self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0,bias=True)
		self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
		self.branch5_relu = nn.ReLU(inplace=True)

		self.conv_cat = nn.Sequential(
				nn.Conv2d(dim_out*5, dim_out, 1, 1, padding=0,bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),		
		)

	def forward(self, x):
		[b, c, row, col] = x.size()
        #-----------------------------------------#
        #   一共五个分支
        #-----------------------------------------#
		conv1x1 = self.branch1(x)
		conv3x3_1 = self.branch2(x)
		conv3x3_2 = self.branch3(x)
		conv3x3_3 = self.branch4(x)
        #-----------------------------------------#
        #   第五个分支，全局平均池化+卷积
        #-----------------------------------------#
		global_feature = torch.mean(x,2,True)
		global_feature = torch.mean(global_feature,3,True)
		global_feature = self.branch5_conv(global_feature)
		global_feature = self.branch5_bn(global_feature)
		global_feature = self.branch5_relu(global_feature)
		global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)
		
        #-----------------------------------------#
        #   将五个分支的内容堆叠起来
        #   然后1x1卷积整合特征。
        #-----------------------------------------#
		feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
		result = self.conv_cat(feature_cat)
		return result

class DeepLab(nn.Module):
    def __init__(self, num_classes, backbone="mobilenet", pretrained=True, downsample_factor=16, in_channels=3):
        super(DeepLab, self).__init__()
        if backbone=="xception":
            #----------------------------------#
            #   获得两个特征层
            #   浅层特征    [128,128,256]
            #   主干部分    [30,30,2048]
            #----------------------------------#
            self.backbone = xception(downsample_factor=downsample_factor, pretrained=pretrained, in_channels=in_channels)
            in_channels_ = 2048
            low_level_channels = 256
        elif backbone=="mobilenet":
            #----------------------------------#
            #   获得两个特征层
            #   浅层特征    [128,128,24]
            #   主干部分    [30,30,320]
            #----------------------------------#
            self.backbone = MobileNetV2_(downsample_factor=downsample_factor, pretrained=pretrained, in_channels=in_channels)
            in_channels_ = 320
            low_level_channels = 24
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, xception.'.format(backbone))

        #-----------------------------------------#
        #   ASPP特征提取模块
        #   利用不同膨胀率的膨胀卷积进行特征提取
        #-----------------------------------------#
        self.aspp = ASPP(dim_in=in_channels_, dim_out=256, rate=16//downsample_factor)
        
        #----------------------------------#
        #   浅层特征边
        #----------------------------------#
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )		

        self.cat_conv = nn.Sequential(
            nn.Conv2d(48+256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Dropout(0.1),
        )
        self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        #-----------------------------------------#
        #   获得两个特征层
        #   low_level_features: 浅层特征-进行卷积处理
        #   x : 主干部分-利用ASPP结构进行加强特征提取
        #-----------------------------------------#
        low_level_features, x = self.backbone(x)
        x = self.aspp(x)
        low_level_features = self.shortcut_conv(low_level_features)
        
        #-----------------------------------------#
        #   将加强特征边上采样
        #   与浅层特征堆叠后利用卷积进行特征提取
        #-----------------------------------------#
        x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear', align_corners=True)
        x = self.cat_conv(torch.cat((x, low_level_features), dim=1))
        x = self.cls_conv(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x

if __name__ == "__main__":
    model = DeepLab(num_classes=21, backbone="xception", pretrained=True, downsample_factor=16, in_channels=25)
    model.eval()
    inputs = torch.randn(1,25,416,416)
    outputs = model(inputs)
    print(outputs.size())
