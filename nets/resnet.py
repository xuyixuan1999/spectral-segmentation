import math

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import warnings

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class NonBottleneck1D(nn.Module):
    """
    ERFNet-Block
    Paper:
    http://www.robesafe.es/personal/eduardo.romera/pdfs/Romera17tits.pdf
    Implementation from:
    https://github.com/Eromera/erfnet_pytorch/blob/master/train/erfnet.py
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=None, dilation=1, norm_layer=None,
                 activation=nn.ReLU(inplace=True), residual_only=False):
        super().__init__()
        warnings.warn('parameters groups, base_width and norm_layer are '
                      'ignored in NonBottleneck1D')
        dropprob = 0
        self.conv3x1_1 = nn.Conv2d(inplanes, planes, (3, 1),
                                   stride=(stride, 1), padding=(1, 0),
                                   bias=True)
        self.conv1x3_1 = nn.Conv2d(planes, planes, (1, 3),
                                   stride=(1, stride), padding=(0, 1),
                                   bias=True)
        self.bn1 = nn.BatchNorm2d(planes, eps=1e-03)
        self.act = activation
        self.conv3x1_2 = nn.Conv2d(planes, planes, (3, 1),
                                   padding=(1 * dilation, 0), bias=True,
                                   dilation=(dilation, 1))
        self.conv1x3_2 = nn.Conv2d(planes, planes, (1, 3),
                                   padding=(0, 1 * dilation), bias=True,
                                   dilation=(1, dilation))
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-03)
        self.dropout = nn.Dropout2d(dropprob)
        self.downsample = downsample
        self.stride = stride
        self.residual_only = residual_only

    def forward(self, input):
        output = self.conv3x1_1(input)
        output = self.act(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = self.act(output)

        output = self.conv3x1_2(output)
        output = self.act(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if self.dropout.p != 0:
            output = self.dropout(output)

        if self.downsample is None:
            identity = input
        else:
            identity = self.downsample(input)

        if self.residual_only:
            return output
        # +input = identity (residual connection)
        return self.act(output + identity)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # 利用1x1卷积下降通道数
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        # 利用3x3卷积进行特征提取
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        # 利用1x1卷积上升通道数
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, in_channels=3, num_classes=1000):
        #-----------------------------------------------------------#
        #   假设输入图像为600,600,3
        #   当我们使用resnet50的时候
        #-----------------------------------------------------------#
        self.inplanes = 64
        super(ResNet, self).__init__()
        # 600,600,3 -> 300,300,64
        self.conv1  = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1    = nn.BatchNorm2d(64)
        self.relu   = nn.ReLU(inplace=True)
        # 300,300,64 -> 150,150,64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True) # change
        # 150,150,64 -> 150,150,256
        self.layer1 = self._make_layer(block, 64, layers[0])
        # 150,150,256 -> 75,75,512
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # 75,75,512 -> 38,38,1024
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # 38,38,1024 -> 19,19,2048
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)

        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            x       = self.conv1(x)
            x       = self.bn1(x)
            feat1   = self.relu(x)

            x       = self.maxpool(feat1)
            feat2   = self.layer1(x)

            feat3   = self.layer2(feat2)
            feat4   = self.layer3(feat3)
            feat5   = self.layer4(feat4)
            return [feat1, feat2, feat3, feat4, feat5]

def resnet50(pretrained=False, in_channels=3, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], in_channels, **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet50-0676ba61.pth', model_dir='model_data')
        state_dict = {k: v for k, v in state_dict.items() if (k in model.state_dict()) and (v.size() == model.state_dict()[k].size())}
        model_dict = model.state_dict()
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
        
    del model.avgpool
    del model.fc
    return model

def resnet18(pretrained=False, in_channels=3, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], in_channels, **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet18-f37072fd.pth', model_dir='model_data')
        state_dict = {k: v for k, v in state_dict.items() if (k in model.state_dict()) and (v.size() == model.state_dict()[k].size())}
        model_dict = model.state_dict()
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
        
    del model.avgpool
    del model.fc
    return model

def resnet34(pretrained=False, in_channels=3, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], in_channels, **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet34-b627a593.pth', model_dir='model_data')
        state_dict = {k: v for k, v in state_dict.items() if (k in model.state_dict()) and (v.size() == model.state_dict()[k].size())}
        model_dict = model.state_dict()
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
        
    del model.avgpool
    del model.fc
    return model

def resnet(type='resnet50', pretrained=False, in_channels=3, **kwargs):
    if type == 'resnet50':
        model = ResNet(Bottleneck, [3, 4, 6, 3], in_channels)
        if pretrained:
            state_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet50-0676ba61.pth', model_dir='model_data')
    elif type == 'resnet18':
        model = ResNet(BasicBlock, [2, 2, 2, 2], in_channels)
        if pretrained:
            state_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet18-f37072fd.pth', model_dir='model_data')
    elif type == 'resnet34':
        model = ResNet(BasicBlock, [3, 4, 6, 3], in_channels)
        if pretrained:
            state_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet34-b627a593.pth', model_dir='model_data')
    else:
        raise ValueError('Unknown model type: {}'.format)
    
    state_dict = {k: v for k, v in state_dict.items() if (k in model.state_dict()) and (v.size() == model.state_dict()[k].size())}
    model_dict = model.state_dict()
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    
    del model.avgpool
    del model.fc
    return model
    
        
if __name__ == "__main__":
    import torch
    model = resnet18(pretrained=True, in_channels=4)
    dummy_input = torch.rand(1, 4, 224, 224)
    output = model(dummy_input)

