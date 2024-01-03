import warnings
import sys
sys.path.append('/root/spectral-segmentation')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from nets.resnet import NonBottleneck1D, conv1x1, conv3x3

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes,
                 stride=1, downsample=None, groups=1, base_width=64,
                 dilation=1, norm_layer=None,
                 activation=nn.ReLU(inplace=True), residual_only=False):
        super(BasicBlock, self).__init__()
        self.residual_only = residual_only
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and '
                             'base_width=64')
        # Both self.conv1 and self.downsample layers downsample the input
        # when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
        self.bn1 = norm_layer(planes)
        self.act = activation
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        if self.residual_only:
            return out
        out = out + identity
        out = self.act(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes,
                 stride=1, downsample=None, groups=1, base_width=64,
                 dilation=1, norm_layer=None,
                 activation=nn.ReLU(inplace=True)):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # both self.conv2 and self.downsample layers downsample the input
        # when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.act = activation
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.act(out)
        return out

class ResNet(nn.Module):

    def __init__(self, layers, block,
                 zero_init_residual=False, groups=1, width_per_group=64,
                 replace_stride_with_dilation=None, dilation=None,
                 norm_layer=None, input_channels=3,
                 activation=nn.ReLU(inplace=True)):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        self.replace_stride_with_dilation = replace_stride_with_dilation
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got "
                             "{}".format(replace_stride_with_dilation))
        if dilation is not None:
            dilation = dilation
            if len(dilation) != 4:
                raise ValueError("dilation should be None "
                                 "or a 4-element tuple, got "
                                 "{}".format(dilation))
        else:
            dilation = [1, 1, 1, 1]

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(input_channels, self.inplanes,
                               kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.act = activation
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.down_2_channels_out = 64
        if self.replace_stride_with_dilation == [False, False, False]:
            self.down_4_channels_out = 64 * block.expansion
            self.down_8_channels_out = 128 * block.expansion
            self.down_16_channels_out = 256 * block.expansion
            self.down_32_channels_out = 512 * block.expansion
        elif self.replace_stride_with_dilation == [False, True, True]:
            self.down_4_channels_out = 64 * block.expansion
            self.down_8_channels_out = 512 * block.expansion

        self.layer1 = self._make_layer(
            block, 64, layers[0], dilate=dilation[0]
        )
        self.layer2 = self._make_layer(
            block, 128, layers[1],
            stride=2, dilate=dilation[1],
            replace_stride_with_dilation=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2],
            stride=2, dilate=dilation[2],
            replace_stride_with_dilation=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3],
            stride=2, dilate=dilation[3],
            replace_stride_with_dilation=replace_stride_with_dilation[2]
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual
        # block behaves like an identity. This improves the model by 0.2~0.3%
        # according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks,
                    stride=1, dilate=1, replace_stride_with_dilation=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if replace_stride_with_dilation:
            self.dilation *= stride
            stride = 1
        if dilate > 1:
            self.dilation = dilate
            dilate_first_block = dilate
        else:
            dilate_first_block = previous_dilation
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            self.groups, self.base_width, dilate_first_block,
                            norm_layer,
                            activation=self.act))
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                groups=self.groups,
                                base_width=self.base_width,
                                dilation=self.dilation, norm_layer=norm_layer,
                                activation=self.act))

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x_down2 = self.act(x)
        x = self.maxpool(x_down2)

        x_layer1 = self.forward_resblock(x, self.layer1)
        x_layer2 = self.forward_resblock(x_layer1, self.layer2)
        x_layer3 = self.forward_resblock(x_layer2, self.layer3)
        x_layer4 = self.forward_resblock(x_layer3, self.layer4)

        if self.replace_stride_with_dilation == [False, False, False]:
            features = [x_layer4, x_layer3, x_layer2, x_layer1]

            self.skip3_channels = x_layer3.size()[1]
            self.skip2_channels = x_layer2.size()[1]
            self.skip1_channels = x_layer1.size()[1]
        elif self.replace_stride_with_dilation == [False, True, True]:
            # x has resolution 1/8
            # skip4 has resolution 1/8
            # skip3 has resolution 1/8
            # skip2 has resolution 1/8
            # skip1 has resolution 1/4
            # x_down2 has resolution 1/2
            features = [x, x_layer1, x_down2]

            self.skip3_channels = x_layer3.size()[1]
            self.skip2_channels = x_layer2.size()[1]
            self.skip1_channels = x_layer1.size()[1]

        return features

    def forward_resblock(self, x, layers):
        for l in layers:
            x = l(x)
        return x

    def forward_first_conv(self, x):
        # be aware that maxpool still needs to be applied after this function
        # and before forward_layer1()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        return x

    def forward_layer1(self, x):
        # be ware that maxpool still needs to be applied after
        # forward_first_conv() and before this function
        x = self.forward_resblock(x, self.layer1)
        self.skip1_channels = x.size()[1]
        return x

    def forward_layer2(self, x):
        x = self.forward_resblock(x, self.layer2)
        self.skip2_channels = x.size()[1]
        return x

    def forward_layer3(self, x):
        x = self.forward_resblock(x, self.layer3)
        self.skip3_channels = x.size()[1]
        return x

    def forward_layer4(self, x):
        x = self.forward_resblock(x, self.layer4)
        return x

def ResNet18(pretrained_on_imagenet=False,in_channels=3):
    model = ResNet([2, 2, 2, 2], BasicBlock, input_channels=in_channels)

    if pretrained_on_imagenet:
        state_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet50-0676ba61.pth', model_dir='model_data')
        state_dict = {k: v for k, v in state_dict.items() if (k in model.state_dict()) and (v.size() == model.state_dict()[k].size())}
        model_dict = model.state_dict()
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
        print('Loaded ResNet18 pretrained on ImageNet')
    return model


def ResNet34(pretrained_on_imagenet=False,in_channels=3):
    model = ResNet([3, 4, 6, 3],BasicBlock,  input_channels=in_channels)
    if pretrained_on_imagenet:
        state_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet50-0676ba61.pth', model_dir='model_data')
        state_dict = {k: v for k, v in state_dict.items() if (k in model.state_dict()) and (v.size() == model.state_dict()[k].size())}
        model_dict = model.state_dict()
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
        print('Loaded ResNet34 pretrained on ImageNet')
    return model

def ResNet50(pretrained_on_imagenet=False,in_channels=3):
    model = ResNet([3, 4, 6, 3], Bottleneck, input_channels=in_channels)
    if pretrained_on_imagenet:
        state_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet50-0676ba61.pth', model_dir='model_data')
        state_dict = {k: v for k, v in state_dict.items() if (k in model.state_dict()) and (v.size() == model.state_dict()[k].size())}
        model_dict = model.state_dict()
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
        print('Loaded ResNet50 pretrained on ImageNet')
    return model
    
class SqueezeAndExcitation(nn.Module):
    def __init__(self, channel,
                 reduction=16, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExcitation, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            activation,
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weighting = F.adaptive_avg_pool2d(x, 1)
        weighting = self.fc(weighting)
        y = x * weighting
        return y
    
class SqueezeAndExciteFusionAdd(nn.Module):
    def __init__(self, channels_in, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExciteFusionAdd, self).__init__()

        self.se_rgb = SqueezeAndExcitation(channels_in,
                                           activation=activation)
        self.se_depth = SqueezeAndExcitation(channels_in,
                                             activation=activation)

    def forward(self, rgb, depth):
        rgb = self.se_rgb(rgb)
        depth = self.se_depth(depth)
        out = rgb + depth
        return out

class Swish(nn.Module):
    def forward(self, x):
        return swish(x)

def swish(x):
    return x * torch.sigmoid(x)

class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.

class ConvBNAct(nn.Sequential):
    def __init__(self, channels_in, channels_out, kernel_size,
                 activation=nn.ReLU(inplace=True), dilation=1, stride=1):
        super(ConvBNAct, self).__init__()
        padding = kernel_size // 2 + dilation - 1
        self.add_module('conv', nn.Conv2d(channels_in, channels_out,
                                          kernel_size=kernel_size,
                                          padding=padding,
                                          bias=False,
                                          dilation=dilation,
                                          stride=stride))
        self.add_module('bn', nn.BatchNorm2d(channels_out))
        self.add_module('act', activation)

def get_context_module(context_module_name, channels_in, channels_out,
                       input_size, activation, upsampling_mode='bilinear'):
    if 'appm' in context_module_name:
        if context_module_name == 'appm-1-2-4-8':
            bins = (1, 2, 4, 8)
        else:
            bins = (1, 5)
        context_module = AdaptivePyramidPoolingModule(
            channels_in, channels_out,
            bins=bins,
            input_size=input_size,
            activation=activation,
            upsampling_mode=upsampling_mode)
        channels_after_context_module = channels_out
    elif 'ppm' in context_module_name:
        if context_module_name == 'ppm-1-2-4-8':
            bins = (1, 2, 4, 8)
        else:
            bins = (1, 5)
        context_module = PyramidPoolingModule(
            channels_in, channels_out,
            bins=bins,
            activation=activation,
            upsampling_mode=upsampling_mode)
        channels_after_context_module = channels_out
    else:
        context_module = nn.Identity()
        channels_after_context_module = channels_in
    return context_module, channels_after_context_module


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_dim, out_dim, bins=(1, 2, 3, 6),
                 activation=nn.ReLU(inplace=True),
                 upsampling_mode='bilinear'):
        reduction_dim = in_dim // len(bins)
        super(PyramidPoolingModule, self).__init__()
        self.features = []
        self.upsampling_mode = upsampling_mode
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                ConvBNAct(in_dim, reduction_dim, kernel_size=1,
                          activation=activation)
            ))
        in_dim_last_conv = in_dim + reduction_dim * len(bins)
        self.features = nn.ModuleList(self.features)

        self.final_conv = ConvBNAct(in_dim_last_conv, out_dim,
                                    kernel_size=1, activation=activation)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            h, w = x_size[2:]
            y = f(x)
            if self.upsampling_mode == 'nearest':
                out.append(F.interpolate(y, (int(h), int(w)), mode='nearest'))
            elif self.upsampling_mode == 'bilinear':
                out.append(F.interpolate(y, (int(h), int(w)),
                                         mode='bilinear',
                                         align_corners=False))
            else:
                raise NotImplementedError(
                    'For the PyramidPoolingModule only nearest and bilinear '
                    'interpolation are supported. '
                    f'Got: {self.upsampling_mode}'
                )
        out = torch.cat(out, 1)
        out = self.final_conv(out)
        return out


class AdaptivePyramidPoolingModule(nn.Module):
    def __init__(self, in_dim, out_dim, input_size, bins=(1, 2, 3, 6),
                 activation=nn.ReLU(inplace=True), upsampling_mode='bilinear'):
        reduction_dim = in_dim // len(bins)
        super(AdaptivePyramidPoolingModule, self).__init__()
        self.features = []
        self.upsampling_mode = upsampling_mode
        self.input_size = input_size
        self.bins = bins
        for _ in bins:
            self.features.append(
                ConvBNAct(in_dim, reduction_dim, kernel_size=1,
                          activation=activation)
            )
        in_dim_last_conv = in_dim + reduction_dim * len(bins)
        self.features = nn.ModuleList(self.features)

        self.final_conv = ConvBNAct(in_dim_last_conv, out_dim,
                                    kernel_size=1, activation=activation)

    def forward(self, x):
        x_size = x.size()
        h, w = x_size[2:]
        h_inp, w_inp = self.input_size
        bin_multiplier_h = int((h / h_inp) + 0.5)
        bin_multiplier_w = int((w / w_inp) + 0.5)
        out = [x]
        for f, bin in zip(self.features, self.bins):
            h_pool = bin * bin_multiplier_h
            w_pool = bin * bin_multiplier_w
            pooled = F.adaptive_avg_pool2d(x, (h_pool, w_pool))
            y = f(pooled)
            if self.upsampling_mode == 'nearest':
                out.append(F.interpolate(y, (int(h), int(w)), mode='nearest'))
            elif self.upsampling_mode == 'bilinear':
                out.append(F.interpolate(y, (int(h), int(w)),
                                         mode='bilinear',
                                         align_corners=False))
            else:
                raise NotImplementedError(
                    'For the PyramidPoolingModule only nearest and bilinear '
                    'interpolation are supported. '
                    f'Got: {self.upsampling_mode}'
                )
        out = torch.cat(out, 1)
        out = self.final_conv(out)
        return out

class ESANet(nn.Module):
    def __init__(self,
                 height=480,
                 width=640,
                 num_classes=37,
                 encoder_rgb='resnet18',
                 encoder_depth='resnet18',
                 channels_decoder=None,  # default: [128, 128, 128]
                 pretrained_on_imagenet=True,
                 activation='relu',
                 encoder_decoder_fusion='add',
                 context_module='ppm',
                 nr_decoder_blocks=None,  # default: [1, 1, 1]
                 fuse_depth_in_rgb_encoder='SE-add',
                 upsampling='bilinear'):

        super(ESANet, self).__init__()

        if channels_decoder is None:
            channels_decoder = [128, 128, 128]
        if nr_decoder_blocks is None:
            nr_decoder_blocks = [1, 1, 1]

        self.fuse_depth_in_rgb_encoder = fuse_depth_in_rgb_encoder

        # set activation function
        if activation.lower() == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation.lower() in ['swish', 'silu']:
            self.activation = Swish()
        elif activation.lower() == 'hswish':
            self.activation = Hswish()
        else:
            raise NotImplementedError(
                'Only relu, swish and hswish as activation function are '
                'supported so far. Got {}'.format(activation))

        if encoder_rgb == 'resnet50' or encoder_depth == 'resnet50':
            warnings.warn('Parameter encoder_block is ignored for ResNet50. '
                          'ResNet50 always uses Bottleneck')

        # rgb encoder
        if encoder_rgb == 'resnet18':
            self.encoder_rgb = ResNet18(pretrained_on_imagenet)
        elif encoder_rgb == 'resnet34':
            self.encoder_rgb = ResNet34(pretrained_on_imagenet)
        elif encoder_rgb == 'resnet50':
            self.encoder_rgb = ResNet50(pretrained_on_imagenet)
        else:
            raise NotImplementedError(
                'Only ResNets are supported for '
                'encoder_rgb. Got {}'.format(encoder_rgb))

        # depth encoder
        if encoder_depth == 'resnet18':
            self.encoder_depth = ResNet18(pretrained_on_imagenet, in_channels=25)
        elif encoder_depth == 'resnet34':
            self.encoder_depth = ResNet34(pretrained_on_imagenet, in_channels=25)
        elif encoder_depth == 'resnet50':
            self.encoder_depth = ResNet50(pretrained_on_imagenet, in_channels=25)
        else:
            raise NotImplementedError(
                'Only ResNets are supported for '
                'encoder_depth. Got {}'.format(encoder_rgb))

        self.channels_decoder_in = self.encoder_rgb.down_32_channels_out

        if fuse_depth_in_rgb_encoder == 'SE-add':
            self.se_layer0 = SqueezeAndExciteFusionAdd(
                64, activation=self.activation)
            self.se_layer1 = SqueezeAndExciteFusionAdd(
                self.encoder_rgb.down_4_channels_out,
                activation=self.activation)
            self.se_layer2 = SqueezeAndExciteFusionAdd(
                self.encoder_rgb.down_8_channels_out,
                activation=self.activation)
            self.se_layer3 = SqueezeAndExciteFusionAdd(
                self.encoder_rgb.down_16_channels_out,
                activation=self.activation)
            self.se_layer4 = SqueezeAndExciteFusionAdd(
                self.encoder_rgb.down_32_channels_out,
                activation=self.activation)

        if encoder_decoder_fusion == 'add':
            layers_skip1 = list()
            if self.encoder_rgb.down_4_channels_out != channels_decoder[2]:
                layers_skip1.append(ConvBNAct(
                    self.encoder_rgb.down_4_channels_out,
                    channels_decoder[2],
                    kernel_size=1,
                    activation=self.activation))
            self.skip_layer1 = nn.Sequential(*layers_skip1)

            layers_skip2 = list()
            if self.encoder_rgb.down_8_channels_out != channels_decoder[1]:
                layers_skip2.append(ConvBNAct(
                    self.encoder_rgb.down_8_channels_out,
                    channels_decoder[1],
                    kernel_size=1,
                    activation=self.activation))
            self.skip_layer2 = nn.Sequential(*layers_skip2)

            layers_skip3 = list()
            if self.encoder_rgb.down_16_channels_out != channels_decoder[0]:
                layers_skip3.append(ConvBNAct(
                    self.encoder_rgb.down_16_channels_out,
                    channels_decoder[0],
                    kernel_size=1,
                    activation=self.activation))
            self.skip_layer3 = nn.Sequential(*layers_skip3)

        elif encoder_decoder_fusion == 'None':
            self.skip_layer0 = nn.Identity()
            self.skip_layer1 = nn.Identity()
            self.skip_layer2 = nn.Identity()
            self.skip_layer3 = nn.Identity()

        # context module
        if 'learned-3x3' in upsampling:
            warnings.warn('for the context module the learned upsampling is '
                          'not possible as the feature maps are not upscaled '
                          'by the factor 2. We will use nearest neighbor '
                          'instead.')
            upsampling_context_module = 'nearest'
        else:
            upsampling_context_module = upsampling
        self.context_module, channels_after_context_module = \
            get_context_module(
                context_module,
                self.channels_decoder_in,
                channels_decoder[0],
                input_size=(height // 32, width // 32),
                activation=self.activation,
                upsampling_mode=upsampling_context_module
            )

        # decoder
        self.decoder = Decoder(
            channels_in=channels_after_context_module,
            channels_decoder=channels_decoder,
            activation=self.activation,
            nr_decoder_blocks=nr_decoder_blocks,
            encoder_decoder_fusion=encoder_decoder_fusion,
            upsampling_mode=upsampling,
            num_classes=num_classes
        )

    def forward(self, depth, rgb):
        rgb = self.encoder_rgb.forward_first_conv(rgb)
        depth = self.encoder_depth.forward_first_conv(depth)

        if self.fuse_depth_in_rgb_encoder == 'add':
            fuse = rgb + depth
        else:
            fuse = self.se_layer0(rgb, depth)

        rgb = F.max_pool2d(fuse, kernel_size=3, stride=2, padding=1)
        depth = F.max_pool2d(depth, kernel_size=3, stride=2, padding=1)

        # block 1
        rgb = self.encoder_rgb.forward_layer1(rgb)
        depth = self.encoder_depth.forward_layer1(depth)
        if self.fuse_depth_in_rgb_encoder == 'add':
            fuse = rgb + depth
        else:
            fuse = self.se_layer1(rgb, depth)
        skip1 = self.skip_layer1(fuse)

        # block 2
        rgb = self.encoder_rgb.forward_layer2(fuse)
        depth = self.encoder_depth.forward_layer2(depth)
        if self.fuse_depth_in_rgb_encoder == 'add':
            fuse = rgb + depth
        else:
            fuse = self.se_layer2(rgb, depth)
        skip2 = self.skip_layer2(fuse)

        # block 3
        rgb = self.encoder_rgb.forward_layer3(fuse)
        depth = self.encoder_depth.forward_layer3(depth)
        if self.fuse_depth_in_rgb_encoder == 'add':
            fuse = rgb + depth
        else:
            fuse = self.se_layer3(rgb, depth)
        skip3 = self.skip_layer3(fuse)

        # block 4
        rgb = self.encoder_rgb.forward_layer4(fuse)
        depth = self.encoder_depth.forward_layer4(depth)
        if self.fuse_depth_in_rgb_encoder == 'add':
            fuse = rgb + depth
        else:
            fuse = self.se_layer4(rgb, depth)

        out = self.context_module(fuse)
        out = self.decoder(enc_outs=[out, skip3, skip2, skip1])

        return out


class Decoder(nn.Module):
    def __init__(self,
                 channels_in,
                 channels_decoder,
                 activation=nn.ReLU(inplace=True),
                 nr_decoder_blocks=1,
                 encoder_decoder_fusion='add',
                 upsampling_mode='bilinear',
                 num_classes=37):
        super().__init__()

        self.decoder_module_1 = DecoderModule(
            channels_in=channels_in,
            channels_dec=channels_decoder[0],
            activation=activation,
            nr_decoder_blocks=nr_decoder_blocks[0],
            encoder_decoder_fusion=encoder_decoder_fusion,
            upsampling_mode=upsampling_mode,
            num_classes=num_classes
        )

        self.decoder_module_2 = DecoderModule(
            channels_in=channels_decoder[0],
            channels_dec=channels_decoder[1],
            activation=activation,
            nr_decoder_blocks=nr_decoder_blocks[1],
            encoder_decoder_fusion=encoder_decoder_fusion,
            upsampling_mode=upsampling_mode,
            num_classes=num_classes
        )

        self.decoder_module_3 = DecoderModule(
            channels_in=channels_decoder[1],
            channels_dec=channels_decoder[2],
            activation=activation,
            nr_decoder_blocks=nr_decoder_blocks[2],
            encoder_decoder_fusion=encoder_decoder_fusion,
            upsampling_mode=upsampling_mode,
            num_classes=num_classes
        )
        out_channels = channels_decoder[2]

        self.conv_out = nn.Conv2d(out_channels,
                                  num_classes, kernel_size=3, padding=1)

        # upsample twice with factor 2
        self.upsample1 = Upsample(mode=upsampling_mode,
                                  channels=num_classes)
        self.upsample2 = Upsample(mode=upsampling_mode,
                                  channels=num_classes)

    def forward(self, enc_outs):
        enc_out, enc_skip_down_16, enc_skip_down_8, enc_skip_down_4 = enc_outs

        out, out_down_32 = self.decoder_module_1(enc_out, enc_skip_down_16)
        out, out_down_16 = self.decoder_module_2(out, enc_skip_down_8)
        out, out_down_8 = self.decoder_module_3(out, enc_skip_down_4)

        out = self.conv_out(out)
        out = self.upsample1(out)
        out = self.upsample2(out)

        # if self.training:
        #     return out, out_down_8, out_down_16, out_down_32
        return out


class DecoderModule(nn.Module):
    def __init__(self,
                 channels_in,
                 channels_dec,
                 activation=nn.ReLU(inplace=True),
                 nr_decoder_blocks=1,
                 encoder_decoder_fusion='add',
                 upsampling_mode='bilinear',
                 num_classes=37):
        super().__init__()
        self.upsampling_mode = upsampling_mode
        self.encoder_decoder_fusion = encoder_decoder_fusion

        self.conv3x3 = ConvBNAct(channels_in, channels_dec, kernel_size=3,
                                 activation=activation)

        blocks = []
        for _ in range(nr_decoder_blocks):
            blocks.append(NonBottleneck1D(channels_dec,
                                          channels_dec,
                                          activation=activation)
                          )
        self.decoder_blocks = nn.Sequential(*blocks)

        self.upsample = Upsample(mode=upsampling_mode,
                                 channels=channels_dec)

        # for pyramid supervision
        self.side_output = nn.Conv2d(channels_dec,
                                     num_classes,
                                     kernel_size=1)

    def forward(self, decoder_features, encoder_features):
        out = self.conv3x3(decoder_features)
        out = self.decoder_blocks(out)

        if self.training:
            out_side = self.side_output(out)
        else:
            out_side = None

        out = self.upsample(out)

        if self.encoder_decoder_fusion == 'add':
            out += encoder_features

        return out, out_side


class Upsample(nn.Module):
    def __init__(self, mode, channels=None):
        super(Upsample, self).__init__()
        self.interp = nn.functional.interpolate

        if mode == 'bilinear':
            self.align_corners = False
        else:
            self.align_corners = None

        if 'learned-3x3' in mode:
            # mimic a bilinear interpolation by nearest neigbor upscaling and
            # a following 3x3 conv. Only works as supposed when the
            # feature maps are upscaled by a factor 2.

            if mode == 'learned-3x3':
                self.pad = nn.ReplicationPad2d((1, 1, 1, 1))
                self.conv = nn.Conv2d(channels, channels, groups=channels,
                                      kernel_size=3, padding=0)
            elif mode == 'learned-3x3-zeropad':
                self.pad = nn.Identity()
                self.conv = nn.Conv2d(channels, channels, groups=channels,
                                      kernel_size=3, padding=1)

            # kernel that mimics bilinear interpolation
            w = torch.tensor([[[
                [0.0625, 0.1250, 0.0625],
                [0.1250, 0.2500, 0.1250],
                [0.0625, 0.1250, 0.0625]
            ]]])

            self.conv.weight = torch.nn.Parameter(torch.cat([w] * channels))

            # set bias to zero
            with torch.no_grad():
                self.conv.bias.zero_()

            self.mode = 'nearest'
        else:
            # define pad and conv just to make the forward function simpler
            self.pad = nn.Identity()
            self.conv = nn.Identity()
            self.mode = mode

    def forward(self, x):
        size = (int(x.shape[2]*2), int(x.shape[3]*2))
        x = self.interp(x, size, mode=self.mode,
                        align_corners=self.align_corners)
        x = self.pad(x)
        x = self.conv(x)
        return x


def build_model(n_classes=14, 
                pretrained_on_imagenet=False,
                decoder_channels_mode='decreasing', 
                modality='rgbd', 
                encoder='resnet18', 
                height=416, width=416, 
                activation='relu', 
                encoder_decoder_fusion='add', 
                context_module='ppm', 
                nr_decoder_blocks=3, 
                channels_decoder=128, 
                encoder_depth=None,
                fuse_depth_in_rgb_encoder='SE-add', 
                upsampling='bilinear', 
                he_init=True):

    # set the number of channels in the encoder and for the
    # fused encoder features
    if 'decreasing' in  decoder_channels_mode:
        if  decoder_channels_mode == 'decreasing':
            channels_decoder = [512, 256, 128]

        warnings.warn('Argument --channels_decoder is ignored when '
                      '--decoder_chanels_mode decreasing is set.')
    else:
        channels_decoder = [ channels_decoder] * 3

    if isinstance( nr_decoder_blocks, int):
        nr_decoder_blocks = [ nr_decoder_blocks] * 3
    elif len( nr_decoder_blocks) == 1:
        nr_decoder_blocks =  nr_decoder_blocks * 3
    else:
        nr_decoder_blocks =  nr_decoder_blocks
        assert len(nr_decoder_blocks) == 3

    if  modality == 'rgbd':
        # use the same encoder for depth encoder and rgb encoder if no
        # specific depth encoder is provided
        if  encoder_depth in [None, 'None']:
             encoder_depth =  encoder

        model = ESANet(
            height= height,
            width= width,
            num_classes=n_classes,
            pretrained_on_imagenet=pretrained_on_imagenet,
            encoder_rgb= encoder,
            encoder_depth= encoder_depth,
            activation= activation,
            encoder_decoder_fusion= encoder_decoder_fusion,
            context_module= context_module,
            nr_decoder_blocks=nr_decoder_blocks,
            channels_decoder=channels_decoder,
            fuse_depth_in_rgb_encoder= fuse_depth_in_rgb_encoder,
            upsampling= upsampling
        )

    if  he_init:
        module_list = []

        # first filter out the already pretrained encoder(s)
        for c in model.children():
            if pretrained_on_imagenet and isinstance(c, ResNet):
                # already initialized
                continue
            for m in c.modules():
                module_list.append(m)
                
        for i, m in enumerate(module_list):
            if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                if m.out_channels == n_classes or \
                        isinstance(module_list[i+1], nn.Sigmoid) or \
                        m.groups == m.in_channels:
                    continue
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        print('Applied He init.')

    return model


if __name__ == '__main__':
    input1 = torch.randn(2, 25, 416, 416)
    input2 = torch.randn(2, 3, 416, 416)
    model = build_model()
    output = model(input1, input2)
    print(output.shape)
    