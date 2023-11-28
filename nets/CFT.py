import torch.nn as nn
from torch.nn import init, Sequential
import torch
import torch.nn.functional as F
import numpy as np

import sys
sys.path.append('/root/spectral-segmentation')
from nets.resnet import *
from nets.vgg import *
from nets.unet import unetUp

class SelfAttention(nn.Module):
    """
     Multi-head masked self-attention layer
    """

    def __init__(self, d_model, d_k, d_v, h, attn_pdrop=.1, resid_pdrop=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(SelfAttention, self).__init__()
        assert d_k % h == 0
        self.d_model = d_model
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.h = h

        # key, query, value projections for all heads
        self.que_proj = nn.Linear(d_model, h * self.d_k)  # query projection
        self.key_proj = nn.Linear(d_model, h * self.d_k)  # key projection
        self.val_proj = nn.Linear(d_model, h * self.d_v)  # value projection
        self.out_proj = nn.Linear(h * self.d_v, d_model)  # output projection

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, attention_mask=None, attention_weights=None):
        '''
        Computes Self-Attention
        Args:
            x (tensor): input (token) dim:(b_s, nx, c),
                b_s means batch size
                nx means length, for CNN, equals H*W, i.e. the length of feature maps
                c means channel, i.e. the channel of feature maps
            attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
            attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        Return:
            output (tensor): dim:(b_s, nx, c)
        '''

        b_s, nq = x.shape[:2]
        nk = x.shape[1]
        q = self.que_proj(x).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.key_proj(x).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk) K^T
        v = self.val_proj(x).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        # Self-Attention
        #  :math:`(\text(Attention(Q,K,V) = Softmax((Q*K^T)/\sqrt(d_k))`
        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)

        # weight and mask
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)

        # get attention matrix
        att = torch.softmax(att, -1)
        att = self.attn_drop(att)

        # output
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.resid_drop(self.out_proj(out))  # (b_s, nq, d_model)

        return out


class myTransformerBlock(nn.Module):
    """ Transformer block """

    def __init__(self, d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        :param block_exp: Expansion factor for MLP (feed foreword network)

        """
        super().__init__()
        self.ln_input = nn.LayerNorm(d_model)
        self.ln_output = nn.LayerNorm(d_model)
        self.sa = SelfAttention(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, block_exp * d_model),
            # nn.SiLU(),  # changed from GELU
            nn.GELU(),  # changed from GELU
            nn.Linear(block_exp * d_model, d_model),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        bs, nx, c = x.size()

        x = x + self.sa(self.ln_input(x))
        x = x + self.mlp(self.ln_output(x))

        return x


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, d_model, h=8, block_exp=4,
                 n_layer=8, vert_anchors=8, horz_anchors=8,
                 embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()

        self.n_embd = d_model
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors

        d_k = d_model
        d_v = d_model

        # positional embedding parameter (learnable), rgb_fea + ir_fea
        self.pos_emb = nn.Parameter(torch.zeros(1, 2 * vert_anchors * horz_anchors, self.n_embd))

        # transformer
        self.trans_blocks = nn.Sequential(*[myTransformerBlock(d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop)
                                            for layer in range(n_layer)])

        # decoder head
        self.ln_f = nn.LayerNorm(self.n_embd)

        # regularization
        self.drop = nn.Dropout(embd_pdrop)

        # avgpool
        self.avgpool = nn.AdaptiveAvgPool2d((self.vert_anchors, self.horz_anchors))

        # init weights
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        """
        Args:
            x (tuple?)

        """
        rgb_fea = x[0]  # rgb_fea (tensor): dim:(B, C, H, W)
        ir_fea = x[1]   # ir_fea (tensor): dim:(B, C, H, W)
        assert rgb_fea.shape[0] == ir_fea.shape[0]
        bs, c, h, w = rgb_fea.shape

        # -------------------------------------------------------------------------
        # AvgPooling
        # -------------------------------------------------------------------------
        # AvgPooling for reduce the dimension due to expensive computation
        rgb_fea = self.avgpool(rgb_fea)
        ir_fea = self.avgpool(ir_fea)

        # -------------------------------------------------------------------------
        # Transformer
        # -------------------------------------------------------------------------
        # pad token embeddings along number of tokens dimension
        rgb_fea_flat = rgb_fea.view(bs, c, -1)  # flatten the feature
        ir_fea_flat = ir_fea.view(bs, c, -1)  # flatten the feature
        token_embeddings = torch.cat([rgb_fea_flat, ir_fea_flat], dim=2)  # concat
        token_embeddings = token_embeddings.permute(0, 2, 1).contiguous()  # dim:(B, 2*H*W, C)

        # transformer
        x = self.drop(self.pos_emb + token_embeddings)  # sum positional embedding and token    dim:(B, 2*H*W, C)
        x = self.trans_blocks(x)  # dim:(B, 2*H*W, C)

        # decoder head
        x = self.ln_f(x)  # dim:(B, 2*H*W, C)
        x = x.view(bs, 2, self.vert_anchors, self.horz_anchors, self.n_embd)
        x = x.permute(0, 1, 4, 2, 3)  # dim:(B, 2, C, H, W)

        # 这样截取的方式, 是否采用映射的方式更加合理？
        rgb_fea_out = x[:, 0, :, :, :].contiguous().view(bs, self.n_embd, self.vert_anchors, self.horz_anchors)
        ir_fea_out = x[:, 1, :, :, :].contiguous().view(bs, self.n_embd, self.vert_anchors, self.horz_anchors)

        # -------------------------------------------------------------------------
        # Interpolate (or Upsample)
        # -------------------------------------------------------------------------
        rgb_fea_out = F.interpolate(rgb_fea_out, size=([h, w]), mode='bilinear')
        ir_fea_out = F.interpolate(ir_fea_out, size=([h, w]), mode='bilinear')

        return rgb_fea_out, ir_fea_out

class CFT(nn.Module):
    def __init__(self, num_classes=2, pretrained=False, backbone='vgg', in_channels=3):
        super(CFT, self).__init__()
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
        
        # self.concat5 = conv1x1(2*filters[4], filters[4])
        # self.concat4 = conv1x1(2*filters[3], filters[3])
        # self.concat3 = conv1x1(2*filters[2], filters[2])
        # self.concat2 = conv1x1(2*filters[1], filters[1])
        # self.concat1 = conv1x1(2*filters[0], filters[0])
        
        self.cft5 = GPT(filters[4], n_layer=2, attn_pdrop=0, resid_pdrop=0, embd_pdrop=0, vert_anchors=16, horz_anchors=16)
        self.cft4 = GPT(filters[3], n_layer=2, attn_pdrop=0, resid_pdrop=0, embd_pdrop=0, vert_anchors=16, horz_anchors=16)
        self.cft3 = GPT(filters[2], n_layer=2, attn_pdrop=0, resid_pdrop=0, embd_pdrop=0, vert_anchors=16, horz_anchors=16)
        self.cft2 = GPT(filters[1], n_layer=2, attn_pdrop=0, resid_pdrop=0, embd_pdrop=0, vert_anchors=16, horz_anchors=16)
        self.cft1 = GPT(filters[0], n_layer=2, attn_pdrop=0, resid_pdrop=0, embd_pdrop=0, vert_anchors=16, horz_anchors=16)
        
        self.up_concat4 = unetUp(filters[4] * 2 + filters[3] * 2, out_filters[3])
        self.up_concat3 = unetUp(filters[2] * 2 + out_filters[3], out_filters[2])
        self.up_concat2 = unetUp(filters[1] * 2 + out_filters[2], out_filters[1])
        self.up_concat1 = unetUp(filters[0] * 2 + out_filters[1], out_filters[0])
        
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
        
        # up4 = self.up_concat4(self.concat4(torch.cat(self.cft4([spec_feat4, rgb_feat4]), dim=1)),
        #                       self.concat5(torch.cat(self.cft5([spec_feat5, rgb_feat5]), dim=1)))
        # up3 = self.up_concat3(self.concat3(torch.cat(self.cft3([spec_feat3, rgb_feat3]), dim=1)), up4)
        # up2 = self.up_concat2(self.concat2(torch.cat(self.cft2([spec_feat2, rgb_feat2]), dim=1)), up3)
        # up1 = self.up_concat1(self.concat1(torch.cat(self.cft1([spec_feat1, rgb_feat1]), dim=1)), up2)
        
        up4 = self.up_concat4(torch.cat(self.cft4([spec_feat4, rgb_feat4]), dim=1),
                              torch.cat(self.cft5([spec_feat5, rgb_feat5]), dim=1))
        up3 = self.up_concat3(torch.cat(self.cft3([spec_feat3, rgb_feat3]), dim=1), up4)
        up2 = self.up_concat2(torch.cat(self.cft2([spec_feat2, rgb_feat2]), dim=1), up3)
        up1 = self.up_concat1(torch.cat(self.cft1([spec_feat1, rgb_feat1]), dim=1), up2)
        
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
    

if __name__ == '__main__':
    # img1 = torch.randn([1,128,256,256])
    # img2 = torch.randn([1,128,256,256])
    # img = [img1,img2]
    # print(img[0].shape)
    # cft = GPT(128)
    # I , J = cft(img)
    # print(I.shape)
    
    spec = torch.randn(1, 25, 416, 416)
    rgb  = torch.randn(1, 3, 416, 416)
    model = CFT(11, True, 'resnet18')
    output = model(spec, rgb)
    print(output.shape)