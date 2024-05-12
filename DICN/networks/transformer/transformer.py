import math
import torch.nn as nn
import torch.nn.functional as F
from .basic_blocks import *
from torchvision import models
from functools import partial


class OneHot(nn.Module):
    def __init__(self, depth):
        super(OneHot, self).__init__()
        emb = nn.Embedding(depth, depth)
        emb.weight.data = torch.eye(depth)
        emb.weight.requires_grad = False
        self.emb = emb

    def forward(self, input_):
        return self.emb(input_)


class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, mask):
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class SelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        assert n_embd % n_head == 0

        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)

        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class CrossAttention(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x, x_value):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        q = self.query(x_value).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, block_exp, attn_pdrop, resid_pdrop):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, block_exp * n_embd),
            nn.ReLU(True), # changed from GELU
            nn.Linear(block_exp * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )
        self.ln3 = nn.LayerNorm(n_embd)
        self.ln4 = nn.LayerNorm(n_embd)
        self.ln5 = nn.LayerNorm(n_embd)
        self.mlp2 = nn.Sequential(
            nn.Linear(n_embd, block_exp * n_embd),
            nn.ReLU(True),  # changed from GELU
            nn.Linear(block_exp * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )
        self.crossattn = CrossAttention(n_embd, n_head, attn_pdrop, resid_pdrop)

    def forward(self, x):
        x_value = x
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        x = x + self.crossattn(self.ln3(x), self.ln4(x_value))
        x = x + self.mlp2(self.ln5(x))

        return x


class SAFM(nn.Module):
    def __init__(self, n_embd, n_head, block_exp, n_layer,
                 img_vert_anchors, img_horz_anchors,
                 trj_vert_anchors, trj_horz_anchors,
                 embd_pdrop, attn_pdrop, resid_pdrop):
        super().__init__()
        self.n_embd = n_embd

        self.img_vert_anchors = img_vert_anchors
        self.img_horz_anchors = img_horz_anchors
        self.trj_vert_anchors = trj_vert_anchors
        self.trj_horz_anchors = trj_horz_anchors


        self.pos_emb = nn.Parameter(torch.zeros(1,
                    img_vert_anchors * img_horz_anchors + trj_vert_anchors * trj_horz_anchors, n_embd))


        self.drop = nn.Dropout(embd_pdrop)


        self.blocks = nn.Sequential(*[Block(n_embd, n_head,
                                            block_exp, attn_pdrop, resid_pdrop)
                                      for layer in range(n_layer)])

        self.ln_f = nn.LayerNorm(n_embd)

        self.block_size = 1

        self.apply(self._reset_parameters)

    def _reset_parameters(self, module):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)


    def forward(self, image_tensor, trj_tensor, placeholder=False):
        bz = trj_tensor.shape[0]
        trj_h, trj_w = trj_tensor.shape[2:4]
        img_h, img_w = image_tensor.shape[2:4]

        image_tensor = image_tensor.view(bz, 1, -1, img_h, img_w).permute(0, 1, 3, 4, 2).contiguous().view(
            bz, -1, self.n_embd)
        trj_tensor = trj_tensor.view(bz, 1, -1, trj_h, trj_w).permute(0, 1, 3, 4,
                         2).contiguous().view(bz, -1,self.n_embd)


        token_embeddings = torch.cat((image_tensor, trj_tensor), dim=1)

        x = self.drop(self.pos_emb + token_embeddings)
        x = self.blocks(x)  # (B, an * T, C)
        x = self.ln_f(x)  # (B, an * T, C)

        x = x.view(bz,
                   self.img_vert_anchors * self.img_horz_anchors +   self.trj_vert_anchors * self.trj_horz_anchors,
                   self.n_embd)

        image_tensor_out = x[:, : self.img_vert_anchors * self.img_horz_anchors, :].contiguous().view(
            bz, -1, img_h, img_w)
        trj_tensor_out = x[:,  self.trj_vert_anchors * self.trj_horz_anchors:, :].contiguous().view(
            bz, -1, trj_h, trj_w)

        return image_tensor_out, trj_tensor_out

nonlinearity = partial(F.relu, inplace=False)
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity
        self.deconv2 = nn.ConvTranspose2d(
            in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1
        )
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

class ProjectionHead(nn.Module):
    def __init__(self, dim_in, proj_dim=256, proj='convmlp', bn_type='torchsyncbn'):
        super(ProjectionHead, self).__init__()

        if proj == 'linear':
            self.proj = nn.Conv2d(dim_in, proj_dim, kernel_size=1)
        elif proj == 'convmlp':
            self.proj = nn.Sequential(
                nn.Conv2d(dim_in, dim_in, kernel_size=1),
                nn.BatchNorm2d(dim_in),
                nn.ReLU(),
                nn.Conv2d(dim_in, proj_dim, kernel_size=1)
            )

    def forward(self, x):
        return F.normalize(self.proj(x), p=2, dim=1)

class Edge_Module(nn.Module):
    """
    Edge Learning Branch
    """
    def __init__(self, in_fea=[128, 256, 512], mid_fea=256, out_fea=1):
        super(Edge_Module, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_fea[0], mid_fea, kernel_size=1, padding=0, dilation=1, bias=False),
            # nn.BatchNorm2d(mid_fea),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_fea[1], mid_fea, kernel_size=1, padding=0, dilation=1, bias=False),
            # nn.BatchNorm2d(mid_fea),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_fea[2], mid_fea, kernel_size=1, padding=0, dilation=1, bias=False),
            # nn.BatchNorm2d(mid_fea),
            nn.ReLU()
        )
        self.conv4 = nn.Conv2d(mid_fea, out_fea, kernel_size=3, padding=1, dilation=1, bias=True)
        self.conv5 = nn.Conv2d(out_fea * 3, out_fea, kernel_size=1, padding=0, dilation=1, bias=True)

    def forward(self, x1, x2, x3):
        _, _, h, w = x1.size()

        edge1_fea = self.conv1(x1)
        edge1 = self.conv4(edge1_fea)
        edge2_fea = self.conv2(x2)
        edge2 = self.conv4(edge2_fea)
        edge3_fea = self.conv3(x3)
        edge3 = self.conv4(edge3_fea)

        edge2 = F.interpolate(edge2, size=(h, w), mode='bilinear', align_corners=True)
        edge3 = F.interpolate(edge3, size=(h, w), mode='bilinear', align_corners=True)

        edge = torch.cat([edge1, edge2, edge3], dim=1)

        edge = self.conv5(edge)

        return edge, None

class SegTrans(nn.Module):
    def __init__(self, num_classes=1, num_channels=3, encoder_1dconv=0,  decoder_1dconv=0, pretrain=True):
        super().__init__()

        # 遥感图像的ResNet编码器
        filters = [64, 128, 256, 512]
        self.num_channels = num_channels
        resnet = models.resnet34(pretrained=True)
        if num_channels < 3:
            self.firstconv = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3,
                                       bias=False)
        else:
            self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool

        if encoder_1dconv == 0:
            self.encoder1 = resnet.layer1
            self.encoder2 = resnet.layer2
            self.encoder3 = resnet.layer3
            self.encoder4 = resnet.layer4
        else:
            myresnet = ResnetBlock()
            layers = [3, 4, 6, 3]
            basicBlock = BasicBlock1DConv
            self.encoder1 = myresnet._make_layer(basicBlock, 64, layers[0])
            self.encoder2 = myresnet._make_layer(
                basicBlock, 128, layers[1], stride=2)
            self.encoder3 = myresnet._make_layer(
                basicBlock, 256, layers[2], stride=2)
            self.encoder4 = myresnet._make_layer(
                basicBlock, 512, layers[3], stride=2)

        # GPS特征的ResNet编码器 self.firstconv_gpscnn、self.firstbn_gpscnn、self.firstrelu_gpscnn 和
        # self.firstmaxpool_gpscnn是对应于GPS特征的第一层卷积、批归一化、激活函数和最大池化层
        # self.encoder1_gpscnn、self.encoder2_gpscnn、self.encoder3_gpscnn 和 self.encoder4_gpscnn是GPS特征提取的四个编码器层
        gpscnn = models.resnet34(pretrained=True)
        self.num_channels_gpscnn = 3
        if self.num_channels_gpscnn < 3:
            self.firstconv_gpscnn = nn.Conv2d(self.num_channels_gpscnn, 64, kernel_size=7, stride=2, padding=3,
                                              bias=False)
        else:
            self.firstconv_gpscnn = gpscnn.conv1
        self.firstbn_gpscnn = gpscnn.bn1
        self.firstrelu_gpscnn = gpscnn.relu
        self.firstmaxpool_gpscnn = gpscnn.maxpool
        self.encoder1_gpscnn = gpscnn.layer1
        self.encoder2_gpscnn = gpscnn.layer2
        self.encoder3_gpscnn = gpscnn.layer3
        self.encoder4_gpscnn = gpscnn.layer4


        if decoder_1dconv == 0:
            self.decoder = DecoderBlock
        elif decoder_1dconv == 2:
            self.decoder = DecoderBlock1DConv2
        elif decoder_1dconv == 4:
            self.decoder = DecoderBlock1DConv4

        self.decoder4 = self.decoder(filters[3], filters[2])
        self.decoder3 = self.decoder(filters[2], filters[1])
        self.decoder2 = self.decoder(filters[1], filters[0])
        self.decoder1 = self.decoder(filters[0], filters[0])

        self.decoder_gps = DecoderBlock
        self.decoder4_gps = self.decoder_gps(filters[3], filters[2])
        self.decoder3_gps = self.decoder_gps(filters[2], filters[1])
        self.decoder2_gps = self.decoder_gps(filters[1], filters[0])
        self.decoder1_gps = self.decoder_gps(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0] * 2, 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)
        self.proj_head = ProjectionHead(dim_in=32, proj_dim=32)
        if self.num_channels > 3:
            self.addconv = nn.Conv2d(
                self.num_channels - 3, 64, kernel_size=7, stride=2, padding=3)

        self.edge = Edge_Module()

        self.avgpool_img = nn.AdaptiveAvgPool2d((32, 32))
        self.avgpool_gps = nn.AdaptiveAvgPool2d((32, 32))
        self.input_chs = [64, 128, 256, 512]
        self.transformer1 = SAFM(n_embd=self.input_chs[0],
                                n_head=4,  
                                block_exp=4,  
                                n_layer=1,  
                                img_vert_anchors=32,  
                                img_horz_anchors=32,  
                                trj_vert_anchors=32,  
                                trj_horz_anchors=32,  
                                embd_pdrop=0.1,  
                                attn_pdrop=0.1,  
                                resid_pdrop=0.1,  
                                )
        self.transformer2 = SAFM(n_embd=self.input_chs[1], 
                                 n_head=4,  
                                 block_exp=4,  
                                 n_layer=1,  
                                 img_vert_anchors=32,  
                                 img_horz_anchors=32,  
                                 trj_vert_anchors=32,  
                                 trj_horz_anchors=32,  
                                 embd_pdrop=0.1,  
                                 attn_pdrop=0.1,  
                                 resid_pdrop=0.1,  
                                 )
        self.transformer3 = SAFM(n_embd=self.input_chs[2], 
                                 n_head=4,  
                                 block_exp=4,  
                                 n_layer=1,  
                                 img_vert_anchors=32,  
                                 img_horz_anchors=32,  
                                 trj_vert_anchors=32,  
                                 trj_horz_anchors=32,  
                                 embd_pdrop=0.1,  
                                 attn_pdrop=0.1,  
                                 resid_pdrop=0.1,  
                                 )
        self.transformer4 = SAFM(n_embd=self.input_chs[3], 
                                 n_head=4,  
                                 block_exp=4,  
                                 n_layer=1,  
                                 img_vert_anchors=32,  
                                 img_horz_anchors=32,  
                                 trj_vert_anchors=32,  
                                 trj_horz_anchors=32,  
                                 embd_pdrop=0.1,  
                                 attn_pdrop=0.1,  
                                 resid_pdrop=0.1,  
                                 )
        self.transformer5 = SAFM(n_embd=self.input_chs[2], 
                                 n_head=4,  
                                 block_exp=4,  
                                 n_layer=1,  
                                 img_vert_anchors=32,  
                                 img_horz_anchors=32,  
                                 trj_vert_anchors=32,  
                                 trj_horz_anchors=32,  
                                 embd_pdrop=0.1,  
                                 attn_pdrop=0.1,  
                                 resid_pdrop=0.1,  
                                 )
        self.transformer6 = SAFM(n_embd=self.input_chs[1], 
                                 n_head=4,  
                                 block_exp=4,  
                                 n_layer=1,  
                                 img_vert_anchors=32,  
                                 img_horz_anchors=32,  
                                 trj_vert_anchors=32,  
                                 trj_horz_anchors=32,  
                                 embd_pdrop=0.1,  
                                 attn_pdrop=0.1,  
                                 resid_pdrop=0.1,  
                                 )
        self.transformer7 = SAFM(n_embd=self.input_chs[0], 
                                 n_head=4,  
                                 block_exp=4,  
                                 n_layer=1,  
                                 img_vert_anchors=32,  
                                 img_horz_anchors=32,  
                                 trj_vert_anchors=32,  
                                 trj_horz_anchors=32,  
                                 embd_pdrop=0.1,  
                                 attn_pdrop=0.1,  
                                 resid_pdrop=0.1,  
                                 )
        self.transformer8 = SAFM(n_embd=self.input_chs[0], 
                                 n_head=4,  
                                 block_exp=4,  
                                 n_layer=1,  
                                 img_vert_anchors=32,  
                                 img_horz_anchors=32,  
                                 trj_vert_anchors=32,  
                                 trj_horz_anchors=32,  
                                 embd_pdrop=0.1,  
                                 attn_pdrop=0.1,  
                                 resid_pdrop=0.1,  
                                 )

    def forward(self, input_):
        # channels = input_.shape[1]
        # if channels == 4:
        #     gps_feature = input_[:, 3, :, :].unsqueeze(1)
        #     gps_feature = gps_feature.repeat(1, 3, 1, 1)
        # elif channels ==6:
        #     gps_feature = input_[:, 3:, :, :]
        # x = input_[:, :3, :, :]

        # 获取输入数据的通道数
        channels = input_.shape[1]

        # 如果通道数为4，说明只有一个GPS特征通道
        if channels == 4:
            # 提取第四个通道作为GPS特征，并增加一个维度
            gps_feature = input_[:, 3, :, :].unsqueeze(1)
            # 将GPS特征复制三次，以匹配其他特征的维度
            gps_feature = gps_feature.repeat(1, 3, 1, 1)
        # 如果通道数为6，说明有多个GPS特征通道
        elif channels == 6:
            # 直接提取最后三个通道作为GPS特征
            gps_feature = input_[:, 3:, :, :]

        # 提取前三个通道作为其他特征
        x = input_[:, :3, :, :]

        # Encoder
        if self.num_channels > 3:
            add = self.addconv(x.narrow(1, 3, self.num_channels - 3))
            x = self.firstconv(x.narrow(1, 0, 3))
            x = x + add
        else:
            x = self.firstconv(x)

        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        gps_feature = self.firstconv_gpscnn(gps_feature)
        gps_feature = self.firstbn_gpscnn(gps_feature)
        gps_feature = self.firstrelu_gpscnn(gps_feature)
        gps_feature = self.firstmaxpool_gpscnn(gps_feature)

        e1 = self.encoder1(x)
        e1_gps = self.encoder1_gpscnn(gps_feature)
        e1_trans = self.avgpool_img(e1)
        e1_gps_trans = self.avgpool_gps(e1_gps)
        e1_trans, e1_gps_trans = self.transformer1(e1_trans, e1_gps_trans, False)
        e1_trans = F.interpolate(e1_trans, size=(e1.shape[2], e1.shape[3]), mode='bilinear', align_corners=False)
        e1_gps_trans = F.interpolate(e1_gps_trans, size=(e1_gps.shape[2], e1_gps.shape[3]), mode='bilinear', align_corners=False)
        e1 = e1 + e1_trans
        e1_gps = e1_gps + e1_gps_trans

        e2 = self.encoder2(e1)
        e2_gps = self.encoder2_gpscnn(e1_gps)
        e2_trans = self.avgpool_img(e2)
        e2_gps_trans = self.avgpool_gps(e2_gps)
        e2_trans, e2_gps_trans = self.transformer2(e2_trans, e2_gps_trans, False)
        e2_trans = F.interpolate(e2_trans, size=(e2.shape[2], e2.shape[3]), mode='bilinear', align_corners=False)
        e2_gps_trans = F.interpolate(e2_gps_trans, size=(e2_gps.shape[2], e2_gps.shape[3]), mode='bilinear', align_corners=False)
        e2 = e2 + e2_trans
        e2_gps = e2_gps + e2_gps_trans

        e3 = self.encoder3(e2)
        e3_gps = self.encoder3_gpscnn(e2_gps)
        e3_trans = self.avgpool_img(e3)
        e3_gps_trans = self.avgpool_gps(e3_gps)
        e3_trans, e3_gps_trans = self.transformer3(e3_trans, e3_gps_trans, False)
        e3_trans = F.interpolate(e3_trans, size=(e3.shape[2], e3.shape[3]), mode='bilinear', align_corners=False)
        e3_gps_trans = F.interpolate(e3_gps_trans, size=(e3_gps.shape[2], e3_gps.shape[3]), mode='bilinear', align_corners=False)
        e3 = e3 + e3_trans
        e3_gps = e3_gps + e3_gps_trans

        e4 = self.encoder4(e3)
        e4_gps = self.encoder4_gpscnn(e3_gps)
        e4_trans = self.avgpool_img(e4)
        e4_gps_trans = self.avgpool_gps(e4_gps)

        e4_trans, e4_gps_trans = self.transformer4(e4_trans, e4_gps_trans, False)
        e4_trans = F.interpolate(e4_trans, size=(e4.shape[2], e4.shape[3]), mode='bilinear', align_corners=False)
        e4_gps_trans = F.interpolate(e4_gps_trans, size=(e4_gps.shape[2], e4_gps.shape[3]), mode='bilinear', align_corners=False)
        e4 = e4 + e4_trans
        e4_gps = e4_gps + e4_gps_trans

        # # Decoder
        d4 = self.decoder4(e4) + e3
        d4_gps = self.decoder4_gps(e4_gps) + e3_gps
        d4_trans = self.avgpool_img(d4)
        d4_gps_trans = self.avgpool_gps(d4_gps)
        d4_trans, d4_gps_trans = self.transformer5(d4_trans, d4_gps_trans, False)
        d4_trans = F.interpolate(d4_trans, size=(d4.shape[2], d4.shape[3]), mode='bilinear', align_corners=False)
        d4_gps_trans = F.interpolate(d4_gps_trans, size=(d4_gps.shape[2], d4_gps.shape[3]), mode='bilinear', align_corners=False)
        d4 = d4 + d4_trans
        d4_gps = d4_gps + d4_gps_trans

        d3 = self.decoder3(d4)   + e2
        d3_gps = self.decoder3_gps(d4_gps) + e2_gps
        d3_trans = self.avgpool_img(d3)
        d3_gps_trans = self.avgpool_gps(d3_gps)
        d3_trans, d3_gps_trans = self.transformer6(d3_trans, d3_gps_trans, False)
        d3_trans = F.interpolate(d3_trans, size=(d3.shape[2], d3.shape[3]), mode='bilinear', align_corners=False)
        d3_gps_trans = F.interpolate(d3_gps_trans, size=(d3_gps.shape[2], d3_gps.shape[3]), mode='bilinear', align_corners=False)
        d3 = d3 + d3_trans
        d3_gps = d3_gps + d3_gps_trans

        d2 = self.decoder2(d3)    + e1
        d2_gps = self.decoder2_gps(d3_gps)   + e1_gps
        d2_trans = self.avgpool_img(d2)
        d2_gps_trans = self.avgpool_gps(d2_gps)
        d2_trans, d2_gps_trans = self.transformer7(d2_trans, d2_gps_trans, False)
        d2_trans = F.interpolate(d2_trans, size=(d2.shape[2], d2.shape[3]), mode='bilinear', align_corners=False)
        d2_gps_trans = F.interpolate(d2_gps_trans, size=(d2_gps.shape[2], d2_gps.shape[3]), mode='bilinear', align_corners=False)
        d2 = d2 + d2_trans
        d2_gps = d2_gps + d2_gps_trans

        d1 = self.decoder1(d2)
        d1_gps = self.decoder1_gps(d2_gps)
        d1_trans = self.avgpool_img(d1)
        d1_gps_trans = self.avgpool_gps(d1_gps)
        d1_trans, d1_gps_trans = self.transformer8(d1_trans, d1_gps_trans, False)
        d1_trans = F.interpolate(d1_trans, size=(d1.shape[2], d1.shape[3]), mode='bilinear', align_corners=False)
        d1_gps_trans = F.interpolate(d1_gps_trans, size=(d1_gps.shape[2], d1_gps.shape[3]), mode='bilinear', align_corners=False)
        d1 = d1 + d1_trans
        d1_gps = d1_gps + d1_gps_trans

        edge_result, edge_fea = self.edge(e2, e3, e4)

        out = self.finaldeconv1(torch.concat((d1, d1_gps), dim = 1))
        final_feat = self.finalrelu1(out)
        porj_final_feat = self.proj_head(final_feat)
        out = self.finalconv2(final_feat)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return [torch.sigmoid(out), torch.sigmoid(edge_result), porj_final_feat]

