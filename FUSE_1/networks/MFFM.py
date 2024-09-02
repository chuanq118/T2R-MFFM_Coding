import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from FUSE_1.networks.basic_blocks import DecoderBlock


# 道路特征阈融合模块
class PixelWiseDFF(nn.Module):
    def __init__(self, num_channels, init_weight1=0.5, init_weight2=0.5):
        """
        初始化 PixelWiseDFF 模块
        参数:
        - num_channels: 输入 遥感/轨迹 特征图的通道数
        """
        super(PixelWiseDFF, self).__init__()
        # 定义两个可学习的权重参数，形状与输入特征图通道数匹配
        self.weight1 = nn.Parameter(torch.full((1, num_channels, 1, 1), init_weight1))
        self.weight2 = nn.Parameter(torch.full((1, num_channels, 1, 1), init_weight2))

    def forward(self, gpschan_features, imgchan_features):
        """
        前向传播
        - gpschan_features: GPS 输入特征图1，形状为 (N, C, H, W)
        - imgchan_features: 遥感 输入特征图2，形状为 (N, C, H, W)
        返回:
        - output: 逐个像素融合后的特征图，形状为 (N, C, H, W)
        """
        # 确保输入特征图的形状是正确的
        assert gpschan_features.shape == imgchan_features.shape, "Input feature maps must have the same shape"
        # 逐像素融合处理,通道数保持不变
        output = self.weight1 * gpschan_features + self.weight2 * imgchan_features
        return output


# 多尺度特征融合模块
class MFFM(nn.Module):
    # 特征 shape channel_nums
    #    512 -> 64 64
    def __init__(self, filters, out_channels, shape_n=64):
        super().__init__()
        # 输出通道数目 对应参数 C
        self.out_channels = out_channels
        # 将 shape 统一, 对应参数 N
        self.shape_n = shape_n
        # 创建对应的四个卷积层将输出通道进行统一
        self.mff_conv2d_1 = nn.Conv2d(filters[0], out_channels,
                                      kernel_size=7, stride=2, padding=3, bias=False)
        self.mff_conv2d_2 = nn.Conv2d(filters[1], out_channels,
                                      kernel_size=5, stride=2, padding=2, bias=False)
        self.mff_conv2d_3 = nn.Conv2d(filters[2], out_channels,
                                      kernel_size=3, stride=2, padding=1, bias=False)
        self.mff_conv2d_4 = nn.Conv2d(filters[3], out_channels,
                                      kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, fused_feature1, fused_feature2, fused_feature3, fused_feature4):
        # 四个融合道路特征块 统一 channel
        fused_feature1 = self.mff_conv2d_1(fused_feature1)
        fused_feature2 = self.mff_conv2d_2(fused_feature2)
        fused_feature3 = self.mff_conv2d_3(fused_feature3)
        fused_feature4 = self.mff_conv2d_4(fused_feature4)
        # reshape
        reshaped_ff1 = F.interpolate(fused_feature1, size=(self.shape_n, self.shape_n), mode='bilinear',
                                     align_corners=False)
        reshaped_ff2 = F.interpolate(fused_feature2, size=(self.shape_n, self.shape_n), mode='bilinear',
                                     align_corners=False)
        reshaped_ff3 = F.interpolate(fused_feature3, size=(self.shape_n, self.shape_n), mode='bilinear',
                                     align_corners=False)
        reshaped_ff4 = fused_feature4
        # fused_feature4 一般不需要调整，默认已经是 64 x 64
        if self.shape_n != 64:
            reshaped_ff4 = F.interpolate(fused_feature4, size=(self.shape_n, self.shape_n), mode='bilinear',
                                         align_corners=False)
        # 计算注意力参数矩阵
        # 计算点乘
        attention_scores = reshaped_ff1 * reshaped_ff2 * reshaped_ff3 * reshaped_ff4
        # 计算指数
        exp_attention_scores = torch.exp(attention_scores)
        # 计算归一化因子
        sum_exp_attention_scores = torch.sum(exp_attention_scores.view(exp_attention_scores.size(0), -1), dim=1,
                                             keepdim=True)
        # 计算注意力矩阵 A
        attention_matrix = exp_attention_scores / sum_exp_attention_scores.view(-1, 1, 1, 1)
        # 计算新的特征矩阵块
        weighted_ff4 = attention_matrix * reshaped_ff4
        # 加上原始的 reshaped_ff4，形成残差连接，作为最终的道路特征输出
        return weighted_ff4 + reshaped_ff4


# 道路多尺度特征融合网络
class MFFRNet(nn.Module):

    def __init__(self, num_classes=1, num_channels=3):
        """
        initialization
        :param num_classes:
        :param num_channels: number of input channels
        """
        super().__init__()

        # 初始化一些参数
        filters = [64, 128, 256, 512]
        self.num_channels = num_channels

        # load pretrained resnet34 models
        imgchan_resnet = models.resnet34(pretrained=True)

        # 初步道路生成模块
        # 在输入通道数等于或大于 3 时，使用ResNet34模型的第一个卷积层的原因是
        # 因为这个模型是在ImageNet数据集上预训练的，该数据集的输入图像是RGB格式的，
        # 也就是说有三个通道。预训练模型的第一层卷积层已经在大量的图像上学习到了
        # 一些基本而通用的图像特征（例如边缘、颜色、纹理等）。如果我们的输入图像也是3通道的，
        # 那么可以直接利用这些已经学习到的特征，这将加速我们的模型训练并且可能提升模型性能。
        # 相反，如果输入通道数少于3，那么我们不能直接使用预训练的卷积层，
        # 因为它们的输入通道数不匹配。这种情况下，我们创建一个新的卷积层来从头开始学习特征。
        if num_channels < 3:
            self.first_conv_imgchan = nn.Conv2d(num_channels, filters[0],
                                                kernel_size=7, stride=2, padding=3, bias=False)
        else:
            # resnet.conv1 => (输入通道数：3, 输出通道数：64, 卷积核大小：7x7, 步幅：2x2, 填充：3x3, bias=False)
            self.first_conv_imgchan = imgchan_resnet.conv1

        # -> BN -> Relu -> MaxPool
        self.first_bn_imgchan = imgchan_resnet.bn1
        self.first_relu_imgchan = imgchan_resnet.relu
        self.first_maxpool_imgchan = imgchan_resnet.maxpool

        # 遥感道路编码通道
        # resnet34: layers=[3, 4, 6, 3]
        # resnet34 的结构具体参考 https://towardsdatascience.com/understanding-and-visualizing-resnets-442284831be8
        # 第一遥感道路编码块. 提取图像的一些低层次的图像特征（例如边缘、线条和颜色块）
        # 输出特征图大小不变，通道数变为64
        # layer1 = (block, 64, layers[0])
        self.image_encoder1 = imgchan_resnet.layer1
        # 第二遥感道路编码块. 进一步提取图像的特征并降低空间大小
        # 输出特征图的大小变为原来的一半，通道数变为128
        # layer2 = (block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.image_encoder2 = imgchan_resnet.layer2
        # 第三遥感道路编码块. 进一步提取高层次的特征并降低空间大小
        # 输出特征图的大小变为原来的一半，通道数变为256
        self.image_encoder3 = imgchan_resnet.layer3
        # 第四遥感道路编码块. 进一步提取更高层次的特征信息,更强调是否为背景/道路区域.
        # 输出特征图的大小再次变为原来的一半，通道数变为512
        self.image_encoder4 = imgchan_resnet.layer4

        # GPS轨迹数据道路编码通道

        # GPS通道 初步道路生成模块
        gpschan_resnet = models.resnet34(pretrained=True)
        if num_channels < 3:
            self.first_conv_gpschan = nn.Conv2d(num_channels, filters[0],
                                                kernel_size=7, stride=2, padding=3, bias=False)
        else:
            # resnet.conv1 => (输入通道数：3, 输出通道数：64, 卷积核大小：7x7, 步幅：2x2, 填充：3x3, bias=False)
            self.first_conv_gpschan = gpschan_resnet.conv1
        # -> BN -> Relu -> MaxPool
        self.first_bn_gpschan = gpschan_resnet.bn1
        self.first_relu_gpschan = gpschan_resnet.relu
        self.first_maxpool_gpschan = gpschan_resnet.maxpool

        # 第一轨迹道路特征编码块
        self.gpschan_encoder1 = gpschan_resnet.layer1
        # 第二轨迹道路特征编码块
        # 输出特征图的大小变为原来的一半，通道数变为128
        self.gpschan_encoder2 = gpschan_resnet.layer2
        # 第三轨迹道路特征编码块
        # 输出特征图的大小变为原来的一半，通道数变为256
        self.gpschan_encoder3 = gpschan_resnet.layer3
        # 第四轨迹道路特征编码块.
        # 输出特征图的大小再次变为原来的一半，通道数变为512
        self.gpschan_encoder4 = gpschan_resnet.layer4

        # 对应四个遥感\轨迹道路特征编码块的 像素级道路特征阈融合模块
        self.pixel_wise_dff_1 = PixelWiseDFF(num_channels=filters[0])
        self.pixel_wise_dff_2 = PixelWiseDFF(num_channels=filters[1])
        self.pixel_wise_dff_3 = PixelWiseDFF(num_channels=filters[2], init_weight1=0.6, init_weight2=0.4)
        self.pixel_wise_dff_4 = PixelWiseDFF(num_channels=filters[3], init_weight1=0.8, init_weight2=0.2)

        # 创建多尺度融合模块
        self.mff_module = MFFM(filters=filters, out_channels=512)

        # 道路解码块
        self.decoderBlock = DecoderBlock
        # 第四道路特征解码块
        self.decoder4 = self.decoderBlock(filters[3], filters[2])
        # 第三道路特征解码块
        self.decoder3 = self.decoderBlock(filters[2], filters[1])
        # 第二道路特征解码块
        self.decoder2 = self.decoderBlock(filters[1], filters[0])
        # 第一道路特征解码块
        self.decoder1 = self.decoderBlock(filters[0], filters[0])

    def forward(self, img_features, gps_features):

        # ----------------------------
        # 1. 初步道路生成模块 (Initial Road Generation Modules)
        # ----------------------------

        # 图像通道 (Image Channel)
        # img_features 通过初始的卷积层 -> BN -> ReLU -> MaxPool
        img_features = self.first_conv_imgchan(img_features)
        img_features = self.first_bn_imgchan(img_features)
        img_features = self.first_relu_imgchan(img_features)
        img_features = self.first_maxpool_imgchan(img_features)

        # GPS通道 (GPS Channel)
        # gps_features 通过初始的卷积层 -> BN -> ReLU -> MaxPool
        gps_features = self.first_conv_gpschan(gps_features)
        gps_features = self.first_bn_gpschan(gps_features)
        gps_features = self.first_relu_gpschan(gps_features)
        gps_features = self.first_maxpool_gpschan(gps_features)

        # ----------------------------
        # 2. 遥感与GPS的道路特征编码 (Road Feature Encoding for Image and GPS Channels)
        # ----------------------------

        # 遥感通道编码块 (Image Channel Encoding Blocks)
        img_encoded1 = self.image_encoder1(img_features)  # 第一层编码 (64 channels)
        img_encoded2 = self.image_encoder2(img_encoded1)  # 第二层编码 (128 channels)
        img_encoded3 = self.image_encoder3(img_encoded2)  # 第三层编码 (256 channels)
        img_encoded4 = self.image_encoder4(img_encoded3)  # 第四层编码 (512 channels)

        # 轨迹点通道编码块 (GPS Channel Encoding Blocks)
        gps_encoded1 = self.gpschan_encoder1(gps_features)  # 第一层编码 (64 channels)
        gps_encoded2 = self.gpschan_encoder2(gps_encoded1)  # 第二层编码 (128 channels)
        gps_encoded3 = self.gpschan_encoder3(gps_encoded2)  # 第三层编码 (256 channels)
        gps_encoded4 = self.gpschan_encoder4(gps_encoded3)  # 第四层编码 (512 channels)

        # ----------------------------
        # 3. 像素级阈融合道路特征 (Pixel-wise Road Feature Fusion)
        # ----------------------------

        # 通过像素级融合模块 (PixelWiseDFF) 融合图像和GPS的道路特征
        fused_feature1 = self.pixel_wise_dff_1(img_encoded1, gps_encoded1)  # 第一层融合 (64 channels)
        fused_feature2 = self.pixel_wise_dff_2(img_encoded2, gps_encoded2)  # 第二层融合 (128 channels)
        fused_feature3 = self.pixel_wise_dff_3(img_encoded3, gps_encoded3)  # 第三层融合 (256 channels)
        fused_feature4 = self.pixel_wise_dff_4(img_encoded4, gps_encoded4)  # 第四层融合 (512 channels)

        # ----------------------------
        # 4. 多尺度特征融合 (Multi-Scale Feature Fusion)
        # ----------------------------
        # TODO 修改 MFFM (Multi-Scale Feature Fusion Module) 名称
        # 通过多尺度融合模块 (MFFM) 融合四个层级的特征
        mff_feature = self.mff_module(fused_feature1, fused_feature2, fused_feature3, fused_feature4)

        # ----------------------------
        # 5. 道路解码块 (Road Decoding Blocks)
        # ----------------------------

        # 通过解码块逐步解码融合后的特征
        # 第四层解码 (512 -> 256 channels)
        decoded4 = self.decoder4(mff_feature)
        # 第三层解码 (256 -> 128 channels)
        decoded3 = self.decoder3(decoded4)
        # 第二层解码 (128 -> 64 channels)
        decoded2 = self.decoder2(decoded3)
        # 第一层解码 (64 -> 64 channels)
        decoded1 = self.decoder1(decoded2)
        # TODO: 再加上一层反卷积
        final_output = decoded1
        return final_output
