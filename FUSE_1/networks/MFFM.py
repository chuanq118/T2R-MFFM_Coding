import torch.nn as nn
from torchvision import models


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
        resnet = models.resnet34(pretrained=True)

        # 初步道路生成模块
        # 在输入通道数等于或大于 3 时，使用ResNet34模型的第一个卷积层的原因是
        # 因为这个模型是在ImageNet数据集上预训练的，该数据集的输入图像是RGB格式的，
        # 也就是说有三个通道。预训练模型的第一层卷积层已经在大量的图像上学习到了
        # 一些基本而通用的图像特征（例如边缘、颜色、纹理等）。如果我们的输入图像也是3通道的，
        # 那么可以直接利用这些已经学习到的特征，这将加速我们的模型训练并且可能提升模型性能。
        # 相反，如果输入通道数少于3，那么我们不能直接使用预训练的卷积层，
        # 因为它们的输入通道数不匹配。这种情况下，我们创建一个新的卷积层来从头开始学习特征。
        if num_channels < 3:
            self.first_conv = nn.Conv2d(num_channels, filters[0],
                                        kernel_size=7, stride=2, padding=3, bias=False)
        else:
            # resnet.conv1 => (输入通道数：3, 输出通道数：64, 卷积核大小：7x7, 步幅：2x2, 填充：3x3, bias=False)
            self.first_conv = resnet.conv1

        # -> BN -> Relu -> MaxPool
        self.first_bn = resnet.bn1
        self.first_relu = resnet.relu
        self.first_maxpool = resnet.maxpool

        # 遥感道路编码通道
        # resnet34: layers=[3, 4, 6, 3]
        # resnet34 的结构具体参考 https://towardsdatascience.com/understanding-and-visualizing-resnets-442284831be8
        # 第一遥感道路编码块. 提取图像的一些低层次的图像特征（例如边缘、线条和颜色块）
        # 输出特征图大小不变，通道数变为64
        # layer1 = (block, 64, layers[0])
        self.image_encoder1 = resnet.layer1
        # 第二遥感道路编码块. 进一步提取图像的特征并降低空间大小
        # 输出特征图的大小变为原来的一半，通道数变为128
        # layer2 = (block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.image_encoder2 = resnet.layer2
        # 第三遥感道路编码块. 进一步提取高层次的特征并降低空间大小
        # 输出特征图的大小变为原来的一半，通道数变为256
        self.image_encoder3 = resnet.layer3
        # 第四遥感道路编码块. 进一步提取更高层次的特征信息,更强调是否为背景/道路区域.
        # 输出特征图的大小再次变为原来的一半，通道数变为512
        self.image_encoder4 = resnet.layer4

        # GPS轨迹数据道路编码通道



