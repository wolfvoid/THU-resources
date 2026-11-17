## Baseline
import torch
import torch.nn as nn
class Baseline(nn.Module):
    def __init__(self, height, width, num_targets):
        super(Baseline, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self._feature_size = self._get_feature_size(height, width)
        self.regressor = nn.Sequential(
            nn.Linear(self._feature_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_targets)
        )

    def _get_feature_size(self, height, width):
        dummy_input = torch.zeros(1, 1, height, width)
        with torch.no_grad():
            features = self.conv(dummy_input)
        return features.view(1, -1).shape[1]

    def forward(self, x):
        features = self.conv(x)
        features = features.view(features.size(0), -1)
        output = self.regressor(features)
        means = output[:, :2]
        log_sigmas = output[:, 2:]
        sigmas = torch.exp(log_sigmas)
        return means, sigmas


# TODO : Define your model here
## ResNet18
import torch
import torch.nn as nn
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, height, width, num_targets):
        super(ResNet18, self).__init__()
        self.in_planes = 64

        # 初始卷积层
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # ResNet-18的4个残差块层
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        # 计算特征图尺寸
        self._feature_size = self._get_feature_size(height, width)
        # 回归头 - 与Baseline保持一致的结构
        self.regressor = nn.Sequential(
            nn.Linear(self._feature_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_targets)
        )

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def _get_feature_size(self, height, width):
        dummy_input = torch.zeros(1, 1, height, width)
        with torch.no_grad():
            features = self.forward_features(dummy_input)
        return features.view(1, -1).shape[1]

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 全局平均池化
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        features = self.forward_features(x)
        output = self.regressor(features)
        means = output[:, :2]
        log_sigmas = output[:, 2:]
        sigmas = torch.exp(log_sigmas)
        return means, sigmas


## ResNet50
import torch
import torch.nn as nn

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        # 第一个1x1卷积，降维
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        # 第二个3x3卷积，特征提取
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # 第三个1x1卷积，升维
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = torch.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNet50(nn.Module):
    def __init__(self, height, width, num_targets):
        super(ResNet50, self).__init__()
        self.in_planes = 64

        # 初始卷积层
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet-50的4个残差块层
        self.layer1 = self._make_layer(64, 3, stride=1)    # 3个Bottleneck块
        self.layer2 = self._make_layer(128, 4, stride=2)   # 4个Bottleneck块
        self.layer3 = self._make_layer(256, 6, stride=2)   # 6个Bottleneck块
        self.layer4 = self._make_layer(512, 3, stride=2)   # 3个Bottleneck块

        # 计算特征图尺寸
        self._feature_size = self._get_feature_size(height, width)

        # 回归头 - 与Baseline保持一致的结构
        self.regressor = nn.Sequential(
            nn.Linear(self._feature_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_targets)
        )

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(Bottleneck(self.in_planes, planes, stride))
            self.in_planes = planes * Bottleneck.expansion
        return nn.Sequential(*layers)

    def _get_feature_size(self, height, width):
        dummy_input = torch.zeros(1, 1, height, width)
        with torch.no_grad():
            features = self.forward_features(dummy_input)
        return features.view(1, -1).shape[1]

    def forward_features(self, x):
        # 初始卷积和池化
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.maxpool(x)

        # 四个残差层
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 全局平均池化
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        features = self.forward_features(x)
        output = self.regressor(features)
        means = output[:, :2]
        log_sigmas = output[:, 2:]
        sigmas = torch.exp(log_sigmas)
        return means, sigmas

## ConvNeXt
import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm2d(nn.LayerNorm):
    """正确的2D Layer Normalization实现"""
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__(normalized_shape, eps=eps)

    def forward(self, x):
        # 输入x的形状: (B, C, H, W)
        # 需要转换为 (B, H, W, C) 进行LayerNorm，然后再转回
        x = x.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        return x

class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block"""

    def __init__(self, dim, layer_scale_init_value=1e-6):
        super().__init__()

        # 深度可分离卷积 (7x7大核)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)

        # 层归一化
        self.norm = LayerNorm2d(dim, eps=1e-6)

        # 倒置瓶颈结构: 使用1x1卷积而不是Linear，避免形状问题
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)

        # 激活函数
        self.act = nn.GELU()

        # Layer Scale
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        identity = x

        # 深度可分离卷积
        x = self.dwconv(x)

        # LayerNorm
        x = self.norm(x)

        # 倒置瓶颈 - 使用卷积，避免形状转换问题
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        # Layer Scale
        if self.gamma is not None:
            # 需要将gamma扩展到正确的形状
            gamma = self.gamma.view(1, -1, 1, 1)
            x = gamma * x

        # 残差连接
        x = identity + x
        return x

class ConvNeXt(nn.Module):
    # def __init__(self, height, width, num_targets, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]):
    def __init__(self, height, width, num_targets, depths=[2, 2, 2, 2], dims=[16, 32, 64, 128]):
        super(ConvNeXt, self).__init__()

        # Stem: 4x4卷积模拟Patchify
        self.stem = nn.Sequential(
            nn.Conv2d(1, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0], eps=1e-6)
        )

        # 下采样层
        self.downsample_layers = nn.ModuleList()
        self.stages = nn.ModuleList()

        # 构建每个阶段
        for i in range(len(dims)):
            # 下采样层 (除了第一个stage)
            if i > 0:
                downsample_layer = nn.Sequential(
                    LayerNorm2d(dims[i-1], eps=1e-6),
                    nn.Conv2d(dims[i-1], dims[i], kernel_size=2, stride=2),
                )
                self.downsample_layers.append(downsample_layer)
            else:
                # 第一个stage不需要下采样
                self.downsample_layers.append(nn.Identity())

            # 阶段层
            stage = nn.Sequential(
                *[ConvNeXtBlock(dim=dims[i]) for _ in range(depths[i])]
            )
            self.stages.append(stage)

        # 全局归一化
        self.norm = LayerNorm2d(dims[-1], eps=1e-6)

        # 计算特征图尺寸
        self._feature_size = self._get_feature_size(height, width, dims[-1])

        # 回归头 - 与ResNet18保持一致的结构
        self.regressor = nn.Sequential(
            nn.Linear(self._feature_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_targets)
        )

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _get_feature_size(self, height, width, last_dim):
        """计算最终特征向量尺寸"""
        dummy_input = torch.zeros(1, 1, height, width)
        with torch.no_grad():
            features = self.forward_features(dummy_input)
        return features.view(1, -1).shape[1]

    def forward_features(self, x):
        # Stem
        x = self.stem(x)

        # 四个阶段
        for i in range(len(self.stages)):
            if i > 0:  # 从第二个阶段开始进行下采样
                x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        # 全局平均池化前的归一化
        x = self.norm(x)

        # 全局平均池化
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        features = self.forward_features(x)
        output = self.regressor(features)
        means = output[:, :2]
        log_sigmas = output[:, 2:]
        sigmas = torch.exp(log_sigmas)
        return means, sigmas


## ResNeXt18
import torch
import torch.nn as nn

class BasicBlock_ResNeXt(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, cardinality=64):
        super(BasicBlock_ResNeXt, self).__init__()
        self.cardinality = cardinality

        # 计算每组的宽度，保持总计算量接近原始ResNet
        group_width = planes // cardinality

        # 第一个卷积：使用分组卷积实现多路径
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                              stride=stride, padding=1,
                              groups=cardinality, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        # 第二个卷积：标准卷积
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # 捷径连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNeXt18(nn.Module):
    def __init__(self, height, width, num_targets, cardinality=32):
        super(ResNeXt18, self).__init__()
        self.in_planes = 64
        self.cardinality = cardinality

        # 初始卷积层 - 与ResNet18保持一致
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNeXt18的4个残差块层 - 与ResNet18相同的层结构
        # 每层2个基本块，总共8个块（2-2-2-2）
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        # 计算特征图尺寸
        self._feature_size = self._get_feature_size(height, width)

        # 回归头 - 与ResNet18保持一致的结构
        self.regressor = nn.Sequential(
            nn.Linear(self._feature_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_targets)
        )

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock_ResNeXt(self.in_planes, planes, stride, self.cardinality))
            self.in_planes = planes * BasicBlock_ResNeXt.expansion
        return nn.Sequential(*layers)

    def _get_feature_size(self, height, width):
        dummy_input = torch.zeros(1, 1, height, width)
        with torch.no_grad():
            features = self.forward_features(dummy_input)
        return features.view(1, -1).shape[1]

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 全局平均池化
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        features = self.forward_features(x)
        output = self.regressor(features)
        means = output[:, :2]
        log_sigmas = output[:, 2:]
        sigmas = torch.exp(log_sigmas)
        return means, sigmas



## MSAFNet(废弃)
import torch
import torch.nn as nn
import torch.nn.functional as F
class MSAFNet(nn.Module):
    def __init__(self, height, width, num_targets):
        super(MSAFNet, self).__init__()

        # 初始特征提取层
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 多尺度特征提取模块
        self.multiscale_block1 = MultiScaleBlock(32, 64)
        self.multiscale_block2 = MultiScaleBlock(64, 128)
        self.multiscale_block3 = MultiScaleBlock(128, 256)

        # 注意力机制
        self.attention = DualAttention(256)

        # 全局池化前的卷积
        self.final_conv = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1)
        )

        # 获取特征尺寸
        self._feature_size = self._get_feature_size(height, width)
        # 回归器 - 保持与baseline相同的输出结构
        self.regressor = nn.Sequential(
            nn.Linear(self._feature_size, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_targets)  # 输出均值和log方差
        )

    def _get_feature_size(self, height, width):
        dummy_input = torch.zeros(1, 1, height, width)
        with torch.no_grad():
            x = self.stem(dummy_input)
            x = self.multiscale_block1(x)
            x = self.multiscale_block2(x)
            x = self.multiscale_block3(x)
            x = self.attention(x)
            x = self.final_conv(x)
        return x.view(1, -1).shape[1]

    def forward(self, x):
        # 特征提取
        # print(x.shape)
        # torch.Size([16, 1, 58, 474])

        x = self.stem(x)
        # print(x.shape)
        # torch.Size([16, 32, 14, 118])

        x = self.multiscale_block1(x)
        # print(x.shape)
        # torch.Size([16, 64, 7, 59])

        x = self.multiscale_block2(x)
        # print(x.shape)
        # torch.Size([16, 128, 4, 30])

        x = self.multiscale_block3(x)
        # print(x.shape)
        # torch.Size([16, 256, 2, 15])

        x = self.attention(x)
        # print(x.shape)
        # torch.Size([16, 256, 2, 15])

        x = self.final_conv(x)
        # print(x.shape)
        # torch.Size([16, 512, 1, 1])

        # 展平
        x = x.view(x.size(0), -1)

        # 回归预测
        output = self.regressor(x)

        # 分割输出：前num_targets个是均值，后num_targets个是log方差
        means = output[:, :2]
        log_sigmas = output[:, 2:]
        sigmas = torch.exp(log_sigmas)  # 转换为标准差

        return means, sigmas

class MultiScaleBlock(nn.Module):
    """多尺度卷积块"""
    def __init__(self, in_channels, out_channels):
        super(MultiScaleBlock, self).__init__()

        # 确保输出通道数能被4整除
        assert out_channels % 4 == 0, "out_channels must be divisible by 4"

        # 分支1: 1x1卷积
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, kernel_size=1),
            nn.BatchNorm2d(out_channels//4),
            nn.GELU()
        )

        # 分支2: 3x3深度可分离卷积
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, out_channels//4, kernel_size=1),
            nn.BatchNorm2d(out_channels//4),
            nn.GELU()
        )

        # 分支3: 5x5深度可分离卷积
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels),
            nn.Conv2d(in_channels, out_channels//4, kernel_size=1),
            nn.BatchNorm2d(out_channels//4),
            nn.GELU()
        )

        # 分支4: 池化分支
        self.branch4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels//4, kernel_size=1),
            # nn.BatchNorm2d(out_channels//4),
            nn.GELU()
        )

        # 下采样
        self.downsample = nn.Sequential(
            nn.Conv2d(out_channels, out_channels,
                     kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),  # 32→64，同时下采样
            nn.BatchNorm2d(out_channels)
        )


    def forward(self, x):
        # 保存原始输入尺寸
        b, c, h, w = x.shape

        # 多尺度特征提取
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        b4 = F.interpolate(b4, size=b1.shape[2:], mode='bilinear', align_corners=False) # 池化分支需要上采样到相同尺寸

        # 特征融合
        out = torch.cat([b1, b2, b3, b4], dim=1)

        # 下采样
        out = self.downsample(out)

        # shortcut连接
        x_short = self.shortcut(x)
        out = out + x_short

        return out

class DualAttention(nn.Module):
    """双重注意力机制"""
    def __init__(self, channels, reduction=16):
        super(DualAttention, self).__init__()

        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.GELU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道注意力
        ca = self.channel_attention(x)
        x = x * ca

        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        sa = self.spatial_attention(spatial_input)
        x = x * sa

        return x


## MultiScaleResNet18
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock_m(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_m, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class MultiScaleResNet18(nn.Module):
    def __init__(self, height, width, num_targets, fusion_channels=128):
        super(MultiScaleResNet18, self).__init__()
        self.in_planes = 64

        # 初始卷积层
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet-18的4个残差块层
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        # 多尺度特征融合的适配卷积层
        self.adapt_conv1 = nn.Conv2d(64, fusion_channels, 1)   # layer1输出64通道
        self.adapt_conv2 = nn.Conv2d(128, fusion_channels, 1)  # layer2输出128通道
        self.adapt_conv3 = nn.Conv2d(256, fusion_channels, 1)  # layer3输出256通道
        self.adapt_conv4 = nn.Conv2d(512, fusion_channels, 1)  # layer4输出512通道

        # 计算最终特征维度
        self._feature_size = self._get_feature_size(height, width, fusion_channels)

        # 回归头
        self.regressor = nn.Sequential(
            nn.Linear(self._feature_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_targets)
        )


    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock_m(self.in_planes, planes, stride))
            self.in_planes = planes * BasicBlock_m.expansion
        return nn.Sequential(*layers)

    def _get_feature_size(self, height, width, fusion_channels):
        dummy_input = torch.zeros(1, 1, height, width)
        with torch.no_grad():
            features = self.forward_features(dummy_input)
        return features.view(1, -1).shape[1]

    def forward_features(self, x):
        # 初始特征提取
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x0 = self.maxpool(x)

        # 各层特征提取
        x1 = self.layer1(x0)  # [batch, 64, H/4, W/4]
        x2 = self.layer2(x1)  # [batch, 128, H/8, W/8]
        x3 = self.layer3(x2)  # [batch, 256, H/16, W/16]
        x4 = self.layer4(x3)  # [batch, 512, H/32, W/32]

        # 多尺度特征融合 - concat方式
        # 1. 统一通道数
        # f1 = self.adapt_conv1(x1)  # [batch, 128, H/4, W/4]
        # f2 = self.adapt_conv2(x2)  # [batch, 128, H/8, W/8]
        # f3 = self.adapt_conv3(x3)  # [batch, 128, H/16, W/16]
        # f4 = self.adapt_conv4(x4)  # [batch, 128, H/32, W/32]
        f1 = x1
        f2 = x2
        f3 = x3
        f4 = x4

        # 2. 上采样到相同尺寸（以layer1的尺寸为目标）
        # target_size = x1.shape[2:]  # (H/4, W/4)
        # f2 = F.interpolate(f2, size=target_size, mode='bilinear', align_corners=False)
        # f3 = F.interpolate(f3, size=target_size, mode='bilinear', align_corners=False)
        # f4 = F.interpolate(f4, size=target_size, mode='bilinear', align_corners=False)

        # 2. 下采样到最小尺寸（layer4的尺寸）
        target_size = x4.shape[2:]  # (H/32, W/32)
        f1 = F.interpolate(f1, size=target_size, mode='bilinear', align_corners=False)
        f2 = F.interpolate(f2, size=target_size, mode='bilinear', align_corners=False)
        f3 = F.interpolate(f3, size=target_size, mode='bilinear', align_corners=False)

        # 3. concat所有特征
        fused = torch.cat([f1, f2, f3, f4], dim=1)  # [batch, 512, H/4, W/4]
        # print(fused.shape)

        # 4. 全局平均池化
        fused = nn.AdaptiveAvgPool2d((1, 1))(fused)
        fused = fused.view(fused.size(0), -1)  # [batch, 512]

        return fused

    def forward(self, x):
        features = self.forward_features(x)
        output = self.regressor(features)
        means = output[:, :2]
        log_sigmas = output[:, 2:]
        sigmas = torch.exp(log_sigmas)
        return means, sigmas


## MultiScaleResNeXt18
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock_ResNeXt_m(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, cardinality=32):
        super(BasicBlock_ResNeXt_m, self).__init__()
        self.cardinality = cardinality

        # 第一个卷积：使用分组卷积实现多路径（ResNeXt核心思想）
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                              stride=stride, padding=1,
                              groups=cardinality, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        # 第二个卷积：标准卷积
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # 捷径连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class MultiScaleResNeXt18(nn.Module):
    def __init__(self, height, width, num_targets, fusion_channels=128, cardinality=32):
        super(MultiScaleResNeXt18, self).__init__()
        self.in_planes = 64
        self.cardinality = cardinality

        # 初始卷积层
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNeXt-18的4个残差块层（使用分组卷积）
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        # 多尺度特征融合的适配卷积层
        self.adapt_conv1 = nn.Conv2d(64, fusion_channels, 1)   # layer1输出64通道
        self.adapt_conv2 = nn.Conv2d(128, fusion_channels, 1)  # layer2输出128通道
        self.adapt_conv3 = nn.Conv2d(256, fusion_channels, 1)  # layer3输出256通道
        self.adapt_conv4 = nn.Conv2d(512, fusion_channels, 1)  # layer4输出512通道

        # 计算最终特征维度
        self._feature_size = self._get_feature_size(height, width, fusion_channels)

        # 回归头（保持与原始模型一致）
        self.regressor = nn.Sequential(
            nn.Linear(self._feature_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_targets)
        )

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock_ResNeXt_m(self.in_planes, planes, stride, self.cardinality))
            self.in_planes = planes * BasicBlock_ResNeXt_m.expansion
        return nn.Sequential(*layers)

    def _get_feature_size(self, height, width, fusion_channels):
        dummy_input = torch.zeros(1, 1, height, width)
        with torch.no_grad():
            features = self.forward_features(dummy_input)
        return features.view(1, -1).shape[1]

    def forward_features(self, x):
        # 初始特征提取
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x0 = self.maxpool(x)

        # 各层特征提取（使用ResNeXt块）
        x1 = self.layer1(x0)  # [batch, 64, H/4, W/4]
        x2 = self.layer2(x1)  # [batch, 128, H/8, W/8]
        x3 = self.layer3(x2)  # [batch, 256, H/16, W/16]
        x4 = self.layer4(x3)  # [batch, 512, H/32, W/32]

        # 多尺度特征融合
        # 使用适配卷积统一通道数（可选，当前注释掉）
        # f1 = self.adapt_conv1(x1)
        # f2 = self.adapt_conv2(x2)
        # f3 = self.adapt_conv3(x3)
        # f4 = self.adapt_conv4(x4)

        f1 = x1
        f2 = x2
        f3 = x3
        f4 = x4

        # 下采样到最小尺寸（layer4的尺寸）
        target_size = x4.shape[2:]  # (H/32, W/32)
        f1 = F.interpolate(f1, size=target_size, mode='bilinear', align_corners=False)
        f2 = F.interpolate(f2, size=target_size, mode='bilinear', align_corners=False)
        f3 = F.interpolate(f3, size=target_size, mode='bilinear', align_corners=False)

        # concat所有特征
        fused = torch.cat([f1, f2, f3, f4], dim=1)  # [batch, 64+128+256+512, H/32, W/32]

        # 全局平均池化
        fused = nn.AdaptiveAvgPool2d((1, 1))(fused)
        fused = fused.view(fused.size(0), -1)

        return fused

    def forward(self, x):
        features = self.forward_features(x)
        output = self.regressor(features)
        means = output[:, :2]
        log_sigmas = output[:, 2:]
        sigmas = torch.exp(log_sigmas)
        return means, sigmas




if __name__ == '__main__':
    model_name = "MultiScaleResNeXt18"

    if model_name == 'Baseline':
        model = Baseline(58, 474, 4).to("cuda")
    elif model_name == 'ResNet18':
        model = ResNet18(58, 474, 4).to("cuda")
    elif model_name == 'ResNet50':
        model = ResNet50(58, 474, 4).to("cuda")
    elif model_name == 'ConvNeXt':
        model = ConvNeXt(58, 474, 4).to("cuda")
    elif model_name == 'ResNeXt18':
        model = ResNeXt18(58, 474, 4).to("cuda")
    elif model_name == 'MultiScaleResNet18':
        model = MultiScaleResNet18(58, 474, 4).to("cuda")
    elif model_name == 'MultiScaleResNeXt18':
        model = MultiScaleResNeXt18(58, 474, 4).to("cuda")
    else:
        raise ValueError(f"Model {model_name} is not supported.")

    input_tensor = torch.randn(16, 1, 58, 474).to("cuda")
    print("start")
    pred_means, pred_sigmas = model(input_tensor)
    print(f"Means shape: {pred_means.shape}, Sigmas shape: {pred_sigmas.shape}")
    print(f"Means: {pred_means[0]}, Sigmas: {pred_sigmas[0]}")
    print("end")

