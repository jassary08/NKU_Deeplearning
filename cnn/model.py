import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Cnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # 4, 6, 28, 28
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y  # 通道加权

# Bottleneck block
class Bottleneck(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride=1, use_se=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.se = SqueezeExcitation(out_channels) if use_se else nn.Identity()

        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.se(out)

        out += identity
        out = self.relu(out)
        return out

# ResNet 主体
class ResNet(nn.Module):
    def __init__(self, img_channels, nums_blocks, nums_channels, first_kernel_size, num_labels, is_se=False):
        super().__init__()
        self.is_se = is_se
        self.block = Bottleneck

        # 输入卷积层
        padding = first_kernel_size // 2
        self.conv1 = nn.Conv2d(img_channels, nums_channels[0], first_kernel_size, stride=1, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(nums_channels[0])
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 残差层
        self.layer1 = self._make_layer(nums_blocks[0], nums_channels[0], nums_channels[1], stride=1)
        self.layer2 = self._make_layer(nums_blocks[1], nums_channels[1], nums_channels[2], stride=2)
        self.layer3 = self._make_layer(nums_blocks[2], nums_channels[2], nums_channels[3], stride=2)
        self.layer4 = self._make_layer(nums_blocks[3], nums_channels[3], nums_channels[4], stride=2)

        # 分类头
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.cls_head = nn.Linear(nums_channels[4], num_labels)

    def _make_layer(self, num_blocks, in_channels, out_channels, stride):
        layers = [self.block(in_channels, in_channels // 4, out_channels, stride, self.is_se)]
        for _ in range(1, num_blocks):
            layers.append(self.block(out_channels, out_channels // 4, out_channels, stride=1, use_se=self.is_se))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # out = self.max_pool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.cls_head(out)
        return out

class DenseBottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size=4):
        super().__init__()
        inter_channels = bn_size * growth_rate
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))
        out = torch.cat([x, out], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = self.pool(out)
        return out


class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate, bn_size=4):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(DenseBottleneck(in_channels + i * growth_rate, growth_rate, bn_size))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DenseNet121(nn.Module):
    def __init__(self, num_classes=10, growth_rate=32):
        super().__init__()
        num_init_features = 64
        block_layers = [6, 12, 24, 16]
        bn_size = 4
        reduction = 0.5

        # For CIFAR: use smaller first conv
        self.features = nn.Sequential(
            nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
        )

        num_features = num_init_features
        for i, num_layers in enumerate(block_layers):
            block = DenseBlock(num_layers, num_features, growth_rate, bn_size)
            self.features.add_module(f'denseblock{i + 1}', block)
            num_features += num_layers * growth_rate
            if i != len(block_layers) - 1:
                out_features = int(num_features * reduction)
                trans = Transition(num_features, out_features)
                self.features.add_module(f'transition{i + 1}', trans)
                num_features = out_features

        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.classifier = nn.Linear(num_features, num_classes)

        # Init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out