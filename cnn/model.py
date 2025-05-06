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

class Bottleneck(nn.Module):
    def __init__(self,in_channels,bottleneck_channels,out_channels,stride,se):
        super().__init__()
        # conv1降低维度
        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=bottleneck_channels,kernel_size=1,stride=stride,bias=False)
        # conv2提取特征
        self.conv2 = nn.Conv2d(in_channels=bottleneck_channels,out_channels=bottleneck_channels,kernel_size=3,stride=1,padding=1,bias=False)
        # conv3增加维度
        self.conv3 = nn.Conv2d(in_channels=bottleneck_channels,out_channels=out_channels,kernel_size=1,stride=1,bias=False)
        # relu激活函数
        self.relu = nn.ReLU()
        # Batch Norm Layer
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.se = SqueezeExcitation(out_channels)
        self.is_se = se
        self.stride = stride

        if self.stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,1,stride ,bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self,x):
        residual = self.shortcut(x)

        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)

        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu(output)

        output = self.conv3(output)
        output = self.bn3(output)
        output = self.relu(output)

        if self.is_se:
            output = self.se(output)

        output = output + residual
        output = self.relu(output)

        return output


class ResNet(nn.Module):
    def __init__(self,img_channels,nums_blocks,nums_channels,first_kernel_size,num_labels,is_se = False):
        super().__init__()
        self.block = Bottleneck

        #原始输入600*600*3->300*300*3
        self.conv1 = nn.Conv2d(img_channels,nums_channels[0], first_kernel_size, 2, first_kernel_size, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        self.max_pool = nn.MaxPool2d(3,2,0)

        self.layer1 = self._make_layer(nums_blocks[0], nums_channels[0], nums_channels[1], stride=1)
        self.layer2 = self._make_layer(nums_blocks[1], nums_channels[1], nums_channels[2], stride=2)
        self.layer3 = self._make_layer(nums_blocks[2], nums_channels[2], nums_channels[3], stride=2)
        self.layer4 = self._make_layer(nums_blocks[3], nums_channels[3], nums_channels[4], stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.cls_head = nn.Linear(nums_channels[4],num_labels)


    def _make_layer(self, num_blocks, in_channels, out_channels, stride):
        layers = []
        layers.append(self.block(in_channels, in_channels // 4, out_channels, stride, self.is_se))

        for _ in range(1, num_blocks):
            layers.append(self.block(out_channels, out_channels // 4, out_channels, stride=1, se = self.is_se))

        return nn.Sequential(*layers)

    def forward(self,x):
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)

        output = self.max_pool(output)

        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)

        output = self.avg_pool(output)
        output = output.view(output.size(0),-1)
        output = self.cls_head(output)

        return output

class DenseLayer(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(DenseLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out

class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, growthRate, depth, reduction, nClasses):
        super(DenseNet, self).__init__()

        nDenseBlocks = (depth-4) // 3

        nChannels = 2*growthRate
        self.conv1 = nn.Conv2d(3, nChannels, kernel_size=3, padding=1,
                               bias=False)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans1 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans2 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks)
        nChannels += nDenseBlocks*growthRate

        self.bn1 = nn.BatchNorm2d(nChannels)
        self.fc = nn.Linear(nChannels, nClasses)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense(self, nChannels, growthRate, nDenseBlocks):
        layers = []
        for i in range(int(nDenseBlocks)):
            layers.append(DenseLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)), 8))
        out = F.log_softmax(self.fc(out))
        return out
