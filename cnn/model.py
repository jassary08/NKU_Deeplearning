import torch
import torch.nn as nn
import torch.nn.functional as F


class cnn(nn.Module):
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

class Bottleneck(nn.Module):
    def __init__(self,in_channels,bottleneck_channels,out_channels,stride):
        super().__init__()
        # conv1降低维度
        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=bottleneck_channels,kernel_size=1,stride=stride,bias=False)
        # conv2提取特征
        self.conv2 = nn.Conv2d(in_channels=bottleneck_channels,out_channels=bottleneck_channels,kernel_size=3,stride=1,padding=1,bias=False)
        # conv3增加维度
        self.conv3 = nn.Conv2d(in_channels=bottleneck_channels,out_channels=out_channels,kernel_size=1,stride=1,bias=False)
        # relu激活函数
        self.Relu = nn.ReLU()
        # Batch Norm Layer
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)

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
        output = self.Relu(output)

        output = self.conv2(output)
        output = self.bn2(output)
        output = self.Relu(output)

        output = self.conv3(output)
        output = self.bn3(output)
        output = self.Relu(output)

        if self.down_sample is not None:
            output = self.down_sample(output)

        output += residual
        output = self.Relu(output)

        return output


class ResNet(nn.Module):
    def __init__(self,img_channels,nums_blocks,nums_channels,first_kernel_size,num_labels):
        super().__init__()
        self.block = Bottleneck()

        #原始输入600*600*3->300*300*3
        self.conv1 = nn.Conv2d(img_channels,nums_channels[0], first_kernel_size, 2, first_kernel_size, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.Relu = nn.ReLU()

        self.max_pool = nn.MaxPool2d(3,2,0)

        self.layer1 = self._make_layer(nums_blocks[0], nums_channels[0], nums_channels[1], stride=1)
        self.layer2 = self._make_layer(nums_blocks[1], nums_channels[1], nums_channels[2], stride=2)
        self.layer3 = self._make_layer(nums_blocks[2], nums_channels[2], nums_channels[3], stride=2)
        self.layer4 = self._make_layer(nums_blocks[3], nums_channels[3], nums_channels[4], stride=2)

        self.avg_pool = nn.AvgPool2d((1,1))
        self.cls_head = nn.Linear(nums_channels[4],num_labels)


    def _make_layer(self, num_blocks, in_channels, out_channels, stride):
        layers = []
        layers.append(self.block(in_channels, in_channels // 4, out_channels, stride))

        for _ in range(1, num_blocks):
            layers.append(self.block(out_channels, out_channels // 4, out_channels, stride=1))

        return nn.Sequential(*layers)

    def forward(self,x):
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.Relu(output)

        output = self.max_pool(output)

        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)

        output = self.avg_pool(output)
        output = output.view(output.size(0),-1)
        output = self.cls_head(output)

        return output


