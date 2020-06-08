import torch.nn as nn
import torch


def basic_conv(in_channels, out_channels, stride=1, kernel_size=3):
    return nn.Conv2d(in_channels, out_channels, stride=stride,
                     kernel_size=kernel_size, padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.main_block = nn.Sequential(
            basic_conv(in_channels, out_channels, stride=stride),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            basic_conv(out_channels, out_channels))

        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels))

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.main_block(x) + self.downsample(x))


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=100):
        super(ResNet, self).__init__()
        self.in_channels = 32
        self.conv1 = nn.Sequential(basic_conv(in_channels=3, out_channels=32),
                                   nn.BatchNorm2d(num_features=32),
                                   nn.ReLU(inplace=True))

        self.middle_blocks = nn.Sequential(self._build_layer(block, layers[0], out_channels=32, stride=2),
                                           self._build_layer(block, layers[1], out_channels=64, stride=2),
                                           self._build_layer(block, layers[2], out_channels=128, stride=2),
                                           self._build_layer(block, layers[3], out_channels=256, stride=2),
                                           nn.MaxPool2d(kernel_size=(2, 2)))

        self.output_blocks = nn.Sequential(nn.Dropout(p=0.5),
                                           nn.Linear(256, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _build_layer(self, block, block_size, out_channels, stride=1):
        layers = [block(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        for _ in range(1, block_size):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.middle_blocks(out)
        out = torch.flatten(out, 1, -1)
        out = self.output_blocks(out)
        return out
