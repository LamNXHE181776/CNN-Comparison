import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act_func, stride=1, downsample=None):
        super().__init__()

        self.act_func = act_func()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            act_func()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.act_func(out)

        return out


class ResNet(nn.Module):

    def __init__(self, shape=(1, 1, 28, 28), num_classes=10, act_func=nn.ReLU, layers=[2, 2, 2],  block=ResidualBlock):
        super().__init__()

        self.shape = shape
        self.act_func = act_func
        self.block = block

        self.inplanes = 16

        self.conv1 = nn.Sequential(
            nn.Conv2d(shape[1], 16, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            act_func()
        )

        self.layer1 = self._make_layer(16, layers[0], stride=1)
        self.layer2 = self._make_layer(32, layers[1], stride=2)
        self.layer3 = self._make_layer(64, layers[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        with torch.no_grad():
            dummy = torch.zeros(shape)
            out = self.conv1(dummy)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.avgpool(out)
            flattened_size = out.view(1, -1).size(1)

        self.fc = nn.Linear(flattened_size, num_classes)

        self._initialize_weights()

    def _make_layer(self, planes, blocks, stride):

        downsample = None

        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

        layers = []

        layers.append(
            self.block(
                self.inplanes,
                planes,
                self.act_func,
                stride,
                downsample
            )
        )

        self.inplanes = planes

        for _ in range(1, blocks):
            layers.append(
                self.block(
                    self.inplanes,
                    planes,
                    self.act_func
                )
            )

        return nn.Sequential(*layers)

    def _initialize_weights(self):

        for m in self.modules():

            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x