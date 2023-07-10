import torch.nn as nn


class BasicBlock3D(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1, downsample=None):
        super(BasicBlock3D, self).__init__()
        self.conv = nn.Sequential(nn.Conv3d(inplanes, outplanes, kernel_size=3, padding=1, stride=stride, bias=False),
                                  nn.BatchNorm3d(outplanes),
                                  nn.ReLU(True),
                                  nn.Conv3d(outplanes, outplanes, kernel_size=3, padding=1, bias=False),
                                  nn.BatchNorm3d(outplanes))
        self.downsample=downsample
        self.relu = nn.ReLU(True)
    def forward(self, x):
        feature = self.conv(x)
        if self.downsample is not None:
            return self.relu(self.downsample(x)+feature)
        return self.relu(x+feature)

class VideoModel(nn.Module):
    def __init__(self, num_classes):
        super(VideoModel, self).__init__()
        self.inconv = nn.Sequential(nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=2, padding=[1,3,3], bias=False),
                                    nn.BatchNorm3d(64),
                                    nn.ReLU(True),
                                    nn.MaxPool3d(kernel_size=3, stride=2, padding=1))
        self.layer1 = self._make_layer(BasicBlock3D, 2, 64, 64, stride=1)
        self.layer2 = self._make_layer(BasicBlock3D, 2, 64, 128, stride=2)
        self.layer3 = self._make_layer(BasicBlock3D, 2, 128, 256, stride=2)
        self.layer4 = self._make_layer(BasicBlock3D, 2, 256, 512, stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.proj = nn.Linear(512, num_classes)

    def _make_layer(self, block, num_block, inplanes, outplanes, stride):
        downsample=None
        if inplanes!=outplanes or stride!=1:
            downsample = nn.Sequential(nn.Conv3d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False),
                                       nn.BatchNorm3d(outplanes))
        layers=[block(inplanes, outplanes, stride, downsample)]
        for i in range(num_block):
            layers.append(block(outplanes, outplanes))
        return nn.Sequential(*layers)

    def forward(self, x, fc=False):
        if fc:
            self.proj(x)
        else:
            hx = self.inconv(x.permute(0, 2, 1, 3, 4))
            h1 = self.layer1(hx)
            h2 = self.layer2(h1)
            h3 = self.layer3(h2)
            h4 = self.layer4(h3)
            h5 = self.avgpool(h4).squeeze(-1).squeeze(-1).squeeze(-1)
            return h4, self.proj(h5)


