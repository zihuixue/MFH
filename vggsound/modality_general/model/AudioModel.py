import torch.nn as nn
import torchvision


class AudioModel(nn.Module):
    def __init__(self, num_classes):
        super(AudioModel, self).__init__()
        resnet = torchvision.models.resnet18(num_classes=num_classes)

        self.inconv = nn.Sequential(nn.Conv2d(1, 64, stride=2, kernel_size=7, padding=True, bias=False),
                                    resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self.fc_avg = resnet.fc


    def forward(self, x, fc=False):
        if fc:
            return self.fc_avg(x)
        else:
            feature = self.layer4(self.layer3(self.layer2(self.layer1(self.inconv(x)))))
            output = self.avgpool(feature).flatten(1)
            return output, self.fc_avg(output)