from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F


def my_permute(x, index):  # generate a new tensor
    y = x.reshape(x.shape[0], -1).detach().clone()  # flatten all feature, this function will only be used in the
    # context of with no grad
    perm_index = torch.randperm(x.shape[0])
    for i in index:
        y[:, i] = y[perm_index, i]
    y = y.reshape(*x.size())  # reshape to original size
    return y


def my_permute_new(x, index):
    y = deepcopy(x)
    perm_index = torch.randperm(x.shape[0])
    for i in index:
        y[:, i] = x[perm_index, i]
    return y


def my_freeze(x, index):  # in place modification
    ori_size = x.size()
    x = x.reshape(x.shape[0], -1)
    x[:, index] = 0
    x = x.reshape(*ori_size)
    return x


def my_freeze_new(x, index):  # in place modification
    # y = deepcopy(x)
    # y = x
    y = x.clone()

    # y[:, index] = 0
    tmp_mean = x[:, index].mean(dim=0)
    y[:, index] = tmp_mean

    return y


def my_change(x, change_type, index):
    if change_type == 'permute':
        return my_permute_new(x, index)
    elif change_type == 'freeze':
        return my_freeze_new(x, index)
    else:
        raise ValueError("Undefined change_type")


class ImageNet(nn.Module):
    def __init__(self, num_class=8):
        super(ImageNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.AdaptiveMaxPool2d((7, 14))

        self.fc1 = nn.Linear(128 * 7 * 14, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, num_class)


    def get_feature_dim(self, place=None):
        feature_dim_list = [3 * 256 * 512, 32 * 128 * 256, 64 * 64 * 128, 128 * 7 * 14, 1024, 128, 8]
        return feature_dim_list[place] if place else feature_dim_list

    def forward(self, x, change_type=None, place=None, index=None):
        # change_type = 'permute' / 'freeze'
        if place == 0:
            x = my_change(x, change_type, index)

        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        if place == 1:
            x = my_change(x, change_type, index)

        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        if place == 2:
            x = my_change(x, change_type, index)

        x = F.relu(self.conv3(x))
        x = self.pool2(x)
        if place == 3:
            x = my_change(x, change_type, index)

        x = x.view(-1, 128 * 7 * 14)
        x = F.relu(self.fc1(x))

        if place == 4:
            # tmp = x.detach().clone()
            x = my_change(x, change_type, index)  # (batch_size, 1024)
            # print(index, torch.sum(torch.sum(torch.abs(x-tmp))))

        x = F.relu(self.fc2(x))
        if place == 5:
            x = my_change(x, change_type, index)

        x = self.fc3(x)
        if place == 6:  # never goes into this one, permute/freeze won't happen at output layer
            x = my_change(x, change_type, index)

        return x


class AudioNet(nn.Module):
    def __init__(self, num_class=8):
        super(AudioNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=7, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding=1)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.AdaptiveMaxPool2d((7, 14))

        self.fc1 = nn.Linear(128 * 7 * 14, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, num_class)

    def get_feature_dim(self, place=None):
        feature_dim_list = [1 * 40 * 256, 32 * 128 * 126, 64 * 8 * 62, 128 * 7 * 14, 1024, 128, 8]
        return feature_dim_list[place] if place else feature_dim_list

    def forward(self, x, change_type=None, place=None, index=None):
        if place == 0:
            x = my_change(x, change_type, index)

        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        if place == 1:
            x = my_change(x, change_type, index)

        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        if place == 2:
            x = my_change(x, change_type, index)

        x = F.relu(self.conv3(x))
        x = self.pool2(x)
        if place == 3:
            x = my_change(x, change_type, index)

        x = x.view(-1, 128 * 7 * 14)
        x = F.relu(self.fc1(x))
        if place == 4:
            x = my_change(x, change_type, index)

        x = F.relu(self.fc2(x))
        # print('before', (x == 0).sum(dim=1)[0:5])
        if place == 5:
            x = my_change(x, change_type, index)
        # print('after', (x == 0).sum(dim=1)[0:5])

        x = self.fc3(x)
        if place == 6:  # never goes into this one, permute/freeze won't happen at output layer
            x = my_change(x, change_type, index)

        return x


class FuseNet(torch.nn.Module):
    def __init__(self, ImageNet, AudioNet):
        super(FuseNet, self).__init__()
        self.ImageNet = ImageNet
        self.AudioNet = AudioNet

    def forward(self, x_image, x_audio, change_type_image=None, place_image=None, index_image=None,
                change_type_audio=None, place_audio=None, index_audio=None):
        y = 0.5 * self.ImageNet(x_image, change_type_image, place_image, index_image) \
            + 0.5 * self.AudioNet(x_audio, change_type_audio, place_audio, index_audio)
        return y
