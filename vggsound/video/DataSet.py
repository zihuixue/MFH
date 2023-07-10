import json
import os

import torch
from imageio import imread
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    def __init__(self, path, num_frames, sr=16000):
        self.sr = sr
        with open("/public/MARS/datasets/vggsound/label_tiny.json") as f:
            self.label_list = json.load(f)
        self.num_frames = num_frames
        self.video_list = []

        videos = sorted(os.listdir(os.path.join(path, 'video_jpg')))
        for v in videos:
            if v.endswith('jpg'):
                self.video_list.append(os.path.join(path, 'video_jpg', v))


    def video_normalize(self, video):
        mean = torch.tensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1))
        std = torch.tensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1))
        return (video / 255. - mean) / std

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, item):

        video = imread(self.video_list[item]).reshape(self.num_frames, 256, 256, 3)

        video = self.video_normalize(torch.from_numpy(video).permute(0, 3, 1, 2))

        return {'video': video,  # Seq, C, H, W
                'label': self.label_list[self.video_list[item].split("/")[-1].split("_")[-2]]}


class VideoTestDataset(Dataset):
    def __init__(self, path, num_frames, sr=16000):
        self.sr = sr
        with open("/public/MARS/datasets/vggsound/label_tiny.json") as f:
            self.label_list = json.load(f)
        self.num_frames = num_frames
        self.video_list = []

        videos = sorted(os.listdir(os.path.join(path, 'video_jpg')))
        for v in videos:
            if v.endswith('jpg'):
                self.video_list.append(os.path.join(path, 'video_jpg', v))


    def video_normalize(self, video):
        mean = torch.tensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1))
        std = torch.tensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1))
        return (video / 255. - mean) / std

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, item):

        video = imread(self.video_list[item]).reshape(self.num_frames, 256, 256, 3)

        video = self.video_normalize(torch.from_numpy(video).permute(0, 3, 1, 2))

        return {'video': video,  # Seq, C, H, W
                'label': self.label_list[self.video_list[item].split("/")[-1].split("_")[-2]]}
