import json
import os
import librosa
import torch.nn.functional as F
from nnAudio.Spectrogram import STFT
import numpy as np
import torch
from torch.utils.data import Dataset
from imageio import imread
class AudioVideoDataset(Dataset):
    def __init__(self, path, num_frames, sr=16000, window_size=1024, hop_size=512):
        self.sr = sr
        with open("/public/MARS/datasets/vggsound/label_tiny.json") as f:
            self.label_list = json.load(f)
        self.num_frames = num_frames
        self.audio_list = []
        self.video_list = []
        videos = sorted(os.listdir(os.path.join(path, 'video_jpg')))
        self.stft = STFT(n_fft=window_size, win_length=window_size, hop_length=hop_size, sr=16000,
                         output_format='Magnitude')
        for v in videos:
            if v.endswith('jpg'):
                self.audio_list.append(os.path.join(path, 'audio_npy', v[:-3] + 'npy'))
                self.video_list.append(os.path.join(path, 'video_jpg', v))


    def video_normalize(self, video):
        mean = torch.tensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1))
        std = torch.tensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1))
        return (video / 255. - mean) / std

    def wave_normalize(self, wav):
        norm = torch.max(torch.abs(wav)) * 1.1
        wav = wav / norm
        return wav

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, item):

        video = imread(self.video_list[item]).reshape(self.num_frames, 256, 256, 3)

        video = self.video_normalize(torch.from_numpy(video).permute(0, 3, 1, 2))
        wave = np.load(self.audio_list[item]).squeeze(0)
        wave = self.wave_normalize(torch.from_numpy(wave))
        stft = self.stft(wave)
        return {'video': video,  # Seq, C, H, W
                'audio': stft,
                'label': self.label_list[self.video_list[item].split("/")[-1].split("_")[-2]]}


class AudioDataset(Dataset):
    def __init__(self, path, window_size=1024, hop_size=512):
        with open("/public/MARS/datasets/vggsound/label_tiny.json") as f:
            self.label_list = json.load(f)
        self.test = "test" in path

        self.audio_list = []
        self.video_list = []
        videos = sorted(os.listdir(os.path.join(path, 'video_jpg')))
        self.stft = STFT(n_fft=window_size, win_length=window_size, hop_length=hop_size, sr=16000, output_format='Magnitude')
        for v in videos:
            if v.endswith('jpg'):
                self.audio_list.append(os.path.join(path, 'audio_npy', v[:-3] + 'npy'))
                self.video_list.append(os.path.join(path, 'video_jpg', v))

    def wave_normalize(self, wav):
        norm = torch.max(torch.abs(wav)) * 1.1
        wav = wav / norm
        return wav

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, item):

        wave = np.load(self.audio_list[item]).squeeze(0)
        wave = self.wave_normalize(torch.from_numpy(wave))
        stft = self.stft(wave)
        return {'audio': stft,
                'label': self.label_list[self.video_list[item].split("/")[-1].split("_")[-2]]}


class AudioTestDataset(Dataset):
    def __init__(self, path, window_size=1024, hop_size=512, sr=16000):
        with open("/public/MARS/datasets/vggsound/label_tiny.json") as f:
            self.label_list = json.load(f)
        self.audio_list =[]
        self.video_list=[]
        self.sr = sr

        self.stft = STFT(n_fft=window_size, win_length=window_size, hop_length=hop_size, sr=16000, output_format='Magnitude')

        videos = sorted(os.listdir(os.path.join(path, 'video_jpg')))
        for v in videos:
            if v.endswith('jpg'):
                self.audio_list.append(os.path.join(path, 'audio', v[:-3]+'wav'))
                self.video_list.append(os.path.join(path, 'video_jpg', v))

    def wave_normalize(self, wav):
        norm = torch.max(torch.abs(wav)) * 1.1
        wav = wav/norm
        return wav


    def __len__(self):
        return len(self.video_list)
    def __getitem__(self, item):

        wave, _ = librosa.core.load(self.audio_list[item], sr=self.sr, mono=True)
        wave = self.wave_normalize(torch.from_numpy(wave))
        stft = self.stft(wave)
        stft = F.interpolate(stft.unsqueeze(0), (513, 313), mode="bilinear", align_corners=False).squeeze(0)
        return {
                'audio': stft,
                'label': self.label_list[self.video_list[item].split("/")[-1].split("_")[-2]]
        }


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