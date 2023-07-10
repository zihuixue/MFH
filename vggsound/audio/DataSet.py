import json
import os
import librosa
import torch.nn.functional as F
from nnAudio.Spectrogram import STFT
import numpy as np
import torch
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self, path, window_size=1024, hop_size=512):
        with open("/path/to/vggsound/label.json") as f:
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
        with open("/path/to/vggsound/label.json") as f:
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
        stft = F.interpolate(stft.unsqueeze(0), (513, 313), mode="bilinear").squeeze(0)
        return {
                'audio': stft,
                'label': self.label_list[self.video_list[item].split("/")[-1].split("_")[-2]]
        }
