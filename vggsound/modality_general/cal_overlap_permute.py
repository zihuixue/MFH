import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import DataSet
import model
import numpy as np
import random
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train_dataset', type=str, help='root path of dataset')
    parser.add_argument('--test_dataset', type=str, help='root path of dataset')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate, defaults to 1e-3')
    parser.add_argument('--nb-workers', type=int, default=16, help='Number of workers for dataloader.')
    parser.add_argument('--nb-class', type=int, default=100, help='Number of class for dataset.')
    parser.add_argument('--v_ckpt', type=str, help='pretrained video model')
    parser.add_argument('--a_ckpt', type=str, help='pretrained audio model')
    args = parser.parse_args()
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    if not os.path.exists("ckpt"):
        os.mkdir("ckpt")
    criterion = nn.CrossEntropyLoss()
    audio_net = model.AudioModel(args.nb_class).cuda()
    video_net = model.VideoModel(args.nb_class).cuda()
    audio_net.load_state_dict(torch.load(args.a_ckpt, map_location="cpu"))
    video_net.load_state_dict(torch.load(args.v_ckpt, map_location="cpu"))
    AVDataset = DataSet.AudioVideoDataset(args.train_dataset, 32)
    AVDataloader = DataLoader(AVDataset, batch_size=args.batch_size, num_workers=args.nb_workers, shuffle=True,
                             drop_last=True, pin_memory=True)

    test_ADataset = DataSet.AudioTestDataset(args.test_dataset)
    test_ADataloader = DataLoader(test_ADataset, batch_size=args.batch_size, num_workers=args.nb_workers, shuffle=True,
                             drop_last=False, pin_memory=True)
    test_VDataset = DataSet.VideoTestDataset(args.test_dataset, num_frames=32)
    test_VDataloader = DataLoader(test_VDataset, batch_size=1, num_workers=args.nb_workers, shuffle=True,
                                  drop_last=False, pin_memory=True)
    print("DataSet:{}, DataLoader:{}".format(len(AVDataset), len(AVDataloader)))
    audio_net.eval()
    video_net.eval()
    idx_value = np.zeros(512)
    running_audio_loss, running_video_loss, running_dist_loss = 0., 0., 0.
    count = 0
    correct = 0
    permute_num=100
    for _, data in enumerate(AVDataloader):
        video, audio, label = data['video'].cuda(), data['audio'].cuda(), data['label'].cuda()
        with torch.no_grad():
            _, y_video = video_net(video)
            h_audio, _ = audio_net(audio)
        for i in range(512):
            loss = 0
            for j in range(permute_num):
                perm_index = torch.randperm(video.shape[0])
                h_audio_copy = h_audio.detach().clone()
                h_audio_copy[:, i]=h_audio_copy[perm_index, i]
                with torch.no_grad():
                    y_audio = audio_net(h_audio_copy, fc=True)
                dist_loss = torch.mean(torch.abs(y_audio-y_video))
                loss+=dist_loss.item()
            idx_value[i]+=loss/permute_num
        count += 1
    print(count)
    print(idx_value/count)
    np.save("audio_permute.npy", idx_value)
